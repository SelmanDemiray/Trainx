import os
import json
import pickle
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

@dataclass
class Seq2SeqConfig:
    vocab_size: int
    hidden_size: int = 128
    max_len: int = 32
    sos_id: int = 1
    eos_id: int = 2
    pad_id: int = 0
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    teacher_forcing: float = 0.75
    init_sigma: float = 0.1
    seed: int = 42

class Seq2Seq:
    def __init__(self, cfg: Seq2SeqConfig):
        self.cfg = cfg
        rng = np.random.RandomState(cfg.seed)
        V, H = cfg.vocab_size, cfg.hidden_size

        # Encoder: h_t = tanh(Whh_e h_{t-1} + Wih_e[:, x_t] + b_e)
        self.Wih_e = rng.randn(H, V) * cfg.init_sigma
        self.Whh_e = rng.randn(H, H) * cfg.init_sigma
        self.b_e   = np.zeros((H, 1))

        # Decoder: h_t = tanh(Whh_d h_{t-1} + Wih_d[:, y_{t-1}] + b_d), logits = Wout h_t + bout
        self.Wih_d = rng.randn(H, V) * cfg.init_sigma
        self.Whh_d = rng.randn(H, H) * cfg.init_sigma
        self.b_d   = np.zeros((H, 1))
        self.Wout  = rng.randn(V, H) * cfg.init_sigma
        self.bout  = np.zeros((V, 1))

        # Velocities for momentum
        self._vel = {k: np.zeros_like(v) for k, v in self.params().items()}

    def params(self) -> Dict[str, np.ndarray]:
        return {
            "Wih_e": self.Wih_e, "Whh_e": self.Whh_e, "b_e": self.b_e,
            "Wih_d": self.Wih_d, "Whh_d": self.Whh_d, "b_d": self.b_d,
            "Wout": self.Wout, "bout": self.bout
        }

    def zero_like_grads(self) -> Dict[str, np.ndarray]:
        return {k: np.zeros_like(v) for k, v in self.params().items()}

    @staticmethod
    def _tanh(x): return np.tanh(x)

    @staticmethod
    def _dtanh(y): return 1.0 - y * y  # derivative wrt tanh output

    @staticmethod
    def _softmax_logits(logits: np.ndarray) -> np.ndarray:
        # logits: (V, T)
        m = np.max(logits, axis=0, keepdims=True)
        ex = np.exp(logits - m)
        return ex / np.sum(ex, axis=0, keepdims=True)

    @staticmethod
    def _nll_loss(probs: np.ndarray, target_ids: List[int]) -> Tuple[float, np.ndarray]:
        # probs: (V, T), targets length T
        T = len(target_ids)
        eps = 1e-12
        idx = (target_ids, list(range(T)))
        p = probs[idx]
        loss = -np.sum(np.log(p + eps))
        dlogits = probs
        dlogits[idx] -= 1.0
        return loss, dlogits

    def encode(self, x_ids: List[int]) -> Tuple[List[np.ndarray], np.ndarray, Dict]:
        H = self.cfg.hidden_size
        h_prev = np.zeros((H, 1))
        hs = []
        preacts = []
        x_cols = []
        for token in x_ids[:self.cfg.max_len]:
            x_cols.append(token)
            z = self.Whh_e @ h_prev + self.Wih_e[:, [token]] + self.b_e
            h = self._tanh(z)
            hs.append(h)
            preacts.append(z)
            h_prev = h
        cache = {"hs": hs, "preacts": preacts, "x_cols": x_cols}
        h_last = h_prev if hs else np.zeros((H, 1))
        return hs, h_last, cache

    def decode_train(self, y_in_ids: List[int], y_tgt_ids: List[int], h0: np.ndarray) -> Tuple[float, Dict]:
        H = self.cfg.hidden_size
        T = min(len(y_in_ids), self.cfg.max_len)
        h_prev = h0.copy()
        hs, preacts, y_cols, logits = [], [], [], []
        for t in range(T):
            y_cols.append(y_in_ids[t])
            z = self.Whh_d @ h_prev + self.Wih_d[:, [y_in_ids[t]]] + self.b_d
            h = self._tanh(z)
            o = self.Wout @ h + self.bout
            hs.append(h)
            preacts.append(z)
            logits.append(o)
            h_prev = h
        if T == 0:
            return 0.0, {"T": 0}

        logits_mat = np.concatenate(logits, axis=1)  # (V, T)
        probs = self._softmax_logits(logits_mat)
        loss, dlogits = self._nll_loss(probs, y_tgt_ids[:T])

        cache = {
            "hs": hs, "preacts": preacts, "y_cols": y_cols, "logits": logits_mat,
            "dlogits": dlogits, "T": T, "h0": h0
        }
        return loss, cache

    def bptt(self, enc_cache: Dict, dec_cache: Dict) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        # Returns grads dict and gradient wrt encoder final hidden (to propagate into encoder)
        grads = self.zero_like_grads()
        T = dec_cache["T"]
        if T == 0:
            return grads, np.zeros_like(self.b_e)

        hs = dec_cache["hs"]; pre = dec_cache["preacts"]; y_cols = dec_cache["y_cols"]
        dlogits = dec_cache["dlogits"]
        # Decoder output layer grads
        H = self.cfg.hidden_size
        dWout = np.zeros_like(self.Wout)
        dbout = np.zeros_like(self.bout)
        dh_next = np.zeros((H, 1))

        for t in reversed(range(T)):
            h_t = hs[t]
            dO_t = dlogits[:, [t]]  # (V,1)
            dWout += dO_t @ h_t.T
            dbout += dO_t
            dh = self.Wout.T @ dO_t + dh_next  # from output to h plus through time
            dz = dh * self._dtanh(h_t)  # (H,1)
            grads["Whh_d"] += dz @ (dec_cache["h0"].T if t == 0 else hs[t-1].T)
            grads["Wih_d"][:, [y_cols[t]]] += dz
            grads["b_d"] += dz
            dh_next = self.Whh_d.T @ dz

        grads["Wout"] += dWout
        grads["bout"] += dbout
        d_h0 = dh_next  # gradient wrt decoder initial hidden

        # Backprop into encoder through h_last
        enc_hs = enc_cache["hs"]; enc_pre = enc_cache["preacts"]; x_cols = enc_cache["x_cols"]
        L = len(enc_hs)
        dh_next = d_h0.copy()
        for t in reversed(range(L)):
            h_t = enc_hs[t]
            dz = dh_next * self._dtanh(h_t)
            h_prev_T = enc_hs[t-1].T if t > 0 else np.zeros((1, self.cfg.hidden_size))
            grads["Whh_e"] += dz @ (enc_hs[t-1].T if t > 0 else np.zeros_like(self.Whh_e[:, :1]).T)
            grads["Wih_e"][:, [x_cols[t]]] += dz
            grads["b_e"] += dz
            dh_next = self.Whh_e.T @ dz

        return grads, d_h0

    def update(self, grads: Dict[str, np.ndarray]):
        # Weight decay and momentum SGD
        for k, w in self.params().items():
            g = grads[k] + self.cfg.weight_decay * w
            # Gradient clipping
            if self.cfg.grad_clip is not None:
                norm = np.linalg.norm(g)
                if norm > self.cfg.grad_clip and norm > 0:
                    g = g * (self.cfg.grad_clip / norm)
            v = self._vel[k] = self.cfg.momentum * self._vel[k] - self.cfg.lr * g
            w += v

    def train_step(self, src_ids: List[int], tgt_ids: List[int]) -> float:
        # Teacher forcing: decoder input starts with SOS, targets end with EOS
        y_in = [self.cfg.sos_id] + tgt_ids[:-1]
        y_tg = tgt_ids
        # Encode
        _, h_last, enc_cache = self.encode(src_ids)
        # Decode + loss
        loss, dec_cache = self.decode_train(y_in, y_tg, h_last)
        # Backprop
        grads, _ = self.bptt(enc_cache, dec_cache)
        self.update(grads)
        return float(loss)

    def generate(self, src_ids: List[int], max_new_tokens: Optional[int] = None) -> List[int]:
        _, h_last, _ = self.encode(src_ids)
        H = self.cfg.hidden_size
        h_prev = h_last
        out = []
        token = self.cfg.sos_id
        T = max_new_tokens or self.cfg.max_len
        for _ in range(T):
            z = self.Whh_d @ h_prev + self.Wih_d[:, [token]] + self.b_d
            h = self._tanh(z)
            logits = self.Wout @ h + self.bout
            probs = self._softmax_logits(logits)
            token = int(np.argmax(probs[:, 0]))
            if token == self.cfg.eos_id:
                break
            out.append(token)
            h_prev = h
        return out

    def save(self, out_dir: str, meta: Optional[Dict] = None):
        os.makedirs(out_dir, exist_ok=True)
        # npz weights
        np.savez(os.path.join(out_dir, "weights.npz"), **self.params())
        # json config
        with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(self.cfg.__dict__, f, indent=2)
        # pickle full state
        state = {"cfg": self.cfg.__dict__, "params": {k: v for k, v in self.params().items()}}
        with open(os.path.join(out_dir, "model.pkl"), "wb") as f:
            pickle.dump(state, f)
        # txt metadata/readme
        with open(os.path.join(out_dir, "readme.txt"), "w", encoding="utf-8") as f:
            f.write("Seq2Seq chat model\n")
            if meta:
                for k, v in meta.items():
                    f.write(f"{k}: {v}\n")

    @staticmethod
    def load(model_dir: str) -> "Seq2Seq":
        with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)
        cfg = Seq2SeqConfig(**cfg_dict)
        model = Seq2Seq(cfg)
        data = np.load(os.path.join(model_dir, "weights.npz"))
        for k in model.params().keys():
            getattr(model, k)[:] = data[k]
        return model
