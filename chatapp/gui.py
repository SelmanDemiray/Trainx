import os
import threading
from typing import Optional
from PySide6 import QtCore, QtWidgets
from seq2seq import Seq2Seq, Seq2SeqConfig
from utils import load_tsv_pairs, build_vocab_from_pairs, encode_pairs, Trainer, save_full_package, load_package, Vocab

MODELS_DIR = r"d:\trainerx\models"

class TrainWorker(QtCore.QObject):
    progress = QtCore.Signal(str)
    finished = QtCore.Signal(dict)

    def __init__(self, model: Seq2Seq, dataset, epochs: int):
        super().__init__()
        self.model = model
        self.dataset = dataset
        self.epochs = epochs
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        trainer = Trainer(self.model, self.dataset, shuffle=True)
        def log_cb(msg): self.progress.emit(msg)
        def stop_fn(): return self._stop
        result = trainer.run(self.epochs, log_cb=log_cb, stop_fn=stop_fn)
        self.finished.emit(result)

class ChatApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Enterprise Chat Seq2Seq (Homemade)")
        self.resize(1100, 700)
        os.makedirs(MODELS_DIR, exist_ok=True)

        self.vocab: Optional[Vocab] = None
        self.model: Optional[Seq2Seq] = None
        self.dataset_pairs = []
        self.encoded_dataset = []

        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)

        self._init_dataset_tab()
        self._init_training_tab()
        self._init_chat_tab()
        self._init_models_tab()

    def _init_dataset_tab(self):
        w = QtWidgets.QWidget(); layout = QtWidgets.QVBoxLayout(w)
        # File loader
        hl = QtWidgets.QHBoxLayout()
        self.ds_path = QtWidgets.QLineEdit()
        self.ds_path.setPlaceholderText("Path to TSV dataset: input[TAB]response per line")
        btn_browse = QtWidgets.QPushButton("Browse")
        btn_browse.clicked.connect(self._browse_dataset)
        hl.addWidget(self.ds_path); hl.addWidget(btn_browse)
        layout.addLayout(hl)
        # Options
        opts = QtWidgets.QHBoxLayout()
        self.max_len = QtWidgets.QSpinBox(); self.max_len.setRange(8, 256); self.max_len.setValue(32)
        self.min_freq = QtWidgets.QSpinBox(); self.min_freq.setRange(1, 10); self.min_freq.setValue(1)
        self.limit = QtWidgets.QSpinBox(); self.limit.setRange(0, 100000); self.limit.setValue(0)
        opts.addWidget(QtWidgets.QLabel("Max len")); opts.addWidget(self.max_len)
        opts.addWidget(QtWidgets.QLabel("Min freq")); opts.addWidget(self.min_freq)
        opts.addWidget(QtWidgets.QLabel("Limit(0=all)")); opts.addWidget(self.limit)
        layout.addLayout(opts)
        # Actions
        btn_load = QtWidgets.QPushButton("Load Dataset")
        btn_load.clicked.connect(self._load_dataset)
        layout.addWidget(btn_load)
        # Stats
        self.ds_stats = QtWidgets.QTextEdit(); self.ds_stats.setReadOnly(True)
        layout.addWidget(self.ds_stats)
        self.tabs.addTab(w, "Dataset")

    def _init_training_tab(self):
        w = QtWidgets.QWidget(); layout = QtWidgets.QVBoxLayout(w)
        g = QtWidgets.QGridLayout()
        self.hid = QtWidgets.QSpinBox(); self.hid.setRange(32, 1024); self.hid.setValue(128)
        self.lr = QtWidgets.QDoubleSpinBox(); self.lr.setRange(1e-4, 1.0); self.lr.setSingleStep(0.001); self.lr.setValue(0.01)
        self.mom = QtWidgets.QDoubleSpinBox(); self.mom.setRange(0.0, 0.999); self.mom.setDecimals(3); self.mom.setValue(0.9)
        self.wd = QtWidgets.QDoubleSpinBox(); self.wd.setRange(0.0, 1e-1); self.wd.setDecimals(6); self.wd.setValue(1e-4)
        self.epochs = QtWidgets.QSpinBox(); self.epochs.setRange(1, 1000); self.epochs.setValue(5)
        g.addWidget(QtWidgets.QLabel("Hidden size"), 0, 0); g.addWidget(self.hid, 0, 1)
        g.addWidget(QtWidgets.QLabel("LR"), 0, 2); g.addWidget(self.lr, 0, 3)
        g.addWidget(QtWidgets.QLabel("Momentum"), 1, 0); g.addWidget(self.mom, 1, 1)
        g.addWidget(QtWidgets.QLabel("Weight decay"), 1, 2); g.addWidget(self.wd, 1, 3)
        g.addWidget(QtWidgets.QLabel("Epochs"), 2, 0); g.addWidget(self.epochs, 2, 1)
        layout.addLayout(g)
        # Buttons
        hb = QtWidgets.QHBoxLayout()
        self.btn_init = QtWidgets.QPushButton("Init Model")
        self.btn_train = QtWidgets.QPushButton("Start Training")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_save = QtWidgets.QPushButton("Save Model")
        self.btn_init.clicked.connect(self._init_model)
        self.btn_train.clicked.connect(self._start_training)
        self.btn_stop.clicked.connect(self._stop_training)
        self.btn_save.clicked.connect(self._save_model)
        hb.addWidget(self.btn_init); hb.addWidget(self.btn_train); hb.addWidget(self.btn_stop); hb.addWidget(self.btn_save)
        layout.addLayout(hb)
        # Progress and log
        self.prog = QtWidgets.QProgressBar(); self.prog.setRange(0, 0); self.prog.setVisible(False)
        self.log = QtWidgets.QTextEdit(); self.log.setReadOnly(True)
        layout.addWidget(self.prog); layout.addWidget(self.log)
        self.tabs.addTab(w, "Training")
        self._worker = None
        self._thread = None

    def _init_chat_tab(self):
        w = QtWidgets.QWidget(); layout = QtWidgets.QVBoxLayout(w)
        # Load model dir
        hl = QtWidgets.QHBoxLayout()
        self.model_dir = QtWidgets.QLineEdit(); self.model_dir.setPlaceholderText("Model directory")
        btn_browse = QtWidgets.QPushButton("Browse Model")
        btn_browse.clicked.connect(self._browse_model_dir)
        btn_load = QtWidgets.QPushButton("Load")
        btn_load.clicked.connect(self._load_model_from_dir)
        hl.addWidget(self.model_dir); hl.addWidget(btn_browse); hl.addWidget(btn_load)
        layout.addLayout(hl)
        # Chat UI
        self.chat_view = QtWidgets.QTextEdit(); self.chat_view.setReadOnly(True)
        hl2 = QtWidgets.QHBoxLayout()
        self.chat_input = QtWidgets.QLineEdit(); self.chat_input.setPlaceholderText("Type your message...")
        btn_send = QtWidgets.QPushButton("Send")
        btn_send.clicked.connect(self._chat_send)
        hl2.addWidget(self.chat_input); hl2.addWidget(btn_send)
        layout.addWidget(self.chat_view); layout.addLayout(hl2)
        self.tabs.addTab(w, "Chat")

    def _init_models_tab(self):
        w = QtWidgets.QWidget(); layout = QtWidgets.QVBoxLayout(w)
        self.model_list = QtWidgets.QListWidget()
        btn_refresh = QtWidgets.QPushButton("Refresh")
        btn_refresh.clicked.connect(self._refresh_models)
        btn_use = QtWidgets.QPushButton("Use Selected")
        btn_use.clicked.connect(self._use_selected_model)
        layout.addWidget(self.model_list)
        layout.addWidget(btn_refresh)
        layout.addWidget(btn_use)
        self.tabs.addTab(w, "Models")
        self._refresh_models()

    # Dataset actions
    def _browse_dataset(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select TSV dataset", "", "TSV Files (*.tsv *.txt);;All Files (*.*)")
        if path:
            self.ds_path.setText(path)

    def _load_dataset(self):
        path = self.ds_path.text().strip()
        if not path or not os.path.exists(path):
            QtWidgets.QMessageBox.warning(self, "Error", "Invalid dataset path.")
            return
        limit = self.limit.value() or None
        pairs = load_tsv_pairs(path, limit=limit)
        if not pairs:
            QtWidgets.QMessageBox.warning(self, "Error", "No valid pairs found.")
            return
        vocab = build_vocab_from_pairs(pairs, min_freq=self.min_freq.value())
        enc = encode_pairs(pairs, vocab, max_len=self.max_len.value())
        self.dataset_pairs = pairs
        self.encoded_dataset = enc
        self.vocab = vocab
        self.ds_stats.setPlainText(f"Loaded pairs: {len(pairs)}\nVocab size: {len(vocab.id2token)}\nMax len: {self.max_len.value()}")

    # Training actions
    def _init_model(self):
        if not self.vocab:
            QtWidgets.QMessageBox.warning(self, "Error", "Load dataset first.")
            return
        cfg = Seq2SeqConfig(
            vocab_size=len(self.vocab.id2token),
            hidden_size=self.hid.value(),
            max_len=self.max_len.value(),
            sos_id=self.vocab.token2id["<sos>"],
            eos_id=self.vocab.token2id["<eos>"],
            pad_id=self.vocab.token2id["<pad>"],
            lr=self.lr.value(),
            momentum=self.mom.value(),
            weight_decay=self.wd.value()
        )
        self.model = Seq2Seq(cfg)
        self._log("Model initialized.")

    def _start_training(self):
        if not self.model or not self.encoded_dataset:
            QtWidgets.QMessageBox.warning(self, "Error", "Initialize model and load dataset first.")
            return
        self._log("Starting training...")
        self.prog.setVisible(True)
        self._worker = TrainWorker(self.model, self.encoded_dataset, self.epochs.value())
        self._thread = QtCore.QThread()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._log)
        self._worker.finished.connect(self._train_finished)
        self._thread.start()

    def _stop_training(self):
        if self._worker:
            self._worker.stop()
            self._log("Stop requested.")

    def _train_finished(self, result: dict):
        self.prog.setVisible(False)
        self._log(f"Training finished. {'Stopped early' if result.get('stopped') else 'Completed'}.")
        if self._thread:
            self._thread.quit()
            self._thread.wait()
            self._thread = None
            self._worker = None

    def _save_model(self):
        if not self.model or not self.vocab:
            QtWidgets.QMessageBox.warning(self, "Error", "No model to save.")
            return
        out_dir = save_full_package(self.model, self.vocab, MODELS_DIR, extra_meta={"note": "Saved from GUI"})
        self._log(f"Saved model to: {out_dir}")
        self._refresh_models()

    # Chat actions
    def _browse_model_dir(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Model Directory", MODELS_DIR)
        if d:
            self.model_dir.setText(d)

    def _load_model_from_dir(self):
        d = self.model_dir.text().strip()
        if not d or not os.path.exists(d):
            QtWidgets.QMessageBox.warning(self, "Error", "Invalid model directory.")
            return
        self.model, self.vocab = load_package(d)
        self._log("Model loaded for chat.")

    def _chat_send(self):
        if not self.model or not self.vocab:
            QtWidgets.QMessageBox.warning(self, "Error", "Load a model first.")
            return
        text = self.chat_input.text().strip()
        if not text:
            return
        self.chat_view.append(f"You: {text}")
        src_ids = self.vocab.encode(text, add_sos_eos=True, max_len=self.model.cfg.max_len)[1:]  # remove SOS for encoder
        out_ids = self.model.generate(src_ids, max_new_tokens=self.model.cfg.max_len)
        reply = self.vocab.decode(out_ids)
        self.chat_view.append(f"Bot: {reply}")
        self.chat_input.clear()

    # Models tab
    def _refresh_models(self):
        self.model_list.clear()
        if not os.path.exists(MODELS_DIR):
            return
        for name in sorted(os.listdir(MODELS_DIR)):
            d = os.path.join(MODELS_DIR, name)
            if os.path.isdir(d) and os.path.exists(os.path.join(d, "config.json")):
                self.model_list.addItem(d)

    def _use_selected_model(self):
        items = self.model_list.selectedItems()
        if not items:
            return
        d = items[0].text()
        self.model_dir.setText(d)
        self._load_model_from_dir()

    def _log(self, msg: str):
        self.log.append(msg)
        self.log.ensureCursorVisible()
