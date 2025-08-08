import os
import urllib.request
import tarfile
import gzip
import shutil
import numpy as np
from tqdm import tqdm
import pickle
import struct

class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    """Download file with progress bar."""
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, 
                                  reporthook=t.update_to)

def _download_with_headers(url, output_path, headers=None, callback=None, chunk_size=1024 * 256):
    """Stream download with optional headers and basic progress."""
    req = urllib.request.Request(url, headers=headers or {"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(output_path, "wb") as out:
        total = resp.length or 0
        read = 0
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out.write(chunk)
            read += len(chunk)
            if callback and total:
                callback(f"Downloading {os.path.basename(output_path)} {read // 1024}KB/{total // 1024}KB")

def _is_gzip_file(path):
    """Quickly verify if file has gzip magic header."""
    try:
        with open(path, "rb") as f:
            magic = f.read(2)
        return magic == b"\x1f\x8b"
    except Exception:
        return False

class DatasetDownloader:
    """Handles downloading and extracting datasets."""
    
    BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
    
    def __init__(self):
        # Create datasets directory if it doesn't exist
        if not os.path.exists(self.BASE_DIR):
            os.makedirs(self.BASE_DIR)
    
    def download_mnist(self, callback=None):
        """Download MNIST dataset."""
        mnist_dir = os.path.join(self.BASE_DIR, "mnist")
        if not os.path.exists(mnist_dir):
            os.makedirs(mnist_dir)
        
        # MNIST URLs
        base_url = "http://yann.lecun.com/exdb/mnist/"
        files = [
            'train-images-idx3-ubyte.gz', 
            'train-labels-idx1-ubyte.gz', 
            't10k-images-idx3-ubyte.gz', 
            't10k-labels-idx1-ubyte.gz'
        ]
        
        # Download and extract each file
        for file in files:
            output_file = os.path.join(mnist_dir, file)
            if not os.path.exists(output_file):
                url = base_url + file
                if callback: callback(f"Downloading {file}...")
                download_url(url, output_file)
            
            # Extract .gz file
            output_path = os.path.join(mnist_dir, file[:-3])  # Remove .gz
            if not os.path.exists(output_path):
                if callback: callback(f"Extracting {file}...")
                with gzip.open(output_file, 'rb') as f_in:
                    with open(output_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
        
        return mnist_dir
    
    def download_emnist(self, callback=None):
        """Download all EMNIST splits from GitHub gzip files and extract."""
        emnist_dir = os.path.join(self.BASE_DIR, "emnist")
        os.makedirs(emnist_dir, exist_ok=True)

        extracted_marker = os.path.join(emnist_dir, "gzip_ready.marker")
        if os.path.exists(extracted_marker):
            if callback: callback("EMNIST datasets already prepared.")
            return emnist_dir

        # Candidate URL bases to handle Git LFS pointers and bandwidth/CDN paths.
        owner = "aurelienduarte"
        repo = "emnist"
        branch = "master"
        raw_base = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/gzip"
        media_base = f"https://media.githubusercontent.com/media/{owner}/{repo}/{branch}/gzip"
        blob_raw_base = f"https://github.com/{owner}/{repo}/raw/{branch}/gzip"

        splits = ['byclass', 'bymerge', 'balanced', 'letters', 'digits', 'mnist']
        parts = [
            "train-images-idx3-ubyte",
            "train-labels-idx1-ubyte",
            "test-images-idx3-ubyte",
            "test-labels-idx1-ubyte",
        ]

        def ensure_file(fname_no_gz: str):
            """Ensure a single file (without .gz) exists by downloading and extracting its .gz."""
            gz_name = fname_no_gz + ".gz"
            out_path = os.path.join(emnist_dir, fname_no_gz)
            gz_path = os.path.join(emnist_dir, gz_name)

            if os.path.exists(out_path):
                return  # already extracted

            # Try multiple sources; validate gzip magic
            candidates = [
                f"{raw_base}/{gz_name}",
                f"{media_base}/{gz_name}",
                f"{blob_raw_base}/{gz_name}",
            ]

            # Remove stale/invalid previous download
            if os.path.exists(gz_path) and not _is_gzip_file(gz_path):
                try: os.remove(gz_path)
                except: pass

            success = False
            last_err = None
            for idx, url in enumerate(candidates, start=1):
                try:
                    # Use simple urlretrieve first, then header-based stream as fallback
                    if callback: callback(f"Downloading {gz_name} (source {idx}) ...")
                    try:
                        download_url(url, gz_path)
                    except Exception:
                        _download_with_headers(
                            url, gz_path,
                            headers={"User-Agent": "Mozilla/5.0", "Accept": "application/octet-stream"},
                            callback=callback
                        )
                    if _is_gzip_file(gz_path):
                        success = True
                        break
                    else:
                        # Not a real gzip (likely a Git LFS pointer). Try next source.
                        if callback: callback(f"{gz_name} is not a valid gzip, trying next source...")
                        try: os.remove(gz_path)
                        except: pass
                except Exception as e:
                    last_err = e
                    if callback: callback(f"Download failed from {url}: {e}")

            if not success:
                raise RuntimeError(f"Failed to download a valid gzip for {gz_name}. Last error: {last_err}")

            # Extract
            if callback: callback(f"Extracting {gz_name} ...")
            with gzip.open(gz_path, 'rb') as f_in, open(out_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        for split in splits:
            for part in parts:
                fname_no_gz = f"emnist-{split}-{part}"
                ensure_file(fname_no_gz)

        with open(extracted_marker, "w") as f:
            f.write("EMNIST gzip datasets ready")

        if callback: callback("All EMNIST splits downloaded and extracted.")
        return emnist_dir
    
    def _convert_emnist_format(self, emnist_dir, callback=None):
        """Convert EMNIST .mat files to our binary format."""
        # This path kept for reference; not used when downloading from gzip repo.
        try:
            from scipy.io import loadmat
        except ImportError:
            raise ImportError("scipy is required to process EMNIST dataset. Install it with: pip install scipy")
        
        splits = ['byclass', 'bymerge', 'balanced', 'letters', 'digits', 'mnist']
        
        for split in splits:
            if callback: callback(f"Converting {split} split...")
            mat_file = os.path.join(emnist_dir, f"matlab/emnist-{split}.mat")
            if os.path.exists(mat_file):
                try:
                    data = loadmat(mat_file)
                    train_images = data['dataset'][0][0][0][0][0][0].T
                    train_labels = data['dataset'][0][0][0][0][0][1]
                    test_images = data['dataset'][0][0][1][0][0][0].T
                    test_labels = data['dataset'][0][0][1][0][0][1]
                    self._write_idx_file(os.path.join(emnist_dir, f"emnist-{split}-train-images-idx3-ubyte"), 
                                        train_images, data_type=0x0803)
                    self._write_idx_file(os.path.join(emnist_dir, f"emnist-{split}-train-labels-idx1-ubyte"), 
                                        train_labels, data_type=0x0801)
                    self._write_idx_file(os.path.join(emnist_dir, f"emnist-{split}-test-images-idx3-ubyte"), 
                                        test_images, data_type=0x0803)
                    self._write_idx_file(os.path.join(emnist_dir, f"emnist-{split}-test-labels-idx1-ubyte"), 
                                        test_labels, data_type=0x0801)
                except Exception as e:
                    if callback: callback(f"Error converting {split} split: {str(e)}")

    def _write_idx_file(self, filename, data, data_type):
        """Write data to IDX file format."""
        with open(filename, 'wb') as f:
            if data_type == 0x0803:  # Images
                f.write(struct.pack('>IIII', data_type, len(data), 28, 28))
                f.write(data.astype(np.uint8).tobytes())
            elif data_type == 0x0801:  # Labels
                f.write(struct.pack('>II', data_type, len(data)))
                f.write(data.astype(np.uint8).tobytes())
    
    def download_cifar10(self, callback=None):
        """Download CIFAR-10 dataset."""
        cifar10_dir = os.path.join(self.BASE_DIR, "cifar-10")
        if not os.path.exists(cifar10_dir):
            os.makedirs(cifar10_dir)
        
        # CIFAR-10 URL
        url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        tar_file = os.path.join(cifar10_dir, "cifar-10-python.tar.gz")
        
        # Download if not exists
        if not os.path.exists(tar_file):
            if callback: callback("Downloading CIFAR-10 dataset...")
            download_url(url, tar_file)
        
        # Extract if not already done
        extracted_marker = os.path.join(cifar10_dir, "extracted.marker")
        if not os.path.exists(extracted_marker):
            if callback: callback("Extracting CIFAR-10 dataset...")
            with tarfile.open(tar_file, "r:gz") as tar:
                tar.extractall(path=cifar10_dir)
            
            # Create marker file
            with open(extracted_marker, 'w') as f:
                f.write("CIFAR-10 extraction completed")
        
        return cifar10_dir
    
    def download_cifar100(self, callback=None):
        """Download CIFAR-100 dataset."""
        cifar100_dir = os.path.join(self.BASE_DIR, "cifar-100")
        if not os.path.exists(cifar100_dir):
            os.makedirs(cifar100_dir)
        
        # CIFAR-100 URL
        url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
        tar_file = os.path.join(cifar100_dir, "cifar-100-python.tar.gz")
        
        # Download if not exists
        if not os.path.exists(tar_file):
            if callback: callback("Downloading CIFAR-100 dataset...")
            download_url(url, tar_file)
        
        # Extract if not already done
        extracted_marker = os.path.join(cifar100_dir, "extracted.marker")
        if not os.path.exists(extracted_marker):
            if callback: callback("Extracting CIFAR-100 dataset...")
            with tarfile.open(tar_file, "r:gz") as tar:
                tar.extractall(path=cifar100_dir)
            
            # Create marker file
            with open(extracted_marker, 'w') as f:
                f.write("CIFAR-100 extraction completed")
        
        return cifar100_dir
