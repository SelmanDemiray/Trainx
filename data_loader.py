import numpy as np
import struct
import os

def load_mnist_data(data_path):
    """
    Loads the MNIST dataset for training and testing.
    
    Args:
        data_path: Path to directory containing MNIST files
        
    Returns:
        train_images, train_labels, test_images, test_labels
    """
    # File paths
    train_images_path = os.path.join(data_path, 'train-images-idx3-ubyte')
    train_labels_path = os.path.join(data_path, 'train-labels-idx1-ubyte')
    test_images_path = os.path.join(data_path, 't10k-images-idx3-ubyte')
    test_labels_path = os.path.join(data_path, 't10k-labels-idx1-ubyte')
    
    # Check if files exist
    for path in [train_images_path, train_labels_path, test_images_path, test_labels_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"MNIST file not found: {path}")
    
    # Load training data
    train_images = load_images(train_images_path)
    train_labels = load_labels(train_labels_path)
    
    # Load test data
    test_images = load_images(test_images_path)
    test_labels = load_labels(test_labels_path)
    
    # Verify shapes match
    if train_images.shape[1] != train_labels.shape[1]:
        raise ValueError(f"Training data mismatch: images have {train_images.shape[1]} samples but labels have {train_labels.shape[1]}")
    
    if test_images.shape[1] != test_labels.shape[1]:
        raise ValueError(f"Test data mismatch: images have {test_images.shape[1]} samples but labels have {test_labels.shape[1]}")
    
    # Verify one-hot encoding
    if not np.all(np.sum(train_labels, axis=0) == 1.0):
        raise ValueError("Training labels are not properly one-hot encoded")
    
    if not np.all(np.sum(test_labels, axis=0) == 1.0):
        raise ValueError("Test labels are not properly one-hot encoded")
    
    return train_images, train_labels, test_images, test_labels

def load_images(filepath):
    """Load MNIST images from binary file."""
    with open(filepath, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, rows * cols).T
        return images.astype(np.float32) / 255.0

def load_labels(filepath):
    """Load MNIST labels from binary file and convert to one-hot."""
    with open(filepath, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        # Convert to one-hot encoding
        one_hot = np.zeros((10, num_labels))
        one_hot[labels, np.arange(num_labels)] = 1.0
        return one_hot
