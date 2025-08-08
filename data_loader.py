import numpy as np
import struct
import os
import pickle

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

def load_emnist_data(data_path, dataset_type='letters'):
    """
    Loads the EMNIST dataset for training and testing.
    
    Args:
        data_path: Path to directory containing EMNIST files
        dataset_type: Type of EMNIST dataset ('letters', 'digits', 'balanced', etc.)
        
    Returns:
        train_images, train_labels, test_images, test_labels
    """
    # File paths
    train_images_path = os.path.join(data_path, f'emnist-{dataset_type}-train-images-idx3-ubyte')
    train_labels_path = os.path.join(data_path, f'emnist-{dataset_type}-train-labels-idx1-ubyte')
    test_images_path = os.path.join(data_path, f'emnist-{dataset_type}-test-images-idx3-ubyte')
    test_labels_path = os.path.join(data_path, f'emnist-{dataset_type}-test-labels-idx1-ubyte')
    
    # Check if files exist
    for path in [train_images_path, train_labels_path, test_images_path, test_labels_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"EMNIST file not found: {path}")
    
    # Load training data
    train_images = load_images(train_images_path)
    
    # Get number of classes based on dataset type
    num_classes = get_emnist_classes(dataset_type)
    
    train_labels = load_labels(train_labels_path, num_classes)
    
    # Load test data
    test_images = load_images(test_images_path)
    test_labels = load_labels(test_labels_path, num_classes)
    
    # EMNIST has different orientation - transpose images to match MNIST format
    train_images = reshape_and_transpose_images(train_images)
    test_images = reshape_and_transpose_images(test_images)
    
    return train_images, train_labels, test_images, test_labels

def get_emnist_classes(dataset_type):
    """Get number of classes for different EMNIST datasets."""
    if dataset_type == 'digits':
        return 10
    elif dataset_type == 'letters':
        return 26
    elif dataset_type == 'balanced':
        return 47
    elif dataset_type == 'mnist':
        return 10
    elif dataset_type == 'byclass':
        return 62
    elif dataset_type == 'bymerge':
        return 47
    else:
        raise ValueError(f"Unknown EMNIST dataset type: {dataset_type}")

def reshape_and_transpose_images(images):
    """Reshape and transpose EMNIST images to match MNIST format."""
    # Reshape to 28x28 images, transpose, then flatten back
    n_samples = images.shape[1]
    images = images.reshape(784, n_samples)
    return images

def load_cifar10_data(data_path):
    """
    Loads the CIFAR-10 dataset for training and testing.
    
    Args:
        data_path: Path to directory containing CIFAR-10 files
        
    Returns:
        train_images, train_labels, test_images, test_labels
    """
    cifar_path = os.path.join(data_path, 'cifar-10-batches-py')
    
    # Load training data from batches
    train_images = []
    train_labels = []
    
    for batch_id in range(1, 6):
        batch_file = os.path.join(cifar_path, f'data_batch_{batch_id}')
        with open(batch_file, 'rb') as f:
            batch_data = pickle.load(f, encoding='bytes')
        
        train_images.append(batch_data[b'data'])
        train_labels.extend(batch_data[b'labels'])
    
    # Load test data
    test_file = os.path.join(cifar_path, 'test_batch')
    with open(test_file, 'rb') as f:
        test_data = pickle.load(f, encoding='bytes')
    
    test_images = test_data[b'data']
    test_labels = test_data[b'labels']
    
    # Convert images to format compatible with our network
    train_images = np.vstack(train_images)
    test_images = np.array(test_images)
    
    # Convert labels to one-hot encoding
    train_labels = one_hot_encode(train_labels, 10)
    test_labels = one_hot_encode(test_labels, 10)
    
    # Process images for network (grayscale, reshape to match MNIST format)
    train_images = process_cifar_images(train_images)
    test_images = process_cifar_images(test_images)
    
    return train_images, train_labels, test_images, test_labels

def load_cifar100_data(data_path, use_fine_labels=True):
    """
    Loads the CIFAR-100 dataset for training and testing.
    
    Args:
        data_path: Path to directory containing CIFAR-100 files
        use_fine_labels: If True, use 100 fine-grained labels, else use 20 coarse labels
        
    Returns:
        train_images, train_labels, test_images, test_labels
    """
    cifar_path = os.path.join(data_path, 'cifar-100-python')
    
    # Load training data
    train_file = os.path.join(cifar_path, 'train')
    with open(train_file, 'rb') as f:
        train_data = pickle.load(f, encoding='bytes')
    
    # Load test data
    test_file = os.path.join(cifar_path, 'test')
    with open(test_file, 'rb') as f:
        test_data = pickle.load(f, encoding='bytes')
    
    # Extract images
    train_images = train_data[b'data']
    test_images = test_data[b'data']
    
    # Extract labels based on fine/coarse setting
    if use_fine_labels:
        train_labels = train_data[b'fine_labels']
        test_labels = test_data[b'fine_labels']
        num_classes = 100
    else:
        train_labels = train_data[b'coarse_labels']
        test_labels = test_data[b'coarse_labels']
        num_classes = 20
    
    # Convert labels to one-hot encoding
    train_labels = one_hot_encode(train_labels, num_classes)
    test_labels = one_hot_encode(test_labels, num_classes)
    
    # Process images for network
    train_images = process_cifar_images(train_images)
    test_images = process_cifar_images(test_images)
    
    return train_images, train_labels, test_images, test_labels

def process_cifar_images(images):
    """Process CIFAR images to match network input format."""
    # CIFAR images are 32x32x3, we need to convert them to grayscale 28x28
    
    # Reshape to (num_images, 3, 32, 32)
    images = images.reshape(-1, 3, 32, 32)
    
    # Convert RGB to grayscale using standard formula
    grayscale = 0.2989 * images[:, 0] + 0.5870 * images[:, 1] + 0.1140 * images[:, 2]
    
    # Resize from 32x32 to 28x28 (center crop)
    start_idx = (32 - 28) // 2
    end_idx = start_idx + 28
    grayscale = grayscale[:, start_idx:end_idx, start_idx:end_idx]
    
    # Reshape to match MNIST format (784, n_samples)
    processed_images = grayscale.reshape(grayscale.shape[0], 784).T
    
    # Normalize to 0-1
    processed_images = processed_images.astype(np.float32) / 255.0
    
    return processed_images

def one_hot_encode(labels, num_classes):
    """Convert integer labels to one-hot encoding."""
    one_hot = np.zeros((num_classes, len(labels)))
    one_hot[labels, np.arange(len(labels))] = 1.0
    return one_hot

def load_images(filepath):
    """Load images from binary file."""
    with open(filepath, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, rows * cols).T
        return images.astype(np.float32) / 255.0

def load_labels(filepath, num_classes=10):
    """Load labels from binary file and convert to one-hot."""
    with open(filepath, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        # Convert to one-hot encoding
        one_hot = np.zeros((num_classes, num_labels))
        one_hot[labels, np.arange(num_labels)] = 1.0
        return one_hot
