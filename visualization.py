import matplotlib.pyplot as plt
import numpy as np

def show_weights(weights):
    """Display weight matrices as grayscale images."""
    nfields = weights.shape[0]
    nsqrt = int(np.sqrt(nfields))
    nrows = nsqrt
    ncols = int(np.ceil(nfields / nrows))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 12))
    if nrows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(nfields):
        row = i // ncols
        col = i % ncols
        
        if row < nrows and col < ncols:
            weight_image = weights[i].reshape(28, 28)
            axes[row, col].imshow(weight_image, cmap='gray')
            axes[row, col].axis('off')
    
    # Hide empty subplots
    for i in range(nfields, nrows * ncols):
        row = i // ncols
        col = i % ncols
        if row < nrows and col < ncols:
            axes[row, col].axis('off')
    
    plt.suptitle('Receptive Fields (Weights)')
    plt.tight_layout()
    plt.show()

def show_sample_images(images, labels):
    """Display sample images from each class."""
    fig, axes = plt.subplots(10, 10, figsize=(10, 10))
    
    # Get 10 samples from each class
    for digit in range(10):
        digit_indices = np.where(np.argmax(labels, axis=0) == digit)[0]
        selected_indices = np.random.choice(digit_indices, 10, replace=False)
        
        for i in range(10):
            image = images[:, selected_indices[i]].reshape(28, 28)
            axes[digit, i].imshow(image, cmap='gray')
            axes[digit, i].axis('off')
    
    plt.suptitle('Sample Images from Each Class')
    plt.tight_layout()
    plt.show()
