import matplotlib.pyplot as plt
import numpy as np

# Configure dark theme for matplotlib
plt.style.use('dark_background')

# Try to import seaborn, but provide fallbacks if not available
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    print("Warning: seaborn not found. Some visualizations will use matplotlib instead.")

# Try to import confusion_matrix from sklearn, with fallback
try:
    from sklearn.metrics import confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not found. Confusion matrix will be computed manually.")
    
    # Manual implementation of confusion matrix
    def confusion_matrix(y_true, y_pred, labels=None):
        """Simple implementation of confusion matrix calculation."""
        if labels is None:
            labels = sorted(set(y_true) | set(y_pred))
        n_labels = len(labels)
        cm = np.zeros((n_labels, n_labels), dtype=int)
        for i, true_label in enumerate(labels):
            for j, pred_label in enumerate(labels):
                cm[i, j] = np.sum((y_true == true_label) & (y_pred == pred_label))
        return cm

def show_weights(weights, fig=None, ax=None):
    """Display weight matrices as grayscale images."""
    nfields = weights.shape[0]
    nsqrt = int(np.sqrt(nfields))
    nrows = nsqrt
    ncols = int(np.ceil(nfields / nrows))
    
    if fig is None and ax is None:
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 12), facecolor='#2d2d2d')
    else:
        axes = ax
        
    if nrows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(nfields):
        row = i // ncols
        col = i % ncols
        
        if row < nrows and col < ncols:
            weight_image = weights[i].reshape(28, 28)
            axes[row, col].imshow(weight_image, cmap='viridis')
            axes[row, col].axis('off')
    
    # Hide empty subplots
    for i in range(nfields, nrows * ncols):
        row = i // ncols
        col = i % ncols
        if row < nrows and col < ncols:
            axes[row, col].axis('off')
    
    plt.suptitle('Receptive Fields (Weights)', color='white')
    plt.tight_layout()
    
    if fig is None and ax is None:
        plt.show()
    return fig

def show_sample_images(images, labels, fig=None, ax=None):
    """Display sample images from each class."""
    if fig is None and ax is None:
        fig, axes = plt.subplots(10, 10, figsize=(10, 10), facecolor='#2d2d2d')
    else:
        axes = ax
    
    # Get 10 samples from each class
    for digit in range(10):
        digit_indices = np.where(np.argmax(labels, axis=0) == digit)[0]
        if len(digit_indices) >= 10:
            selected_indices = np.random.choice(digit_indices, 10, replace=False)
        else:
            selected_indices = digit_indices
            
        for i in range(min(10, len(selected_indices))):
            image = images[:, selected_indices[i]].reshape(28, 28)
            axes[digit, i].imshow(image, cmap='viridis')
            axes[digit, i].axis('off')
    
    plt.suptitle('Sample Images from Each Class', color='white')
    plt.tight_layout()
    
    if fig is None and ax is None:
        plt.show()
    return fig

def plot_training_curves(train_losses, test_errors, fig=None, ax=None):
    """Plot training loss and test error curves."""
    if fig is None and ax is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), facecolor='#2d2d2d')
    else:
        ax1, ax2 = ax
    
    epochs = range(1, len(train_losses) + 1)
    
    ax1.plot(epochs, train_losses, color='#00aaff', linewidth=2)
    ax1.set_title('Training Loss', color='white')
    ax1.set_xlabel('Epoch', color='white')
    ax1.set_ylabel('Loss', color='white')
    ax1.tick_params(colors='white')
    ax1.grid(True, color='#555555')
    
    ax2.plot(epochs, [e*100 for e in test_errors], color='#ff5500', linewidth=2)
    ax2.set_title('Test Error', color='white')
    ax2.set_xlabel('Epoch', color='white')
    ax2.set_ylabel('Error (%)', color='white')
    ax2.tick_params(colors='white')
    ax2.grid(True, color='#555555')
    
    plt.tight_layout()
    
    if fig is None and ax is None:
        plt.show()
    return fig

def plot_confusion_matrix(predictions, labels, fig=None, ax=None):
    """Plot confusion matrix for model predictions."""
    true_labels = np.argmax(labels, axis=0)
    
    if SKLEARN_AVAILABLE:
        cm = confusion_matrix(true_labels, predictions)
    else:
        # Manual calculation of confusion matrix
        cm = np.zeros((10, 10), dtype=int)
        for i in range(len(true_labels)):
            cm[true_labels[i], predictions[i]] += 1
    
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=(8, 8), facecolor='#2d2d2d')
    
    # Use seaborn's heatmap if available, otherwise use matplotlib
    if SEABORN_AVAILABLE:
        sns.heatmap(cm, annot=True, fmt="d", cmap="viridis", ax=ax, cbar=True)
    else:
        # Fallback to matplotlib
        cax = ax.matshow(cm, cmap='viridis')
        fig.colorbar(cax)
        
        # Add text annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='white')
    
    ax.set_title('Confusion Matrix', color='white')
    ax.set_xlabel('Predicted Labels', color='white')
    ax.set_ylabel('True Labels', color='white')
    ax.tick_params(colors='white')
    
    # Set ticks for both axes if using matplotlib fallback
    if not SEABORN_AVAILABLE:
        ax.set_xticks(np.arange(10))
        ax.set_yticks(np.arange(10))
        ax.set_xticklabels(np.arange(10))
        ax.set_yticklabels(np.arange(10))
    
    if fig is None and ax is None:
        plt.show()
    return fig

def plot_per_class_accuracy(predictions, labels, fig=None, ax=None):
    """Plot accuracy per class."""
    true_labels = np.argmax(labels, axis=0)
    
    # Calculate per-class accuracy
    class_accuracy = []
    for c in range(10):
        class_indices = np.where(true_labels == c)[0]
        correct = np.sum(predictions[class_indices] == c)
        accuracy = correct / len(class_indices) if len(class_indices) > 0 else 0
        class_accuracy.append(accuracy * 100)
    
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=(10, 5), facecolor='#2d2d2d')
    
    ax.bar(range(10), class_accuracy, color='#00aaff')
    ax.set_title('Accuracy per Class', color='white')
    ax.set_xlabel('Class', color='white')
    ax.set_ylabel('Accuracy (%)', color='white')
    ax.set_xticks(range(10))
    ax.tick_params(colors='white')
    ax.grid(True, axis='y', color='#555555')
    
    for i, acc in enumerate(class_accuracy):
        ax.text(i, acc + 1, f"{acc:.1f}%", ha='center', color='white')
    
    if fig is None and ax is None:
        plt.show()
    return fig

def plot_activation_distribution(activations, fig=None, ax=None):
    """Plot distribution of neuron activations."""
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=(10, 5), facecolor='#2d2d2d')
    
    ax.hist(activations.flatten(), bins=50, color='#00aaff', alpha=0.7)
    ax.set_title('Hidden Layer Activation Distribution', color='white')
    ax.set_xlabel('Activation Value', color='white')
    ax.set_ylabel('Frequency', color='white')
    ax.tick_params(colors='white')
    ax.grid(True, color='#555555')
    
    if fig is None and ax is None:
        plt.show()
    return fig
