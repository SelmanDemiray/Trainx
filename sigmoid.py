import numpy as np

def sigmoid(x):
    """
    Calculates the sigmoid function on each element of a matrix.
    
    Args:
        x: Input matrix or array
        
    Returns:
        phi: Sigmoid activation values
    """
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow
