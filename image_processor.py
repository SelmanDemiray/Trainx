import numpy as np
from PIL import Image, ImageOps
import cv2

def process_image_for_inference(image):
    """
    Process an uploaded image for MNIST inference:
    1. Convert to grayscale
    2. Resize to 28x28
    3. Normalize pixel values
    4. Invert if needed (MNIST has white digits on black background)
    5. Return flattened 784-element array
    
    Args:
        image: PIL Image object
        
    Returns:
        processed_image: 784-element numpy array ready for inference
    """
    try:
        # Convert to grayscale
        image = image.convert('L')
        
        # Resize to 28x28 with antialiasing
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Check if we need to invert the image (MNIST has white digits on black background)
        # If mean pixel value is high (>128), it means the background is white and digit is black
        if np.mean(img_array) > 128:
            img_array = 255 - img_array  # Invert colors
        
        # Normalize to 0-1
        img_array = img_array.astype(np.float32) / 255.0
        
        # Apply thresholding to make digits more clear
        threshold = 0.3
        img_array = np.where(img_array > threshold, img_array, 0.0)
        
        # Center the digit in the image (like MNIST)
        img_array = center_digit(img_array)
        
        # Flatten to 784 elements
        return img_array.reshape(784)
        
    except Exception as e:
        raise Exception(f"Image processing failed: {str(e)}")

def process_image_cv2(image_path):
    """Alternative processing using OpenCV for more advanced preprocessing."""
    try:
        # Read image with OpenCV
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Resize to 28x28
        img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Apply adaptive thresholding
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
        
        # Normalize to 0-1
        img = img.astype(np.float32) / 255.0
        
        # Center the digit
        img = center_digit(img)
        
        return img.reshape(784)
        
    except Exception as e:
        raise Exception(f"OpenCV processing failed: {str(e)}")

def center_digit(img_array):
    """Center the digit in the image using center of mass."""
    # Find center of mass
    cy, cx = np.where(img_array > 0.1)
    if len(cy) == 0 or len(cx) == 0:  # Empty image
        return img_array
        
    # Calculate moments
    cy_mean = np.mean(cy)
    cx_mean = np.mean(cx)
    
    # Calculate shifts
    height, width = img_array.shape
    shift_y = int(height/2 - cy_mean)
    shift_x = int(width/2 - cx_mean)
    
    # Create transformation matrix
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    
    # Apply affine transformation
    centered = cv2.warpAffine(img_array, M, (width, height))
    
    return centered
