import numpy as np
import pickle
from sigmoid import sigmoid

class InferenceModel:
    def __init__(self, model_path):
        """Load exported model for inference."""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.W0 = model_data['W0']
        self.W1 = model_data['W1']
        self.b0 = model_data['b0']
        self.b1 = model_data['b1']
        self.architecture = model_data['architecture']
    
    def predict(self, images):
        """
        Predict classes for input images.
        
        Args:
            images: Input images as numpy array (784 x n_samples)
            
        Returns:
            predictions: Predicted class labels
            probabilities: Output probabilities for each class
        """
        # Ensure correct input shape
        if images.ndim == 1:
            images = images.reshape(-1, 1)
        
        # Forward pass
        r0 = sigmoid(np.dot(self.W0, images) + self.b0)
        r1 = sigmoid(np.dot(self.W1, r0) + self.b1)
        
        # Get predictions
        predictions = np.argmax(r1, axis=0)
        
        return predictions, r1
    
    def predict_single(self, image):
        """Predict class for a single image."""
        if image.shape != (784,):
            raise ValueError(f"Image must be flattened to 784 features, got {image.shape}")
        
        predictions, probabilities = self.predict(image.reshape(-1, 1))
        return predictions[0], probabilities[:, 0]

    def preprocess_image(self, image_path):
        """
        Preprocess an image file for inference.
        This method is a convenience wrapper for external image preprocessing.
        
        Args:
            image_path: Path to image file
            
        Returns:
            processed_image: Numpy array ready for inference
        """
        from image_processor import process_image_cv2
        return process_image_cv2(image_path)

# Example usage
if __name__ == "__main__":
    # Load model
    model = InferenceModel("exported_model.pkl")
    
    # Example prediction (replace with actual image data)
    test_image = np.random.rand(784)  # Replace with real image
    prediction, probabilities = model.predict_single(test_image)
    
    print(f"Predicted digit: {prediction}")
    print(f"Confidence: {probabilities[prediction]:.3f}")
