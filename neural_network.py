import numpy as np
from sigmoid import sigmoid

class NeuralNetwork:
    def __init__(self, nhid=10000, num_classes=10, input_size=784, learning_rate=0.01, momentum=0.001, weight_decay=0.0001):
        self.nhid = nhid
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.input_size = input_size
        
        # Initialize weights and biases with Xavier/Glorot initialization
        sigma = np.sqrt(1.0 / input_size)
        self.W0 = np.random.randn(nhid, input_size) * sigma
        self.W1 = np.random.randn(num_classes, nhid) * sigma
        self.b0 = np.zeros((nhid, 1))
        self.b1 = np.zeros((num_classes, 1))
        
        # Initialize momentum terms
        self.delta_W0 = np.zeros_like(self.W0)
        self.delta_W1 = np.zeros_like(self.W1)
        self.delta_b0 = np.zeros_like(self.b0)
        self.delta_b1 = np.zeros_like(self.b1)
    
    def forward_pass(self, images):
        """Run forward pass through the network."""
        # Hidden layer - ensure broadcasting works correctly with bias
        x0 = np.dot(self.W0, images) + self.b0  # nhid × batch_size
        r0 = sigmoid(x0)
        
        # Output layer
        x1 = np.dot(self.W1, r0) + self.b1  # 10 × batch_size
        r1 = sigmoid(x1)
        
        return r0, r1
    
    def backward_pass(self, images, labels, r0, r1):
        """Run backward pass to calculate gradients, matching MATLAB implementation."""
        batch_size = images.shape[1]
        
        # Gradient of loss w.r.t. output layer activity
        dL_by_dr1 = r1 - labels  # 10 × batch_size
        
        # Gradient of output layer activity w.r.t. input
        dr1_by_dx1 = r1 * (1 - r1)  # 10 × batch_size
        
        # Gradient of loss w.r.t. output layer input
        dL_by_dx1 = dL_by_dr1 * dr1_by_dx1  # 10 × batch_size
        
        # Gradients w.r.t. output weights
        dL_by_dW1 = np.dot(dL_by_dx1, r0.T) / batch_size  # Normalize by batch size
        
        # Gradients w.r.t. output biases
        dL_by_db1 = np.sum(dL_by_dx1, axis=1, keepdims=True) / batch_size  # Normalize by batch size
        
        # Calculate gradient of loss w.r.t. hidden layer activity
        # Following MATLAB: dL_by_dr0 = (dL_by_dx1.' * W1).'
        dL_by_dr0 = np.dot(self.W1.T, dL_by_dx1)  # This matches MATLAB's equation better
        
        # Gradient of hidden layer activity w.r.t. input
        dr0_by_dx0 = r0 * (1 - r0)  # nhid × batch_size
        
        # Gradient of loss w.r.t. hidden layer input
        dL_by_dx0 = dL_by_dr0 * dr0_by_dx0  # nhid × batch_size
        
        # Gradients w.r.t. hidden weights
        dL_by_dW0 = np.dot(dL_by_dx0, images.T) / batch_size  # Normalize by batch size
        
        # Gradients w.r.t. hidden biases
        dL_by_db0 = np.sum(dL_by_dx0, axis=1, keepdims=True) / batch_size  # Normalize by batch size
        
        return dL_by_dW0, dL_by_dW1, dL_by_db0, dL_by_db1
    
    def update_weights(self, dL_by_dW0, dL_by_dW1, dL_by_db0, dL_by_db1):
        """Update weights using gradients with momentum and weight decay."""
        # Calculate parameter updates exactly as in MATLAB
        self.delta_W0 = (-self.learning_rate * dL_by_dW0 + 
                        self.momentum * self.delta_W0 - 
                        self.weight_decay * self.W0)
        self.delta_W1 = (-self.learning_rate * dL_by_dW1 + 
                        self.momentum * self.delta_W1 - 
                        self.weight_decay * self.W1)
        self.delta_b0 = (-self.learning_rate * dL_by_db0 + 
                        self.momentum * self.delta_b0 - 
                        self.weight_decay * self.b0)
        self.delta_b1 = (-self.learning_rate * dL_by_db1 + 
                        self.momentum * self.delta_b1 - 
                        self.weight_decay * self.b1)
        
        # Update parameters
        self.W0 += self.delta_W0
        self.W1 += self.delta_W1
        self.b0 += self.delta_b0
        self.b1 += self.delta_b1
    
    def calculate_loss(self, r1, labels):
        """Calculate mean squared error loss, matching MATLAB exactly."""
        # Calculate per-batch loss to match MATLAB implementation
        batch_size = r1.shape[1]
        return 0.5 * np.sum((r1 - labels) ** 2) / batch_size
        
    def calculate_error(self, r1, labels):
        """Calculate classification error rate using max activation approach."""
        predictions = np.argmax(r1, axis=0)
        true_labels = np.argmax(labels, axis=0)
        return 1.0 - np.mean(predictions == true_labels)
    
    def save_model(self, filepath, dataset_name="unknown"):
        """Save model parameters."""
        np.savez(filepath, W0=self.W0, W1=self.W1, b0=self.b0, b1=self.b1, 
                num_classes=self.num_classes, input_size=self.input_size,
                dataset_name=dataset_name)
    
    def load_model(self, filepath):
        """Load model parameters."""
        data = np.load(filepath, allow_pickle=True)
        self.W0 = data['W0']
        self.W1 = data['W1']
        self.b0 = data['b0']
        self.b1 = data['b1']
        
        # Handle models saved with or without dimension information
        if 'num_classes' in data:
            self.num_classes = int(data['num_classes'])
        else:
            self.num_classes = self.W1.shape[0]
            
        if 'input_size' in data:
            self.input_size = int(data['input_size'])
        else:
            self.input_size = self.W0.shape[1]
            
        # Load dataset name if available
        self.dataset_name = str(data.get('dataset_name', "unknown"))
    
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
        r0, r1 = self.forward_pass(images)
        
        # Get predictions
        predictions = np.argmax(r1, axis=0)
        
        return predictions, r1
    
    def predict_single(self, image):
        """
        Predict class for a single image.
        
        Args:
            image: Single image as a 784-element numpy array
            
        Returns:
            prediction: Predicted class label
            probabilities: Output probabilities for each class
        """
        if image.shape != (self.input_size,):
            raise ValueError(f"Image must be flattened to {self.input_size} features, got {image.shape}")
        
        predictions, probabilities = self.predict(image.reshape(-1, 1))
        return predictions[0], probabilities[:, 0]
