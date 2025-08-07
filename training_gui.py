import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import os
import pickle
from data_loader import load_mnist_data
from neural_network import NeuralNetwork
from visualization import show_weights, show_sample_images

class NeuralNetworkGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Neural Network Training Interface")
        self.root.geometry("1000x800")
        
        self.network = None
        self.train_images = None
        self.train_labels = None
        self.test_images = None
        self.test_labels = None
        self.training = False
        
        self.setup_gui()
        
    def setup_gui(self):
        # Create notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Data tab
        data_frame = ttk.Frame(notebook)
        notebook.add(data_frame, text="Data")
        self.setup_data_tab(data_frame)
        
        # Training tab
        train_frame = ttk.Frame(notebook)
        notebook.add(train_frame, text="Training")
        self.setup_training_tab(train_frame)
        
        # Visualization tab
        viz_frame = ttk.Frame(notebook)
        notebook.add(viz_frame, text="Visualization")
        self.setup_visualization_tab(viz_frame)
        
        # Model tab
        model_frame = ttk.Frame(notebook)
        notebook.add(model_frame, text="Model")
        self.setup_model_tab(model_frame)
    
    def setup_data_tab(self, parent):
        # Dataset loading section
        ttk.Label(parent, text="Dataset Loading", font=('Arial', 12, 'bold')).pack(pady=10)
        
        load_frame = ttk.Frame(parent)
        load_frame.pack(pady=10)
        
        ttk.Button(load_frame, text="Select MNIST Dataset Folder", 
                  command=self.load_dataset).pack(side=tk.LEFT, padx=5)
        
        self.data_status = ttk.Label(parent, text="No dataset loaded")
        self.data_status.pack(pady=5)
        
        # Dataset info
        self.data_info_frame = ttk.LabelFrame(parent, text="Dataset Information")
        self.data_info_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.data_info = tk.Text(self.data_info_frame, height=10, width=60)
        self.data_info.pack(padx=10, pady=10)
    
    def setup_training_tab(self, parent):
        # Hyperparameters
        hyper_frame = ttk.LabelFrame(parent, text="Hyperparameters")
        hyper_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Create hyperparameter controls
        params_frame = ttk.Frame(hyper_frame)
        params_frame.pack(padx=10, pady=10)
        
        # Learning rate
        ttk.Label(params_frame, text="Learning Rate:").grid(row=0, column=0, sticky='w', padx=5)
        self.lr_var = tk.StringVar(value="0.1")  # Increased learning rate
        ttk.Entry(params_frame, textvariable=self.lr_var, width=10).grid(row=0, column=1, padx=5)
        
        # Hidden units
        ttk.Label(params_frame, text="Hidden Units:").grid(row=0, column=2, sticky='w', padx=5)
        self.hidden_var = tk.StringVar(value="100")  # Smaller network to start
        ttk.Entry(params_frame, textvariable=self.hidden_var, width=10).grid(row=0, column=3, padx=5)
        
        # Momentum
        ttk.Label(params_frame, text="Momentum:").grid(row=1, column=0, sticky='w', padx=5)
        self.momentum_var = tk.StringVar(value="0.9")  # Increased momentum
        ttk.Entry(params_frame, textvariable=self.momentum_var, width=10).grid(row=1, column=1, padx=5)
        
        # Weight decay
        ttk.Label(params_frame, text="Weight Decay:").grid(row=1, column=2, sticky='w', padx=5)
        self.decay_var = tk.StringVar(value="0.0001")
        ttk.Entry(params_frame, textvariable=self.decay_var, width=10).grid(row=1, column=3, padx=5)
        
        # Epochs
        ttk.Label(params_frame, text="Max Epochs:").grid(row=2, column=0, sticky='w', padx=5)
        self.epochs_var = tk.StringVar(value="100")
        ttk.Entry(params_frame, textvariable=self.epochs_var, width=10).grid(row=2, column=1, padx=5)
        
        # Batch size
        ttk.Label(params_frame, text="Batch Size:").grid(row=2, column=2, sticky='w', padx=5)
        self.batch_size_var = tk.StringVar(value="100")  # Smaller batch size
        ttk.Entry(params_frame, textvariable=self.batch_size_var, width=10).grid(row=2, column=3, padx=5)
        
        # Training controls
        control_frame = ttk.Frame(parent)
        control_frame.pack(pady=10)
        
        self.start_btn = ttk.Button(control_frame, text="Start Training", 
                                   command=self.start_training)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(control_frame, text="Stop Training", 
                                  command=self.stop_training, state='disabled')
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        # Progress
        self.progress = ttk.Progressbar(parent, mode='indeterminate')
        self.progress.pack(fill=tk.X, padx=10, pady=5)
        
        # Training log
        log_frame = ttk.LabelFrame(parent, text="Training Log")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.log_text = tk.Text(log_frame, height=15)
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def setup_visualization_tab(self, parent):
        # Visualization controls
        viz_controls = ttk.Frame(parent)
        viz_controls.pack(pady=10)
        
        ttk.Button(viz_controls, text="Show Sample Images", 
                  command=self.show_sample_images).pack(side=tk.LEFT, padx=5)
        ttk.Button(viz_controls, text="Show Weight Visualization", 
                  command=self.show_weights).pack(side=tk.LEFT, padx=5)
        ttk.Button(viz_controls, text="Show Training Curves", 
                  command=self.show_training_curves).pack(side=tk.LEFT, padx=5)
        
        # Matplotlib canvas
        self.viz_frame = ttk.Frame(parent)
        self.viz_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def setup_model_tab(self, parent):
        # Model management
        model_frame = ttk.LabelFrame(parent, text="Model Management")
        model_frame.pack(fill=tk.X, padx=10, pady=10)
        
        model_controls = ttk.Frame(model_frame)
        model_controls.pack(pady=10)
        
        ttk.Button(model_controls, text="Save Model", 
                  command=self.save_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(model_controls, text="Load Model", 
                  command=self.load_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(model_controls, text="Export for Inference", 
                  command=self.export_model).pack(side=tk.LEFT, padx=5)
        
        # Model info
        self.model_info = tk.Text(model_frame, height=10)
        self.model_info.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def load_dataset(self):
        folder_path = filedialog.askdirectory(title="Select MNIST Dataset Folder")
        if folder_path:
            try:
                self.train_images, self.train_labels, self.test_images, self.test_labels = load_mnist_data(folder_path)
                
                # Verify data is loaded correctly
                if self.train_labels.shape[0] != 10 or self.test_labels.shape[0] != 10:
                    messagebox.showerror("Error", "Labels are not in one-hot format (10 rows expected)")
                    return
                    
                # Validate images have proper normalization
                if np.max(self.train_images) > 1.0 or np.min(self.train_images) < 0:
                    messagebox.showerror("Error", "Images not normalized between 0-1")
                    return
                
                self.data_status.config(text=f"Dataset loaded from: {folder_path}")
                
                # Update info with validation
                info = f"""Dataset Information:
Training Images: {self.train_images.shape[1]} samples, {self.train_images.shape[0]} features
Training Labels: {self.train_labels.shape}
Test Images: {self.test_images.shape[1]} samples, {self.test_images.shape[0]} features  
Test Labels: {self.test_labels.shape}

Image values: min={np.min(self.train_images):.3f}, max={np.max(self.train_images):.3f}
Label verification: {np.all(np.sum(self.train_labels, axis=0) == 1.0)} (should be True)

Dataset path: {folder_path}"""
                
                self.data_info.delete(1.0, tk.END)
                self.data_info.insert(tk.END, info)
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load dataset: {str(e)}")
    
    def start_training(self):
        if self.train_images is None:
            messagebox.showerror("Error", "Please load dataset first")
            return
            
        if self.training:
            return
            
        self.training = True
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.progress.start()
        
        # Start training in separate thread
        threading.Thread(target=self.train_network, daemon=True).start()
    
    def stop_training(self):
        self.training = False
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.progress.stop()
    
    def train_network(self):
        try:
            # Get hyperparameters
            lr = float(self.lr_var.get())
            nhid = int(self.hidden_var.get())
            momentum = float(self.momentum_var.get())
            weight_decay = float(self.decay_var.get())
            max_epochs = int(self.epochs_var.get())
            batch_size = int(self.batch_size_var.get())
            
            # Create network
            self.network = NeuralNetwork(nhid, lr, momentum, weight_decay)
            
            # Training loop
            num_batches = self.train_images.shape[1] // batch_size
            train_losses = []
            test_errors = []
            
            self.log_message("Starting training...")
            self.log_message(f"Network: {nhid} hidden units, {lr} learning rate")
            
            for epoch in range(max_epochs):
                if not self.training:
                    break
                
                # Shuffle training data for each epoch
                indices = np.random.permutation(self.train_images.shape[1])
                shuffled_images = self.train_images[:, indices]
                shuffled_labels = self.train_labels[:, indices]
                
                batch_losses = []
                
                # Mini-batch training - matching MATLAB implementation
                for batch in range(num_batches):
                    start_idx = batch * batch_size
                    end_idx = start_idx + batch_size
                    
                    batch_images = shuffled_images[:, start_idx:end_idx]
                    batch_labels = shuffled_labels[:, start_idx:end_idx]
                    
                    # Forward pass
                    r0, r1 = self.network.forward_pass(batch_images)
                    
                    # Calculate loss for this batch
                    batch_loss = self.network.calculate_loss(r1, batch_labels)
                    batch_losses.append(batch_loss)
                    
                    # Backward pass
                    grads = self.network.backward_pass(batch_images, batch_labels, r0, r1)
                    
                    # Update weights
                    self.network.update_weights(*grads)
                
                # Calculate epoch metrics
                avg_loss = np.mean(batch_losses)
                train_losses.append(avg_loss)
                
                # Test error
                test_r0, test_r1 = self.network.forward_pass(self.test_images)
                test_error = self.network.calculate_error(test_r1, self.test_labels)
                test_errors.append(test_error)
                
                # Log progress
                self.log_message(f"Epoch {epoch+1}, Loss: {avg_loss:.3f}, Test Error: {test_error*100:.1f}%")
            
            self.training_losses = train_losses
            self.test_errors = test_errors
            self.log_message("Training completed!")
            
        except Exception as e:
            import traceback
            self.log_message(f"Training error: {str(e)}")
            self.log_message(traceback.format_exc())
        finally:
            self.stop_training()
    
    def log_message(self, message):
        self.root.after(0, lambda: self._update_log(message))
    
    def _update_log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
    
    def show_sample_images(self):
        if self.train_images is None:
            messagebox.showerror("Error", "No dataset loaded")
            return
        show_sample_images(self.train_images, self.train_labels)
    
    def show_weights(self):
        if self.network is None:
            messagebox.showerror("Error", "No trained network available")
            return
        show_weights(self.network.W0)
    
    def show_training_curves(self):
        if not hasattr(self, 'training_losses'):
            messagebox.showerror("Error", "No training data available")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(self.training_losses)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        
        ax2.plot([e*100 for e in self.test_errors])
        ax2.set_title('Test Error')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Error (%)')
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self):
        if self.network is None:
            messagebox.showerror("Error", "No trained network to save")
            return
            
        filepath = filedialog.asksaveasfilename(
            defaultextension=".npz",
            filetypes=[("NumPy files", "*.npz"), ("All files", "*.*")]
        )
        if filepath:
            self.network.save_model(filepath)
            messagebox.showinfo("Success", "Model saved successfully")
    
    def load_model(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("NumPy files", "*.npz"), ("All files", "*.*")]
        )
        if filepath:
            try:
                self.network = NeuralNetwork()
                self.network.load_model(filepath)
                messagebox.showinfo("Success", "Model loaded successfully")
                
                # Update model info
                info = f"""Loaded Model Information:
Hidden units: {self.network.W0.shape[0]}
Input features: {self.network.W0.shape[1]}
Output classes: {self.network.W1.shape[0]}
Learning rate: {self.network.learning_rate}
Momentum: {self.network.momentum}
Weight decay: {self.network.weight_decay}

Model file: {filepath}"""
                
                self.model_info.delete(1.0, tk.END)
                self.model_info.insert(tk.END, info)
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {str(e)}")
    
    def export_model(self):
        if self.network is None:
            messagebox.showerror("Error", "No trained network to export")
            return
            
        filepath = filedialog.asksaveasfilename(
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        if filepath:
            try:
                export_data = {
                    'W0': self.network.W0,
                    'W1': self.network.W1,
                    'b0': self.network.b0,
                    'b1': self.network.b1,
                    'architecture': {
                        'input_size': 784,
                        'hidden_size': self.network.nhid,
                        'output_size': 10
                    }
                }
                
                with open(filepath, 'wb') as f:
                    pickle.dump(export_data, f)
                
                messagebox.showinfo("Success", "Model exported for inference")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export model: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = NeuralNetworkGUI(root)
    root.mainloop()
