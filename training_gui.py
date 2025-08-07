import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import threading
import os
import pickle
from data_loader import load_mnist_data
from neural_network import NeuralNetwork
from visualization import (
    show_weights, show_sample_images, plot_training_curves,
    plot_confusion_matrix, plot_per_class_accuracy, plot_activation_distribution
)

class DarkTheme:
    """Dark theme color scheme for the application."""
    BG_COLOR = "#1e1e1e"
    TEXT_COLOR = "#ffffff"
    ACCENT_COLOR = "#3a3a3a"
    HIGHLIGHT_COLOR = "#0078d7"
    BUTTON_BG = "#2d2d2d"
    BUTTON_FG = "#ffffff"
    ENTRY_BG = "#2d2d2d"
    ENTRY_FG = "#ffffff"
    TAB_BG = "#1e1e1e"
    TAB_FG = "#ffffff"
    BORDER_COLOR = "#3a3a3a"

class NeuralNetworkGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Neural Network Training Interface")
        self.root.geometry("1200x800")
        
        self.network = None
        self.train_images = None
        self.train_labels = None
        self.test_images = None
        self.test_labels = None
        self.training = False
        
        self.apply_dark_theme()
        self.setup_gui()
        
    def apply_dark_theme(self):
        """Apply dark theme to the application."""
        self.root.configure(background=DarkTheme.BG_COLOR)
        
        # Configure styles for ttk widgets
        style = ttk.Style()
        style.theme_use('clam')  # Use the 'clam' theme as a base
        
        # Configure TFrame
        style.configure("TFrame", background=DarkTheme.BG_COLOR)
        
        # Configure TLabel
        style.configure("TLabel", background=DarkTheme.BG_COLOR, foreground=DarkTheme.TEXT_COLOR)
        
        # Configure TButton
        style.configure("TButton", 
                       background=DarkTheme.BUTTON_BG, 
                       foreground=DarkTheme.BUTTON_FG,
                       borderwidth=1)
        style.map("TButton",
                 background=[('active', DarkTheme.HIGHLIGHT_COLOR)],
                 foreground=[('active', DarkTheme.TEXT_COLOR)])
        
        # Configure TEntry
        style.configure("TEntry", 
                       fieldbackground=DarkTheme.ENTRY_BG, 
                       foreground=DarkTheme.ENTRY_FG)
        
        # Configure TNotebook
        style.configure("TNotebook", background=DarkTheme.BG_COLOR, bordercolor=DarkTheme.BORDER_COLOR)
        style.configure("TNotebook.Tab", background=DarkTheme.TAB_BG, foreground=DarkTheme.TAB_FG,
                       padding=[10, 5])
        style.map("TNotebook.Tab",
                 background=[('selected', DarkTheme.HIGHLIGHT_COLOR)],
                 foreground=[('selected', DarkTheme.TEXT_COLOR)])
        
        # Configure TProgressbar
        style.configure("TProgressbar", 
                       background=DarkTheme.HIGHLIGHT_COLOR, 
                       troughcolor=DarkTheme.ACCENT_COLOR)
        
        # Configure TLabelframe
        style.configure("TLabelframe", 
                       background=DarkTheme.BG_COLOR, 
                       foreground=DarkTheme.TEXT_COLOR,
                       bordercolor=DarkTheme.BORDER_COLOR)
        style.configure("TLabelframe.Label", 
                       background=DarkTheme.BG_COLOR, 
                       foreground=DarkTheme.TEXT_COLOR)
        
        # Configure text widgets
        self.root.option_add("*Text.background", DarkTheme.ENTRY_BG)
        self.root.option_add("*Text.foreground", DarkTheme.ENTRY_FG)
        self.root.option_add("*Text.borderwidth", 1)
        self.root.option_add("*Text.relief", "solid")
        
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
        # Visualization type selection
        viz_type_frame = ttk.LabelFrame(parent, text="Visualization Type")
        viz_type_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Create a frame for the buttons
        button_frame = ttk.Frame(viz_type_frame)
        button_frame.pack(padx=10, pady=10)
        
        # Create visualization buttons with more options
        viz_buttons = [
            ("Sample Images", self.show_embedded_samples),
            ("Weight Visualization", self.show_embedded_weights),
            ("Training Curves", self.show_embedded_training_curves),
            ("Confusion Matrix", self.show_embedded_confusion_matrix),
            ("Per-Class Accuracy", self.show_embedded_class_accuracy),
            ("Activation Distribution", self.show_embedded_activations)
        ]
        
        # Arrange buttons in a grid (2 rows, 3 columns)
        for i, (text, command) in enumerate(viz_buttons):
            row = i // 3
            col = i % 3
            ttk.Button(button_frame, text=text, command=command).grid(row=row, column=col, padx=5, pady=5, sticky="we")
        
        # Configure column weights for button_frame
        for i in range(3):
            button_frame.columnconfigure(i, weight=1)
        
        # Add a save figure button
        save_frame = ttk.Frame(viz_type_frame)
        save_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(save_frame, text="Save Current Figure", command=self.save_current_figure).pack(side=tk.RIGHT, padx=5)
        
        # Canvas for displaying visualizations
        self.viz_canvas_frame = ttk.Frame(parent)
        self.viz_canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Instructions label
        instructions = ttk.Label(self.viz_canvas_frame, 
                                text="Select a visualization type above to display it here.",
                                font=('Arial', 10))
        instructions.pack(expand=True)
        
        # This will hold the current matplotlib figure
        self.current_figure = None
        self.current_canvas = None
    
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
        plot_training_curves(self.training_losses, self.test_errors)
    
    def _clear_viz_frame(self):
        """Clear the visualization frame before adding a new plot."""
        for widget in self.viz_canvas_frame.winfo_children():
            widget.destroy()
    
    def _setup_canvas(self, figure):
        """Set up a matplotlib canvas in the visualization frame."""
        self._clear_viz_frame()
        
        # Create canvas and toolbar
        canvas = FigureCanvasTkAgg(figure, self.viz_canvas_frame)
        canvas.draw()
        
        toolbar = NavigationToolbar2Tk(canvas, self.viz_canvas_frame, pack_toolbar=False)
        toolbar.config(background=DarkTheme.BG_COLOR)
        for button in toolbar.winfo_children():
            if isinstance(button, tk.Button):
                button.config(background=DarkTheme.BUTTON_BG, foreground=DarkTheme.BUTTON_FG)
        toolbar.update()
        
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.current_figure = figure
        self.current_canvas = canvas
        
        return canvas
    
    def show_embedded_samples(self):
        if self.train_images is None:
            messagebox.showerror("Error", "No dataset loaded")
            return
        
        fig = plt.figure(figsize=(10, 10), facecolor=DarkTheme.BG_COLOR)
        fig.suptitle("Sample Images from Each Class", color="white", fontsize=16)
        
        # Create a grid of subplots
        axes = []
        for i in range(10):
            for j in range(10):
                ax = fig.add_subplot(10, 10, i*10 + j + 1)
                ax.axis('off')
                axes.append(ax)
        
        fig = show_sample_images(self.train_images, self.train_labels, fig=fig, ax=np.array(axes).reshape(10, 10))
        self._setup_canvas(fig)
    
    def show_embedded_weights(self):
        if self.network is None:
            messagebox.showerror("Error", "No trained network available")
            return
        
        # Calculate grid dimensions
        n_weights = min(100, self.network.W0.shape[0])  # Show up to 100 weights
        n_rows = int(np.sqrt(n_weights))
        n_cols = int(np.ceil(n_weights / n_rows))
        
        fig = plt.figure(figsize=(12, 12), facecolor=DarkTheme.BG_COLOR)
        fig.suptitle("Hidden Layer Weights", color="white", fontsize=16)
        
        # Create a grid of subplots
        axes = []
        for i in range(n_rows * n_cols):
            ax = fig.add_subplot(n_rows, n_cols, i + 1)
            ax.axis('off')
            axes.append(ax)
        
        # Show weights
        fig = show_weights(self.network.W0[:n_weights], fig=fig, ax=np.array(axes).reshape(n_rows, n_cols))
        self._setup_canvas(fig)
    
    def show_embedded_training_curves(self):
        if not hasattr(self, 'training_losses'):
            messagebox.showerror("Error", "No training data available")
            return
        
        fig = plt.figure(figsize=(12, 5), facecolor=DarkTheme.BG_COLOR)
        
        # Create two subplots
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        
        fig = plot_training_curves(self.training_losses, self.test_errors, fig=fig, ax=[ax1, ax2])
        self._setup_canvas(fig)
    
    def show_embedded_confusion_matrix(self):
        if self.network is None or self.test_images is None:
            messagebox.showerror("Error", "Network or test data not available")
            return
        
        # Get predictions
        _, test_r1 = self.network.forward_pass(self.test_images)
        predictions = np.argmax(test_r1, axis=0)
        
        fig = plt.figure(figsize=(10, 8), facecolor=DarkTheme.BG_COLOR)
        ax = fig.add_subplot(1, 1, 1)
        
        fig = plot_confusion_matrix(predictions, self.test_labels, fig=fig, ax=ax)
        self._setup_canvas(fig)
    
    def show_embedded_class_accuracy(self):
        if self.network is None or self.test_images is None:
            messagebox.showerror("Error", "Network or test data not available")
            return
        
        # Get predictions
        _, test_r1 = self.network.forward_pass(self.test_images)
        predictions = np.argmax(test_r1, axis=0)
        
        fig = plt.figure(figsize=(10, 5), facecolor=DarkTheme.BG_COLOR)
        ax = fig.add_subplot(1, 1, 1)
        
        fig = plot_per_class_accuracy(predictions, self.test_labels, fig=fig, ax=ax)
        self._setup_canvas(fig)
    
    def show_embedded_activations(self):
        if self.network is None or self.test_images is None:
            messagebox.showerror("Error", "Network or test data not available")
            return
        
        # Get hidden layer activations
        hidden_activations, _ = self.network.forward_pass(self.test_images)
        
        fig = plt.figure(figsize=(10, 5), facecolor=DarkTheme.BG_COLOR)
        ax = fig.add_subplot(1, 1, 1)
        
        fig = plot_activation_distribution(hidden_activations, fig=fig, ax=ax)
        self._setup_canvas(fig)
    
    def save_current_figure(self):
        if self.current_figure is None:
            messagebox.showerror("Error", "No figure to save")
            return
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("PDF files", "*.pdf"),
                ("SVG files", "*.svg"),
                ("All files", "*.*")
            ]
        )
        
        if filepath:
            try:
                self.current_figure.savefig(filepath, facecolor=self.current_figure.get_facecolor())
                messagebox.showinfo("Success", f"Figure saved to {filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save figure: {str(e)}")
    
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
