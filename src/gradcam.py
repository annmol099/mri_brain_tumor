"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for Brain Tumor MRI
Phase 3.2: Model Interpretability and Tumor Localization Visualization
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
import cv2


class GradCAM:
    """Implement Grad-CAM for model interpretability."""
    
    def __init__(self, model, layer_name=None):
        """
        Initialize Grad-CAM.
        
        Args:
            model: Keras model
            layer_name (str): Name of the convolutional layer to visualize
                            If None, uses the last convolutional layer
        """
        self.model = model
        
        # Find the last convolutional layer if not specified
        if layer_name is None:
            # Search for last Conv2D layer
            for layer in reversed(self.model.layers):
                if 'conv' in layer.name.lower():
                    layer_name = layer.name
                    break
        
        self.layer_name = layer_name
        print(f"Using layer for Grad-CAM: {layer_name}")
        
        # Create a model that maps the input image to the activations
        # of the target layer and the output predictions
        self.grad_model = keras.models.Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(layer_name).output, self.model.output]
        )
    
    def make_gradcam_heatmap(self, img_array, pred_index=None):
        """
        Generate Grad-CAM heatmap.
        
        Args:
            img_array: Input image array (preprocessed)
            pred_index (int): Index of the class to visualize
                            If None, uses the predicted class
            
        Returns:
            numpy.ndarray: Grad-CAM heatmap
        """
        # Compute gradient of the top predicted class
        with tf.GradientTape() as tape:
            # Get activations and predictions
            conv_outputs, predictions = self.grad_model(img_array)
            
            # If pred_index is not specified, use the top prediction
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            
            # Get the score for the predicted class
            class_channel = predictions[:, pred_index]
        
        # Compute gradients of the class score with respect to the feature map
        grads = tape.gradient(class_channel, conv_outputs)
        
        # Global average pooling of gradients (importance weights)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Get the activations
        conv_outputs = conv_outputs[0]
        
        # Weight the channels by importance
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize heatmap between 0 and 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        return heatmap.numpy()
    
    def apply_heatmap_to_image(self, img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
        """
        Superimpose heatmap on original image.
        
        Args:
            img: Original image (0-1 or 0-255 range)
            heatmap: Grad-CAM heatmap
            alpha (float): Transparency of heatmap overlay
            colormap: OpenCV colormap
            
        Returns:
            numpy.ndarray: Image with heatmap overlay
        """
        # Ensure image is in 0-255 range
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        
        # Resize heatmap to match image size
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        
        # Convert heatmap to 0-255 range
        heatmap = np.uint8(255 * heatmap)
        
        # Apply colormap to heatmap
        heatmap_colored = cv2.applyColorMap(heatmap, colormap)
        
        # Convert BGR to RGB
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Superimpose heatmap on image
        superimposed = cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)
        
        return superimposed
    
    def visualize_gradcam(self, img, class_names=None, pred_index=None, 
                         save_path=None, show_plot=False):
        """
        Create and visualize Grad-CAM for an image.
        
        Args:
            img: Input image (preprocessed)
            class_names (list): List of class names
            pred_index (int): Class index to visualize
            save_path (str): Path to save visualization
            show_plot (bool): Whether to display the plot
            
        Returns:
            tuple: (heatmap, superimposed_img, prediction)
        """
        # Add batch dimension if needed
        if len(img.shape) == 3:
            img_array = np.expand_dims(img, axis=0)
        else:
            img_array = img
        
        # Get model prediction
        predictions = self.model.predict(img_array, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        # Use predicted class if pred_index not specified
        if pred_index is None:
            pred_index = predicted_class
        
        # Generate heatmap
        heatmap = self.make_gradcam_heatmap(img_array, pred_index)
        
        # Get original image (remove batch dimension)
        original_img = img_array[0] if len(img_array.shape) == 4 else img
        
        # Apply heatmap to image
        superimposed = self.apply_heatmap_to_image(original_img, heatmap)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(original_img if original_img.max() <= 1.0 else original_img / 255)
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Heatmap
        axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # Superimposed
        axes[2].imshow(superimposed)
        title = 'Grad-CAM Overlay'
        if class_names:
            title += f'\nPredicted: {class_names[predicted_class]} ({confidence:.2%})'
        axes[2].set_title(title, fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Grad-CAM visualization saved: {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return heatmap, superimposed, predictions[0]
    
    def visualize_multiple_samples(self, images, class_names, true_labels=None,
                                   num_samples=8, output_dir='../results/gradcam',
                                   save_name='gradcam_samples.png'):
        """
        Create Grad-CAM visualizations for multiple samples.
        
        Args:
            images: Array of images
            class_names (list): List of class names
            true_labels: True class labels (optional)
            num_samples (int): Number of samples to visualize
            output_dir (str): Directory to save visualizations
            save_name (str): Filename for combined plot
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Select random samples
        indices = np.random.choice(len(images), size=min(num_samples, len(images)), 
                                  replace=False)
        
        # Create figure
        rows = num_samples
        cols = 3
        fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 3))
        
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        print(f"\nGenerating Grad-CAM for {num_samples} samples...")
        
        for idx, i in enumerate(indices):
            img = images[i]
            
            # Add batch dimension
            img_array = np.expand_dims(img, axis=0)
            
            # Get prediction
            predictions = self.model.predict(img_array, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class]
            
            # Generate heatmap
            heatmap = self.make_gradcam_heatmap(img_array, predicted_class)
            
            # Apply heatmap
            superimposed = self.apply_heatmap_to_image(img, heatmap)
            
            # Plot original
            axes[idx, 0].imshow(img if img.max() <= 1.0 else img / 255)
            if idx == 0:
                axes[idx, 0].set_title('Original', fontsize=10, fontweight='bold')
            axes[idx, 0].axis('off')
            
            # Plot heatmap
            axes[idx, 1].imshow(heatmap, cmap='jet')
            if idx == 0:
                axes[idx, 1].set_title('Grad-CAM', fontsize=10, fontweight='bold')
            axes[idx, 1].axis('off')
            
            # Plot overlay
            axes[idx, 2].imshow(superimposed)
            title = f'{class_names[predicted_class]} ({confidence:.1%})'
            if true_labels is not None:
                true_class = class_names[true_labels[i]]
                is_correct = true_labels[i] == predicted_class
                color = 'green' if is_correct else 'red'
                title = f'True: {true_class}\nPred: {title}'
                axes[idx, 2].set_title(title, fontsize=9, color=color)
            else:
                axes[idx, 2].set_title(title, fontsize=9)
            if idx == 0:
                axes[idx, 2].text(0.5, 1.15, 'Overlay', 
                                ha='center', transform=axes[idx, 2].transAxes,
                                fontsize=10, fontweight='bold')
            axes[idx, 2].axis('off')
        
        plt.suptitle('Grad-CAM Visualizations - Tumor Localization', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        save_path = output_path / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Grad-CAM samples saved: {save_path}")
        plt.close()


def create_gradcam_for_model(model_path, layer_name=None):
    """
    Convenience function to create GradCAM instance from saved model.
    
    Args:
        model_path (str): Path to saved model
        layer_name (str): Name of layer for Grad-CAM
        
    Returns:
        GradCAM: Initialized GradCAM instance
    """
    print(f"Loading model from: {model_path}")
    model = keras.models.load_model(model_path)
    return GradCAM(model, layer_name)


if __name__ == "__main__":
    print("Grad-CAM Visualization Module")
    print("="*70)
    print("\nThis module provides Grad-CAM (Gradient-weighted Class Activation Mapping)")
    print("for model interpretability and tumor localization.")
    print("\nFeatures:")
    print("  • Generate heatmaps showing which regions influenced the prediction")
    print("  • Visualize attention maps for tumor localization")
    print("  • Create overlay visualizations")
    print("  • Batch processing for multiple samples")
    print("\nUse GradCAM class or create_gradcam_for_model() function.")
