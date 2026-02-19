"""
Data Augmentation Module for Brain Tumor MRI Classification
Phase 1.4: Data Augmentation
"""

import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


class DataAugmentor:
    """Handles data augmentation for MRI images."""
    
    def __init__(self, image_size=224):
        """
        Initialize data augmentor with augmentation strategies.
        
        Args:
            image_size (int): Size of the input images
        """
        self.image_size = image_size
        
        # Training augmentation pipeline
        self.train_transform = A.Compose([
            # Geometric transformations
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0),
            
            # Zoom (Scale)
            A.RandomScale(scale_limit=0.2, p=0.5),
            A.PadIfNeeded(min_height=image_size, min_width=image_size, 
                         border_mode=cv2.BORDER_CONSTANT, value=0),
            A.RandomCrop(height=image_size, width=image_size, p=1.0),
            
            # Brightness and Contrast
            A.RandomBrightnessContrast(brightness_limit=0.2, 
                                      contrast_limit=0.2, p=0.5),
            
            # Gaussian noise
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            
            # Additional helpful augmentations for medical images
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
            A.GridDistortion(p=0.3),
            A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.3),
        ])
        
        # Validation/Test transformation (no augmentation, just normalization)
        self.val_transform = A.Compose([
            A.Resize(height=image_size, width=image_size),
        ])
        
        # Lighter augmentation for fine-tuning stage
        self.light_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.3, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.RandomBrightnessContrast(brightness_limit=0.1, 
                                      contrast_limit=0.1, p=0.3),
        ])
    
    def augment_image(self, image, mode='train'):
        """
        Apply augmentation to a single image.
        
        Args:
            image (numpy.ndarray): Input image (H, W, C)
            mode (str): 'train', 'val', or 'light'
            
        Returns:
            numpy.ndarray: Augmented image
        """
        if mode == 'train':
            augmented = self.train_transform(image=image)
        elif mode == 'light':
            augmented = self.light_transform(image=image)
        else:  # val or test
            augmented = self.val_transform(image=image)
        
        return augmented['image']
    
    def get_tensorflow_augmentation(self):
        """
        Get TensorFlow/Keras augmentation layers for training.
        This can be used directly in the model pipeline.
        
        Returns:
            tf.keras.Sequential: Sequential model with augmentation layers
        """
        try:
            import tensorflow as tf
            from tensorflow import keras
            
            data_augmentation = keras.Sequential([
                keras.layers.RandomFlip("horizontal_and_vertical"),
                keras.layers.RandomRotation(0.0417),  # 15 degrees / 360
                keras.layers.RandomZoom(0.2),
                keras.layers.RandomContrast(0.2),
                keras.layers.GaussianNoise(0.01),
            ], name='data_augmentation')
            
            return data_augmentation
            
        except ImportError:
            print("TensorFlow not installed. Please install tensorflow to use this method.")
            return None
    
    def visualize_augmentations(self, image, num_augmentations=5, save_path=None):
        """
        Visualize multiple augmentations of a single image.
        
        Args:
            image (numpy.ndarray): Input image
            num_augmentations (int): Number of augmented versions to create
            save_path (str, optional): Path to save visualization
        """
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, num_augmentations + 1, figsize=(15, 3))
            
            # Display original image
            axes[0].imshow(image)
            axes[0].set_title('Original')
            axes[0].axis('off')
            
            # Display augmented versions
            for i in range(num_augmentations):
                augmented = self.augment_image(image, mode='train')
                axes[i + 1].imshow(augmented)
                axes[i + 1].set_title(f'Augmented {i+1}')
                axes[i + 1].axis('off')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Visualization saved to {save_path}")
            
            plt.show()
            
        except ImportError:
            print("Matplotlib not installed. Please install matplotlib to visualize.")


class TFDataAugmentor:
    """TensorFlow-native data augmentation for use in tf.data pipeline."""
    
    @staticmethod
    def get_augmentation_layer(image_size=224):
        """
        Create TensorFlow augmentation layer.
        
        Args:
            image_size (int): Size of input images
            
        Returns:
            tf.keras.Sequential: Augmentation layer
        """
        try:
            import tensorflow as tf
            from tensorflow import keras
            
            return keras.Sequential([
                keras.layers.RandomFlip("horizontal_and_vertical"),
                keras.layers.RandomRotation(0.0417),  # 15/360 for ±15 degrees
                keras.layers.RandomZoom(0.2),
                keras.layers.RandomContrast(0.2),
                keras.layers.RandomBrightness(0.2),
                keras.layers.GaussianNoise(0.01),
            ], name='tf_data_augmentation')
            
        except ImportError:
            print("TensorFlow not installed.")
            return None
    
    @staticmethod
    def augment_fn(image, label):
        """
        Augmentation function for use in tf.data.Dataset.map()
        
        Args:
            image: Input image tensor
            label: Image label
            
        Returns:
            tuple: (augmented_image, label)
        """
        try:
            import tensorflow as tf
            
            # Random horizontal flip
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_flip_up_down(image)
            
            # Random rotation (approximately ±15 degrees)
            image = tf.image.rot90(image, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
            
            # Random brightness and contrast
            image = tf.image.random_brightness(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
            
            # Clip values to valid range
            image = tf.clip_by_value(image, 0.0, 1.0)
            
            return image, label
            
        except ImportError:
            print("TensorFlow not installed.")
            return image, label


if __name__ == "__main__":
    print("Data Augmentation Module")
    print("=" * 50)
    
    # Example usage with dummy image
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Create augmentor
    augmentor = DataAugmentor(image_size=224)
    
    # Apply augmentation
    augmented = augmentor.augment_image(dummy_image, mode='train')
    
    print(f"Original shape: {dummy_image.shape}")
    print(f"Augmented shape: {augmented.shape}")
    print("\nAugmentation pipeline created successfully!")
    print("\nAvailable transformations:")
    print("- RandomFlip (horizontal/vertical)")
    print("- RandomRotation (±15°)")
    print("- RandomZoom (0.2)")
    print("- RandomBrightness/Contrast")
    print("- Gaussian noise")
    print("- Elastic transform")
    print("- Grid distortion")
    print("- Optical distortion")
