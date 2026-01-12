"""
Grad-CAM implementation for model interpretability.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D


class GradCAM:
    """Grad-CAM visualization for CNN models."""
    
    def __init__(self, model, img_size: tuple):
        """
        Initialize Grad-CAM.
        
        Args:
            model: Trained Keras model
            img_size: Input image size (height, width)
        """
        self.model = model
        self.img_size = img_size
        self.grad_model = self._build_grad_model()
    
    def _build_grad_model(self):
        """Build the gradient model for Grad-CAM."""
        inputs_fn = tf.keras.Input(shape=self.img_size + (3,))
        x = inputs_fn
        last_conv_out = None
        
        for layer in self.model.layers:
            x = layer(x)
            if isinstance(layer, Conv2D):
                last_conv_out = x
        
        if last_conv_out is None:
            raise ValueError("No Conv2D layer found in model")
        
        functional_model = tf.keras.Model(inputs=inputs_fn, outputs=x)
        
        return tf.keras.Model(
            inputs=functional_model.input,
            outputs=[last_conv_out, functional_model.output]
        )
    
    def compute_heatmap(self, image: np.ndarray) -> np.ndarray:
        """
        Compute Grad-CAM heatmap for a single image.
        
        Args:
            image: Input image (normalized, shape: H, W, 3)
            
        Returns:
            Heatmap array (same size as input image)
        """
        img_tensor = tf.convert_to_tensor(image[None, ...], dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            conv_out, preds = self.grad_model(img_tensor, training=False)
            class_idx = tf.argmax(preds[0])
            loss = preds[:, class_idx]
        
        grads = tape.gradient(loss, conv_out)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_out_0 = conv_out[0]
        heatmap = conv_out_0 @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
        
        heatmap_resized = tf.image.resize(
            heatmap[..., tf.newaxis], self.img_size
        ).numpy().squeeze()
        
        return heatmap_resized
    
    def get_predicted_class(self, image: np.ndarray) -> int:
        """Get predicted class index for an image."""
        img_tensor = tf.convert_to_tensor(image[None, ...], dtype=tf.float32)
        _, preds = self.grad_model(img_tensor, training=False)
        return int(tf.argmax(preds[0]))
