"""
Baseline CNN Model for Image Classification.
Simple CNN architecture with Conv2D, BatchNormalization, MaxPooling, and Dense layers.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, 
    Dense, Dropout, BatchNormalization
)


def build_baseline_model(img_size: tuple, num_classes: int) -> Sequential:
    """
    Build a baseline CNN model.
    
    Architecture:
    - 3 Conv2D blocks with BatchNorm and MaxPooling
    - Flatten + Dense with Dropout
    - Softmax output
    
    Args:
        img_size: Tuple of (height, width) for input images
        num_classes: Number of output classes
        
    Returns:
        Compiled Keras Sequential model
    """
    model = Sequential([
        # Input layer
        Input(shape=(img_size[0], img_size[1], 3)),
        
        # Block 1
        Conv2D(32, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        # Block 2
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        # Block 3
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        # Classification head
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    return model
