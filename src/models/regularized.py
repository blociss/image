"""
Regularized CNN Model for Image Classification.
CNN with L2 regularization to prevent overfitting.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten,
    Dense, Dropout, BatchNormalization
)
from tensorflow.keras import regularizers


def build_regularized_model(
    img_size: tuple, 
    num_classes: int,
    l2_lambda: float = 1e-4,
    dropout_rate: float = 0.5
) -> Sequential:
    """
    Build a regularized CNN model with L2 regularization.
    
    Args:
        img_size: Tuple of (height, width) for input images
        num_classes: Number of output classes
        l2_lambda: L2 regularization strength
        dropout_rate: Dropout rate for Dense layers
        
    Returns:
        Keras Sequential model
    """
    l2 = regularizers.l2(l2_lambda)
    
    model = Sequential([
        # Input layer
        Input(shape=(img_size[0], img_size[1], 3)),
        
        # Block 1
        Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        # Block 2
        Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        # Block 3
        Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        # Classification head
        Flatten(),
        Dense(512, activation='relu', kernel_regularizer=l2),
        Dropout(dropout_rate),
        Dense(num_classes, activation='softmax')
    ])
    
    return model
