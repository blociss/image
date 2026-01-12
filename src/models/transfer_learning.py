"""
Transfer Learning Model using MobileNetV2.
Pre-trained on ImageNet, fine-tuned for custom classification.
"""

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    Input, GlobalAveragePooling2D, Dropout, Dense
)
from tensorflow.keras.models import Model


def build_transfer_learning_model(
    img_size: tuple,
    num_classes: int,
    dropout_rate: float = 0.3,
    trainable_base: bool = False
) -> Model:
    """
    Build a transfer learning model using MobileNetV2.
    
    Args:
        img_size: Tuple of (height, width) for input images (should be 224x224)
        num_classes: Number of output classes
        dropout_rate: Dropout rate before final layer
        trainable_base: Whether to make base model trainable
        
    Returns:
        Keras Model
    """
    # Load pre-trained MobileNetV2
    base_model = MobileNetV2(
        input_shape=(img_size[0], img_size[1], 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = trainable_base
    
    # Build model
    inputs = Input(shape=(img_size[0], img_size[1], 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    return model, base_model


def unfreeze_layers(base_model, num_layers: int = 30):
    """
    Unfreeze the last N layers of the base model for fine-tuning.
    
    Args:
        base_model: The base MobileNetV2 model
        num_layers: Number of layers to unfreeze from the end
    """
    base_model.trainable = True
    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False
