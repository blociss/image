"""
Data loading utilities for image classification.
Handles data augmentation and generator creation.
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

from .config import BATCH_SIZE, SEED


def create_baseline_generators(train_dir, test_dir, img_size, batch_size=BATCH_SIZE):
    """
    Create data generators for baseline/regularized models.
    
    Args:
        train_dir: Path to training data directory
        test_dir: Path to test data directory
        img_size: Tuple of (height, width)
        batch_size: Batch size for generators
        
    Returns:
        Tuple of (train_generator, val_generator, test_generator)
    """
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        brightness_range=[0.8, 1.2],
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )
    
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
    )
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        seed=SEED,
    )
    
    val_generator = val_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        seed=SEED,
    )
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
    )
    
    return train_generator, val_generator, test_generator


def create_transfer_learning_generators(train_dir, test_dir, img_size, batch_size=BATCH_SIZE):
    """
    Create data generators for transfer learning with MobileNetV2 preprocessing.
    
    Args:
        train_dir: Path to training data directory
        test_dir: Path to test data directory
        img_size: Tuple of (height, width) - should be (224, 224)
        batch_size: Batch size for generators
        
    Returns:
        Tuple of (train_generator, val_generator, test_generator)
    """
    train_datagen = ImageDataGenerator(
        preprocessing_function=mobilenet_preprocess,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
    )
    
    val_datagen = ImageDataGenerator(
        preprocessing_function=mobilenet_preprocess,
        validation_split=0.2,
    )
    
    test_datagen = ImageDataGenerator(
        preprocessing_function=mobilenet_preprocess
    )
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        seed=SEED,
    )
    
    val_generator = val_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        seed=SEED,
    )
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
    )
    
    return train_generator, val_generator, test_generator


def get_class_mapping(generator):
    """Get class index to name mapping from a generator."""
    return {v: k for k, v in generator.class_indices.items()}
