"""
Training callbacks for model training.
"""

from tensorflow.keras.callbacks import (
    EarlyStopping, 
    ReduceLROnPlateau, 
    ModelCheckpoint,
    TensorBoard,
    CSVLogger
)
from pathlib import Path
import datetime


def get_training_callbacks(
    model_path: str,
    log_dir: str = None,
    patience_early_stop: int = 5,
    patience_reduce_lr: int = 2,
    min_lr: float = 1e-6
) -> list:
    """
    Get standard training callbacks.
    
    Args:
        model_path: Path to save the best model
        log_dir: Directory for TensorBoard logs
        patience_early_stop: Patience for early stopping
        patience_reduce_lr: Patience for learning rate reduction
        min_lr: Minimum learning rate
        
    Returns:
        List of Keras callbacks
    """
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=patience_early_stop,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience_reduce_lr,
            min_lr=min_lr,
            verbose=1
        ),
        ModelCheckpoint(
            model_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
    ]
    
    # Add TensorBoard if log_dir provided
    if log_dir:
        log_path = Path(log_dir) / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        callbacks.append(TensorBoard(log_dir=str(log_path), histogram_freq=1))
    
    return callbacks


def get_csv_logger(log_path: str) -> CSVLogger:
    """Get a CSV logger callback."""
    return CSVLogger(log_path, append=True)
