"""
Model trainer class for unified training interface.
"""

import math
from tensorflow.keras.optimizers import Adam
from .callbacks import get_training_callbacks


class Trainer:
    """Unified trainer for all model types."""
    
    def __init__(self, model, model_name: str, save_dir: str):
        """
        Initialize trainer.
        
        Args:
            model: Keras model to train
            model_name: Name for saving the model
            save_dir: Directory to save model and logs
        """
        self.model = model
        self.model_name = model_name
        self.save_dir = save_dir
        self.history = None
        
    def compile(self, learning_rate: float = 1e-4):
        """Compile the model with Adam optimizer."""
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
    def train(
        self,
        train_gen,
        val_gen,
        epochs: int,
        callbacks: list = None
    ):
        """
        Train the model.
        
        Args:
            train_gen: Training data generator
            val_gen: Validation data generator
            epochs: Number of epochs
            callbacks: Optional list of callbacks
            
        Returns:
            Training history
        """
        if callbacks is None:
            model_path = f"{self.save_dir}/{self.model_name}.keras"
            callbacks = get_training_callbacks(model_path)
        
        steps_per_epoch = math.ceil(train_gen.samples / train_gen.batch_size)
        validation_steps = math.ceil(val_gen.samples / val_gen.batch_size)
        
        self.history = self.model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=val_gen,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def save(self, path: str = None):
        """Save the model."""
        if path is None:
            path = f"{self.save_dir}/{self.model_name}.keras"
        self.model.save(path)
        print(f"Model saved to {path}")
        
    def get_history(self):
        """Get training history as dict."""
        if self.history:
            return self.history.history
        return None
