"""
Visualization utilities for model evaluation.
All plots are saved to files instead of displaying interactively.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from pathlib import Path


def plot_training_curves(history, save_path: str, title_prefix: str = ""):
    """
    Plot and save training/validation accuracy and loss curves.
    
    Args:
        history: Keras training history or history dict
        save_path: Path to save the figure
        title_prefix: Prefix for plot titles
    """
    if hasattr(history, 'history'):
        history = history.history
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    axes[0].plot(history.get('accuracy', []), label='Train Accuracy', linewidth=2)
    axes[0].plot(history.get('val_accuracy', []), label='Val Accuracy', linewidth=2)
    axes[0].set_title(f'{title_prefix} Accuracy', fontsize=14)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss
    axes[1].plot(history.get('loss', []), label='Train Loss', linewidth=2)
    axes[1].plot(history.get('val_loss', []), label='Val Loss', linewidth=2)
    axes[1].set_title(f'{title_prefix} Loss', fontsize=14)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_confusion_matrix(cm, class_names: list, save_path: str, title: str = "Confusion Matrix"):
    """
    Plot and save confusion matrix.
    
    Args:
        cm: Confusion matrix array
        class_names: List of class names
        save_path: Path to save the figure
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_class_accuracy(report: dict, class_names: list, save_path: str):
    """
    Plot per-class accuracy/F1 scores.
    
    Args:
        report: Classification report dictionary
        class_names: List of class names
        save_path: Path to save the figure
    """
    metrics = ['precision', 'recall', 'f1-score']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(class_names))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        values = [report.get(cls, {}).get(metric, 0) for cls in class_names]
        ax.bar(x + i * width, values, width, label=metric.capitalize())
    
    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Metrics')
    ax.set_xticks(x + width)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_grad_cam(model, images, img_size: tuple, save_path: str, n: int = 4):
    """
    Generate and save Grad-CAM visualizations.
    
    Args:
        model: Trained Keras model
        images: Batch of images (normalized)
        img_size: Image size tuple
        save_path: Path to save the figure
        n: Number of images to visualize
    """
    # Reconstruct functional graph
    inputs_fn = tf.keras.Input(shape=img_size + (3,))
    x = inputs_fn
    last_conv_out = None
    
    for layer in model.layers:
        x = layer(x)
        if isinstance(layer, Conv2D):
            last_conv_out = x
    
    if last_conv_out is None:
        print("No Conv2D layer found for Grad-CAM.")
        return
    
    functional_model = tf.keras.Model(inputs=inputs_fn, outputs=x)
    grad_model = tf.keras.Model(
        inputs=functional_model.input,
        outputs=[last_conv_out, functional_model.output]
    )
    
    n = min(n, len(images))
    fig, axes = plt.subplots(n, 2, figsize=(8, 4 * n))
    if n == 1:
        axes = np.array([axes])
    
    for i in range(n):
        img = images[i]
        img_tensor = tf.convert_to_tensor(img[None, ...], dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(img_tensor, training=False)
            class_idx = tf.argmax(preds[0])
            loss_gc = preds[:, class_idx]
        
        grads = tape.gradient(loss_gc, conv_out)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_out_0 = conv_out[0]
        heatmap = conv_out_0 @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
        
        heatmap_resized = tf.image.resize(
            heatmap[..., tf.newaxis], img_size
        ).numpy().squeeze()
        
        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Original Image")
        axes[i, 0].axis("off")
        
        axes[i, 1].imshow(img)
        axes[i, 1].imshow(heatmap_resized, cmap="jet", alpha=0.4)
        axes[i, 1].set_title("Grad-CAM")
        axes[i, 1].axis("off")
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_model_comparison(results: dict, save_path: str):
    """
    Plot comparison of multiple models.
    
    Args:
        results: Dict with model names as keys and accuracy as values
        save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = list(results.keys())
    accuracies = list(results.values())
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(models)))
    
    bars = ax.bar(models, accuracies, color=colors)
    
    for bar, acc in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f'{acc:.2%}',
            ha='center',
            fontsize=12,
            fontweight='bold'
        )
    
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Model Comparison')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
