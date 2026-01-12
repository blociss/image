#!/usr/bin/env python3
"""
Main Training Pipeline Script
==============================
Runs the complete training pipeline for all models:
1. Baseline CNN
2. Regularized CNN
3. Transfer Learning (MobileNetV2)

All outputs are saved with timestamps for versioning:
- Models: outputs/models/{model_type}_{timestamp}.keras
- Figures: outputs/figures/{run_timestamp}/{figure_name}.png
- Logs: outputs/logs/training_{timestamp}.log

Usage:
    python scripts/train_pipeline.py                    # Train all models
    python scripts/train_pipeline.py --speed-mode       # Quick test run (all models, fewer epochs)
    python scripts/train_pipeline.py --baseline-only    # Train only baseline
    python scripts/train_pipeline.py --regularized-only # Train only regularized
    python scripts/train_pipeline.py --tl-only          # Train only transfer learning
    
    # Combine flags:
    python scripts/train_pipeline.py --regularized-only --speed-mode
    python scripts/train_pipeline.py --tl-only --speed-mode
"""

import sys
import argparse
import logging
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    ensure_dirs, get_train_dir, get_test_dir,
    IMG_SIZE, TL_IMG_SIZE, BATCH_SIZE, NUM_CLASSES,
    BASELINE_EPOCHS, BASELINE_LR,
    REGULARIZED_EPOCHS, REGULARIZED_LR,
    TL_INITIAL_EPOCHS, TL_FINETUNE_EPOCHS, TL_HEAD_LR, TL_FINETUNE_LR,
    TL_UNFREEZE_LAYERS, L2_LAMBDA, DROPOUT_RATE,
    MODELS_DIR, FIGURES_DIR, LOGS_DIR, CLASS_NAMES
)
from src.data_loader import (
    create_baseline_generators,
    create_transfer_learning_generators,
    get_class_mapping
)
from src.models import (  # Das is von __init__.py file. 
    build_baseline_model,
    build_regularized_model,
    build_transfer_learning_model
)
from src.models.transfer_learning import unfreeze_layers
from src.training import Trainer, get_training_callbacks
from src.evaluation import (
    evaluate_model, compute_metrics,
    plot_training_curves, plot_confusion_matrix,
    plot_class_accuracy, plot_grad_cam, plot_model_comparison
)

# Global timestamp for this run
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create run-specific directories
RUN_FIGURES_DIR = FIGURES_DIR / RUN_TIMESTAMP
RUN_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / f'training_{RUN_TIMESTAMP}.log')
    ]
)
logger = logging.getLogger(__name__)


def save_run_metadata(model_results: dict, run_dir: Path):
    """Save metadata about this training run."""
    metadata = {
        "timestamp": RUN_TIMESTAMP,
        "date": datetime.now().isoformat(),
        "models": {},
        "config": {
            "batch_size": BATCH_SIZE,
            "img_size": IMG_SIZE,
            "tl_img_size": TL_IMG_SIZE,
            "num_classes": NUM_CLASSES,
            "class_names": CLASS_NAMES
        }
    }
    for model_name, info in model_results.items():
        metadata["models"][model_name] = {
            "accuracy": info["accuracy"],
            "model_file": info["model_file"],
            "figures": info.get("figures", [])
        }
    
    metadata_path = run_dir / "run_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved run metadata to {metadata_path}")


def train_baseline(train_gen, val_gen, test_gen, epochs: int = BASELINE_EPOCHS):
    """Train baseline CNN model."""
    logger.info("=" * 60)
    logger.info("TRAINING BASELINE MODEL")
    logger.info("=" * 60)
    
    # Model filename with timestamp
    model_filename = f"baseline_{RUN_TIMESTAMP}.keras"
    model_path = MODELS_DIR / model_filename
    
    # Build model
    model = build_baseline_model(IMG_SIZE, NUM_CLASSES)
    
    # Create trainer
    trainer = Trainer(model, f"baseline_{RUN_TIMESTAMP}", str(MODELS_DIR))
    trainer.compile(learning_rate=BASELINE_LR)
    
    # Train
    callbacks = get_training_callbacks(str(model_path), str(LOGS_DIR))
    history = trainer.train(train_gen, val_gen, epochs, callbacks)
    
    # Save model
    trainer.save()
    
    # Figures list for metadata
    figures = []
    
    # Plot training curves
    fig_path = str(RUN_FIGURES_DIR / "baseline_training_curves.png")
    plot_training_curves(history, fig_path, "Baseline")
    figures.append(fig_path)
    
    # Evaluate
    results = evaluate_model(model, test_gen, BATCH_SIZE)
    logger.info(f"Baseline Test Accuracy: {results['accuracy']:.4f}")
    
    # Plot confusion matrix
    fig_path = str(RUN_FIGURES_DIR / "baseline_confusion_matrix.png")
    plot_confusion_matrix(
        results['confusion_matrix'], CLASS_NAMES, fig_path, "Baseline Confusion Matrix"
    )
    figures.append(fig_path)
    
    # Plot per-class metrics
    fig_path = str(RUN_FIGURES_DIR / "baseline_class_metrics.png")
    plot_class_accuracy(results['report'], CLASS_NAMES, fig_path)
    figures.append(fig_path)
    
    # Grad-CAM
    val_gen.reset()
    images, _ = next(val_gen)
    fig_path = str(RUN_FIGURES_DIR / "baseline_grad_cam.png")
    plot_grad_cam(model, images, IMG_SIZE, fig_path, n=4)
    figures.append(fig_path)
    
    return model, {
        "accuracy": results['accuracy'],
        "model_file": model_filename,
        "figures": figures
    }


def train_regularized(train_gen, val_gen, test_gen, epochs: int = REGULARIZED_EPOCHS):
    """Train regularized CNN model."""
    logger.info("=" * 60)
    logger.info("TRAINING REGULARIZED MODEL")
    logger.info("=" * 60)
    
    # Model filename with timestamp
    model_filename = f"regularized_{RUN_TIMESTAMP}.keras"
    model_path = MODELS_DIR / model_filename
    
    # Build model
    model = build_regularized_model(
        IMG_SIZE, NUM_CLASSES,
        l2_lambda=L2_LAMBDA,
        dropout_rate=DROPOUT_RATE
    )
    
    # Create trainer
    trainer = Trainer(model, f"regularized_{RUN_TIMESTAMP}", str(MODELS_DIR))
    trainer.compile(learning_rate=REGULARIZED_LR)
    
    # Train
    callbacks = get_training_callbacks(str(model_path), str(LOGS_DIR))
    history = trainer.train(train_gen, val_gen, epochs, callbacks)
    
    # Save model
    trainer.save()
    
    # Figures list for metadata
    figures = []
    
    # Plot training curves
    fig_path = str(RUN_FIGURES_DIR / "regularized_training_curves.png")
    plot_training_curves(history, fig_path, "Regularized")
    figures.append(fig_path)
    
    # Evaluate
    results = evaluate_model(model, test_gen, BATCH_SIZE)
    logger.info(f"Regularized Test Accuracy: {results['accuracy']:.4f}")
    
    # Plot confusion matrix
    fig_path = str(RUN_FIGURES_DIR / "regularized_confusion_matrix.png")
    plot_confusion_matrix(
        results['confusion_matrix'], CLASS_NAMES, fig_path, "Regularized Confusion Matrix"
    )
    figures.append(fig_path)
    
    # Plot per-class metrics
    fig_path = str(RUN_FIGURES_DIR / "regularized_class_metrics.png")
    plot_class_accuracy(results['report'], CLASS_NAMES, fig_path)
    figures.append(fig_path)
    
    return model, {
        "accuracy": results['accuracy'],
        "model_file": model_filename,
        "figures": figures
    }


def train_transfer_learning(train_dir, test_dir, speed_mode: bool = False):
    """Train transfer learning model with MobileNetV2."""
    logger.info("=" * 60)
    logger.info("TRAINING TRANSFER LEARNING MODEL (MobileNetV2)")
    logger.info("=" * 60)
    
    # Adjust epochs for speed mode
    initial_epochs = 2 if speed_mode else TL_INITIAL_EPOCHS
    finetune_epochs = 2 if speed_mode else TL_FINETUNE_EPOCHS
    
    if speed_mode:
        logger.info(f"SPEED MODE: Using {initial_epochs} initial + {finetune_epochs} finetune epochs")
    
    # Model filename with timestamp
    model_filename = f"tl_{RUN_TIMESTAMP}.keras"
    model_path = MODELS_DIR / model_filename
    
    # Create TL generators
    train_gen, val_gen, test_gen = create_transfer_learning_generators(
        train_dir, test_dir, TL_IMG_SIZE, BATCH_SIZE
    )
    
    num_classes = len(train_gen.class_indices)
    
    # Build model
    model, base_model = build_transfer_learning_model(
        TL_IMG_SIZE, num_classes,
        dropout_rate=0.3,
        trainable_base=False
    )
    
    # Phase 1: Train head only
    logger.info("Phase 1: Training classification head (frozen backbone)")
    trainer = Trainer(model, f"tl_{RUN_TIMESTAMP}", str(MODELS_DIR))
    trainer.compile(learning_rate=TL_HEAD_LR)
    
    callbacks = get_training_callbacks(str(model_path), str(LOGS_DIR))
    history_head = trainer.train(train_gen, val_gen, initial_epochs, callbacks)
    
    # Figures list for metadata
    figures = []
    
    # Plot head training curves
    fig_path = str(RUN_FIGURES_DIR / "tl_head_training_curves.png")
    plot_training_curves(history_head, fig_path, "Transfer Learning (Head)")
    figures.append(fig_path)
    
    # Phase 2: Fine-tuning
    logger.info("Phase 2: Fine-tuning last layers")
    unfreeze_layers(base_model, TL_UNFREEZE_LAYERS)
    
    trainer.compile(learning_rate=TL_FINETUNE_LR)
    history_ft = trainer.train(train_gen, val_gen, finetune_epochs, callbacks)
    
    # Save final model
    model.save(str(model_path))
    logger.info(f"Model saved to {model_path}")
    
    # Plot fine-tuning curves
    fig_path = str(RUN_FIGURES_DIR / "tl_finetune_training_curves.png")
    plot_training_curves(history_ft, fig_path, "Transfer Learning (Fine-tune)")
    figures.append(fig_path)
    
    # Evaluate
    results = evaluate_model(model, test_gen, BATCH_SIZE)
    logger.info(f"Transfer Learning Test Accuracy: {results['accuracy']:.4f}")
    
    # Plot confusion matrix
    fig_path = str(RUN_FIGURES_DIR / "tl_confusion_matrix.png")
    plot_confusion_matrix(
        results['confusion_matrix'], CLASS_NAMES, fig_path, "Transfer Learning Confusion Matrix"
    )
    figures.append(fig_path)
    
    # Plot per-class metrics
    fig_path = str(RUN_FIGURES_DIR / "tl_class_metrics.png")
    plot_class_accuracy(results['report'], CLASS_NAMES, fig_path)
    figures.append(fig_path)
    
    return model, {
        "accuracy": results['accuracy'],
        "model_file": model_filename,
        "figures": figures
    }


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description="Image Classification Training Pipeline")
    parser.add_argument('--speed-mode', action='store_true', help='Quick test run with fewer epochs')
    parser.add_argument('--baseline-only', action='store_true', help='Train only baseline model')
    parser.add_argument('--regularized-only', action='store_true', help='Train only regularized model')
    parser.add_argument('--tl-only', action='store_true', help='Train only transfer learning model')
    args = parser.parse_args()
    
    # Ensure directories exist
    ensure_dirs()
    
    logger.info("=" * 60)
    logger.info("IMAGE CLASSIFICATION TRAINING PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Project Root: {PROJECT_ROOT}")
    logger.info(f"Models Dir: {MODELS_DIR}")
    logger.info(f"Figures Dir: {FIGURES_DIR}")
    
    # Get data directories
    train_dir = get_train_dir()
    test_dir = get_test_dir()
    logger.info(f"Train Dir: {train_dir}")
    logger.info(f"Test Dir: {test_dir}")
    
    # Adjust epochs for speed mode
    baseline_epochs = 2 if args.speed_mode else BASELINE_EPOCHS
    regularized_epochs = 3 if args.speed_mode else REGULARIZED_EPOCHS
    
    # Results storage
    model_results = {}
    
    # Create baseline generators
    train_gen, val_gen, test_gen = create_baseline_generators(
        train_dir, test_dir, IMG_SIZE, BATCH_SIZE
    )
    
    logger.info(f"Classes: {get_class_mapping(train_gen)}")
    logger.info(f"Train samples: {train_gen.samples}")
    logger.info(f"Val samples: {val_gen.samples}")
    logger.info(f"Test samples: {test_gen.samples}")
    
    # Train models
    train_all = not (args.baseline_only or args.regularized_only or args.tl_only)
    
    if train_all or args.baseline_only:
        _, baseline_info = train_baseline(train_gen, val_gen, test_gen, baseline_epochs)
        model_results['Baseline'] = baseline_info
    
    if train_all or args.regularized_only:
        # Recreate generators (they get exhausted)
        train_gen, val_gen, test_gen = create_baseline_generators(
            train_dir, test_dir, IMG_SIZE, BATCH_SIZE
        )
        _, regularized_info = train_regularized(train_gen, val_gen, test_gen, regularized_epochs)
        model_results['Regularized'] = regularized_info
    
    if train_all or args.tl_only:
        _, tl_info = train_transfer_learning(train_dir, test_dir, speed_mode=args.speed_mode)
        model_results['Transfer Learning'] = tl_info
    
    # Plot model comparison (using accuracy values)
    if len(model_results) > 1:
        comparison_data = {k: v['accuracy'] for k, v in model_results.items()}
        fig_path = str(RUN_FIGURES_DIR / "model_comparison.png")
        plot_model_comparison(comparison_data, fig_path)
    
    # Save run metadata
    save_run_metadata(model_results, RUN_FIGURES_DIR)
    
    # Summary
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE - SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Run Timestamp: {RUN_TIMESTAMP}")
    for model_name, info in model_results.items():
        logger.info(f"{model_name}: {info['accuracy']:.4f} ({info['accuracy']*100:.2f}%) - {info['model_file']}")
    
    logger.info(f"\nAll figures saved to: {RUN_FIGURES_DIR}")
    logger.info(f"All models saved to: {MODELS_DIR}")
    logger.info(f"Logs saved to: {LOGS_DIR}")


if __name__ == "__main__":
    main()
