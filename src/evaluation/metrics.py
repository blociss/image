"""
Evaluation metrics for model assessment.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


def evaluate_model(model, test_gen, batch_size: int = 32):
    """
    Evaluate model on test data.
    
    Args:
        model: Trained Keras model
        test_gen: Test data generator
        batch_size: Batch size for prediction
        
    Returns:
        Dictionary with evaluation results
    """
    # Reset generator
    test_gen.reset()
    
    # Predict
    y_prob = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = test_gen.classes
    
    # Align lengths if needed
    n = min(len(y_true), len(y_pred))
    y_true = y_true[:n]
    y_pred = y_pred[:n]
    
    # Get class mapping
    idx_to_class = {v: k for k, v in test_gen.class_indices.items()}
    
    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    labels = list(range(len(idx_to_class)))
    target_names = [idx_to_class[i] for i in labels]
    
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )
    
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'accuracy': accuracy,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob[:n],
        'report': report,
        'confusion_matrix': cm,
        'class_mapping': idx_to_class
    }


def compute_metrics(report: dict):
    """
    Extract key metrics from classification report.
    
    Args:
        report: Classification report dictionary
        
    Returns:
        Dictionary with macro and weighted averages
    """
    macro = report.get('macro avg', {})
    weighted = report.get('weighted avg', {})
    
    return {
        'macro_precision': macro.get('precision', 0.0),
        'macro_recall': macro.get('recall', 0.0),
        'macro_f1': macro.get('f1-score', 0.0),
        'weighted_precision': weighted.get('precision', 0.0),
        'weighted_recall': weighted.get('recall', 0.0),
        'weighted_f1': weighted.get('f1-score', 0.0)
    }
