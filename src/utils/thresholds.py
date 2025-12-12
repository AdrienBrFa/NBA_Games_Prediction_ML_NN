"""
Threshold optimization utilities for binary classification.

This module provides a generic threshold optimizer that can maximize
different metrics (F1, accuracy, balanced accuracy) on validation data.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix
)
from typing import Tuple, Dict, Optional


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = "f1",
    threshold_grid: Optional[np.ndarray] = None
) -> Tuple[float, Dict[str, np.ndarray]]:
    """
    Find optimal classification threshold by maximizing a metric.
    
    Args:
        y_true: True binary labels (0 or 1)
        y_proba: Predicted probabilities (between 0 and 1)
        metric: Metric to optimize. One of:
            - "f1": F1 score (default)
            - "accuracy": Accuracy
            - "balanced_accuracy": Balanced accuracy (good for imbalanced data)
        threshold_grid: Array of thresholds to test. 
            Default: np.arange(0.05, 0.96, 0.01)
    
    Returns:
        best_threshold: Threshold that maximizes the metric
        metric_values: Dictionary with:
            - "thresholds": array of tested thresholds
            - "metric_scores": array of metric values at each threshold
            - "best_score": best metric value achieved
    
    Examples:
        >>> y_true = np.array([0, 1, 1, 0, 1])
        >>> y_proba = np.array([0.3, 0.8, 0.6, 0.2, 0.9])
        >>> threshold, info = find_optimal_threshold(y_true, y_proba, metric="f1")
        >>> print(f"Best threshold: {threshold:.3f}")
        >>> print(f"Best F1 score: {info['best_score']:.3f}")
    
    Notes:
        - In case of ties (multiple thresholds with same metric value),
          chooses the threshold closest to 0.50 for stability.
        - All thresholds in [0.05, 0.95] to avoid extreme values.
    """
    # Default threshold grid
    if threshold_grid is None:
        threshold_grid = np.arange(0.05, 0.96, 0.01)
    
    # Validate inputs
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    
    if len(y_true) != len(y_proba):
        raise ValueError(f"y_true and y_proba must have same length. "
                        f"Got {len(y_true)} and {len(y_proba)}")
    
    if not np.all((y_proba >= 0) & (y_proba <= 1)):
        raise ValueError("y_proba must be between 0 and 1")
    
    if metric not in ["f1", "accuracy", "balanced_accuracy"]:
        raise ValueError(f"Invalid metric: {metric}. "
                        f"Must be 'f1', 'accuracy', or 'balanced_accuracy'")
    
    # Select metric function
    if metric == "f1":
        metric_func = lambda yt, yp: f1_score(yt, yp, zero_division=0)
    elif metric == "accuracy":
        metric_func = accuracy_score
    else:  # balanced_accuracy
        metric_func = balanced_accuracy_score
    
    # Compute metric for each threshold
    metric_scores = []
    for threshold in threshold_grid:
        y_pred = (y_proba >= threshold).astype(int)
        score = metric_func(y_true, y_pred)
        metric_scores.append(score)
    
    metric_scores = np.array(metric_scores)
    
    # Find best threshold
    max_score = np.max(metric_scores)
    
    # In case of ties, choose threshold closest to 0.50
    best_indices = np.where(metric_scores == max_score)[0]
    if len(best_indices) > 1:
        # Multiple thresholds with same score - pick closest to 0.5
        distances_to_half = np.abs(threshold_grid[best_indices] - 0.5)
        best_idx = best_indices[np.argmin(distances_to_half)]
    else:
        best_idx = best_indices[0]
    
    best_threshold = float(threshold_grid[best_idx])
    
    # Return results
    metric_values = {
        "thresholds": threshold_grid,
        "metric_scores": metric_scores,
        "best_score": max_score,
        "best_index": best_idx
    }
    
    return best_threshold, metric_values


def compare_thresholds(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: Dict[str, float]
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple thresholds across different metrics.
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities
        thresholds: Dictionary mapping threshold names to values
            e.g., {"f1_optimal": 0.45, "accuracy_optimal": 0.50}
    
    Returns:
        Dictionary mapping threshold names to metric dictionaries:
        {
            "f1_optimal": {
                "accuracy": 0.62,
                "f1": 0.71,
                "balanced_accuracy": 0.58,
                ...
            },
            ...
        }
    """
    results = {}
    
    for name, threshold in thresholds.items():
        y_pred = (y_proba >= threshold).astype(int)
        
        results[name] = {
            "threshold": threshold,
            "accuracy": accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
        }
    
    return results


if __name__ == "__main__":
    # Simple test
    print("Testing threshold optimizer...")
    
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    y_proba = np.random.rand(100)
    
    for metric in ["f1", "accuracy", "balanced_accuracy"]:
        threshold, info = find_optimal_threshold(y_true, y_proba, metric=metric)
        print(f"\n{metric.upper()}:")
        print(f"  Best threshold: {threshold:.3f}")
        print(f"  Best score: {info['best_score']:.3f}")
    
    print("\n✓ Threshold optimizer working correctly")
