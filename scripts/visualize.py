"""
Visualization utilities for NBA game prediction ML pipeline.

This module provides comprehensive visualization functions for:
- Training monitoring (loss/accuracy curves)
- Model performance analysis (confusion matrix, ROC, PR curves)
- Feature analysis (importance, distributions)
- Data exploration and error analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import json
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    average_precision_score
)
from tensorflow import keras

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


def plot_training_history(
    history: keras.callbacks.History,
    save_path: str = "outputs/plots/training_history.png",
    show: bool = False
):
    """
    Plot training and validation loss and accuracy over epochs.
    
    Args:
        history: Keras training history object
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Plot accuracy
    axes[1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str] = ['Away Win', 'Home Win'],
    save_path: str = "outputs/plots/confusion_matrix.png",
    show: bool = False
):
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels (binary)
        labels: Class labels for display
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', 
        xticklabels=labels, yticklabels=labels,
        cbar_kws={'label': 'Count'},
        ax=ax
    )
    
    # Add percentage annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j + 0.5, i + 0.7, f'({cm_percent[i, j]:.1f}%)',
                ha='center', va='center', fontsize=10, color='gray'
            )
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    save_path: str = "outputs/plots/roc_curve.png",
    show: bool = False
):
    """
    Plot ROC curve with AUC score.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
            label='Random Classifier')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"ROC curve saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    save_path: str = "outputs/plots/precision_recall_curve.png",
    show: bool = False
):
    """
    Plot Precision-Recall curve.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    avg_precision = average_precision_score(y_true, y_pred_proba)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(recall, precision, color='blue', lw=2,
            label=f'PR curve (AP = {avg_precision:.3f})')
    ax.axhline(y=y_true.mean(), color='red', linestyle='--', lw=2,
               label=f'Baseline (AP = {y_true.mean():.3f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax.legend(loc="lower left", fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Precision-Recall curve saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_prediction_distribution(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    save_path: str = "outputs/plots/prediction_distribution.png",
    show: bool = False
):
    """
    Plot distribution of prediction probabilities by true class.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Separate predictions by true class
    proba_class_0 = y_pred_proba[y_true == 0]
    proba_class_1 = y_pred_proba[y_true == 1]
    
    ax.hist(proba_class_0, bins=50, alpha=0.6, label='True: Away Win', 
            color='blue', edgecolor='black')
    ax.hist(proba_class_1, bins=50, alpha=0.6, label='True: Home Win', 
            color='orange', edgecolor='black')
    ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, 
               label='Decision Threshold')
    
    ax.set_xlabel('Predicted Probability (Home Win)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Predicted Probabilities by True Class', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Prediction distribution saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_feature_correlations(
    X: np.ndarray,
    feature_names: List[str],
    save_path: str = "outputs/plots/feature_correlations.png",
    show: bool = False
):
    """
    Plot correlation matrix heatmap for features.
    
    Args:
        X: Feature matrix
        feature_names: List of feature names
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    # Calculate correlation matrix
    df = pd.DataFrame(X, columns=feature_names)
    corr_matrix = df.corr()
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
        center=0, vmin=-1, vmax=1, square=True,
        linewidths=0.5, cbar_kws={'label': 'Correlation'},
        ax=ax
    )
    
    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Feature correlation plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_metrics_comparison(
    train_metrics: Dict,
    val_metrics: Dict,
    test_metrics: Dict,
    save_path: str = "outputs/plots/metrics_comparison.png",
    show: bool = False
):
    """
    Compare metrics across train, validation, and test sets.
    
    Args:
        train_metrics: Training metrics dictionary
        val_metrics: Validation metrics dictionary
        test_metrics: Test metrics dictionary
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    metrics_to_plot = ['accuracy', 'auc', 'log_loss']
    sets = ['Train', 'Validation', 'Test']
    
    data = {
        'accuracy': [train_metrics['accuracy'], val_metrics['accuracy'], test_metrics['accuracy']],
        'auc': [train_metrics['auc'], val_metrics['auc'], test_metrics['auc']],
        'log_loss': [train_metrics['log_loss'], val_metrics['log_loss'], test_metrics['log_loss']]
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for idx, metric in enumerate(metrics_to_plot):
        axes[idx].bar(sets, data[metric], color=colors, alpha=0.7, edgecolor='black')
        axes[idx].set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        axes[idx].set_title(f'{metric.replace("_", " ").title()} by Dataset', 
                           fontsize=12, fontweight='bold')
        axes[idx].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, v in enumerate(data[metric]):
            axes[idx].text(i, v + 0.01, f'{v:.4f}', ha='center', 
                          va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Metrics comparison saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_class_balance(
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    save_path: str = "outputs/plots/class_balance.png",
    show: bool = False
):
    """
    Visualize class balance across train, validation, and test sets.
    
    Args:
        y_train: Training labels
        y_val: Validation labels
        y_test: Test labels
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    datasets = [
        ('Train', y_train),
        ('Validation', y_val),
        ('Test', y_test)
    ]
    
    for idx, (name, y) in enumerate(datasets):
        unique, counts = np.unique(y, return_counts=True)
        percentages = counts / counts.sum() * 100
        
        bars = axes[idx].bar(['Away Win', 'Home Win'], counts, 
                            color=['blue', 'orange'], alpha=0.7, edgecolor='black')
        axes[idx].set_ylabel('Count', fontsize=12)
        axes[idx].set_title(f'{name} Set Class Distribution', 
                           fontsize=12, fontweight='bold')
        axes[idx].grid(True, alpha=0.3, axis='y')
        
        # Add count and percentage labels
        for i, (bar, count, pct) in enumerate(zip(bars, counts, percentages)):
            axes[idx].text(bar.get_x() + bar.get_width()/2, count + 10,
                          f'{count}\n({pct:.1f}%)', ha='center', va='bottom',
                          fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Class balance plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_confidence_vs_accuracy(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 10,
    save_path: str = "outputs/plots/confidence_vs_accuracy.png",
    show: bool = False
):
    """
    Plot prediction confidence vs actual accuracy (calibration plot).
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        n_bins: Number of confidence bins
        save_path: Path to save the plot
        show: Whether to display the plot
    """
    # Bin predictions by confidence
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for i in range(n_bins):
        mask = (y_pred_proba >= bins[i]) & (y_pred_proba < bins[i + 1])
        if mask.sum() > 0:
            bin_acc = (y_true[mask] == (y_pred_proba[mask] >= 0.5)).mean()
            bin_conf = y_pred_proba[mask].mean()
            bin_accuracies.append(bin_acc)
            bin_confidences.append(bin_conf)
            bin_counts.append(mask.sum())
        else:
            bin_accuracies.append(0)
            bin_confidences.append(bin_centers[i])
            bin_counts.append(0)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot calibration
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
    ax.scatter(bin_confidences, bin_accuracies, s=[c*2 for c in bin_counts], 
              alpha=0.6, c=bin_counts, cmap='viridis', edgecolors='black')
    
    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Actual Accuracy', fontsize=12)
    ax.set_title('Calibration Plot: Confidence vs Accuracy', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.colorbar(ax.collections[0], ax=ax, label='Sample Count')
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confidence vs accuracy plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def generate_comprehensive_report(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    history: keras.callbacks.History,
    train_metrics: Dict,
    val_metrics: Dict,
    test_metrics: Dict,
    feature_names: Optional[List[str]] = None,
    output_dir: str = "outputs/plots",
    show: bool = False,
    X_train_home_team: Optional[np.ndarray] = None,
    X_train_away_team: Optional[np.ndarray] = None,
    X_val_home_team: Optional[np.ndarray] = None,
    X_val_away_team: Optional[np.ndarray] = None,
    X_test_home_team: Optional[np.ndarray] = None,
    X_test_away_team: Optional[np.ndarray] = None
):
    """
    Generate a comprehensive visualization report with all plots.
    
    Args:
        model: Trained Keras model
        X_train, y_train: Training data (numeric features for Model 2)
        X_val, y_val: Validation data (numeric features for Model 2)
        X_test, y_test: Test data (numeric features for Model 2)
        history: Training history
        train_metrics, val_metrics, test_metrics: Metrics dictionaries
        feature_names: List of feature names
        output_dir: Directory to save all plots
        show: Whether to display plots
        X_train_home_team, X_train_away_team: Team IDs for Model 2 (optional)
        X_val_home_team, X_val_away_team: Team IDs for Model 2 (optional)
        X_test_home_team, X_test_away_team: Team IDs for Model 2 (optional)
    """
    print("\n" + "="*80)
    print("GENERATING COMPREHENSIVE VISUALIZATION REPORT")
    print("="*80)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Training history
    print("\n1. Plotting training history...")
    plot_training_history(history, save_path=str(output_path / "training_history.png"), show=show)
    
    # 2. Get predictions for test set
    print("\n2. Generating predictions for visualization...")
    # Check if Model 2 (has team inputs)
    if X_test_home_team is not None and X_test_away_team is not None:
        # Ensure team inputs are 2D (batch, 1)
        home_ids = X_test_home_team if X_test_home_team.ndim == 2 else X_test_home_team.reshape(-1, 1)
        away_ids = X_test_away_team if X_test_away_team.ndim == 2 else X_test_away_team.reshape(-1, 1)
        y_test_pred_proba = model.predict([home_ids, away_ids, X_test], verbose=0).flatten()
    else:
        y_test_pred_proba = model.predict(X_test, verbose=0).flatten()
    y_test_pred = (y_test_pred_proba >= 0.5).astype(int)
    
    # 3. Confusion matrix
    print("\n3. Plotting confusion matrix...")
    plot_confusion_matrix(y_test, y_test_pred, 
                         save_path=str(output_path / "confusion_matrix.png"), show=show)
    
    # 4. ROC curve
    print("\n4. Plotting ROC curve...")
    plot_roc_curve(y_test, y_test_pred_proba, 
                   save_path=str(output_path / "roc_curve.png"), show=show)
    
    # 5. Precision-Recall curve
    print("\n5. Plotting Precision-Recall curve...")
    plot_precision_recall_curve(y_test, y_test_pred_proba,
                               save_path=str(output_path / "precision_recall_curve.png"), show=show)
    
    # 6. Prediction distribution
    print("\n6. Plotting prediction distribution...")
    plot_prediction_distribution(y_test, y_test_pred_proba,
                                save_path=str(output_path / "prediction_distribution.png"), show=show)
    
    # 7. Metrics comparison
    print("\n7. Plotting metrics comparison...")
    plot_metrics_comparison(train_metrics, val_metrics, test_metrics,
                           save_path=str(output_path / "metrics_comparison.png"), show=show)
    
    # 8. Class balance
    print("\n8. Plotting class balance...")
    plot_class_balance(y_train, y_val, y_test,
                      save_path=str(output_path / "class_balance.png"), show=show)
    
    # 9. Confidence vs accuracy
    print("\n9. Plotting confidence vs accuracy...")
    plot_confidence_vs_accuracy(y_test, y_test_pred_proba,
                               save_path=str(output_path / "confidence_vs_accuracy.png"), show=show)
    
    # Feature correlations plot removed due to high feature count in Stage B1
    # (would create unreadable heatmap with 260+ features)
    
    print("\n" + "="*80)
    print(f"VISUALIZATION REPORT COMPLETE")
    print(f"All plots saved to: {output_path}")
    print("="*80)


if __name__ == "__main__":
    # Example usage with loaded data
    print("Loading data for visualization examples...")
    
    # Load preprocessed data
    X_train = np.load("outputs/X_train.npy")
    y_train = np.load("outputs/y_train.npy")
    X_val = np.load("outputs/X_val.npy")
    y_val = np.load("outputs/y_val.npy")
    X_test = np.load("outputs/X_test.npy")
    y_test = np.load("outputs/y_test.npy")
    
    # Load model
    model = keras.models.load_model("models/stage_a1_mlp.keras")
    
    # Load results
    with open("outputs/results.json", 'r') as f:
        results = json.load(f)
    
    # Reconstruct history
    class FakeHistory:
        def __init__(self, history_dict):
            self.history = history_dict
    
    history = FakeHistory(results['training_history'])
    
    # Generate comprehensive report
    generate_comprehensive_report(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        history=history,
        train_metrics=results['train_metrics'],
        val_metrics=results['val_metrics'],
        test_metrics=results['test_metrics'],
        output_dir="outputs/plots"
    )
