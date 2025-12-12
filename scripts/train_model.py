"""
Model definition and training for NBA game prediction (Stage A1).

This script defines a simple MLP model and trains it with early stopping.
"""

import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks  # type: ignore
from tensorflow.keras import regularizers  # type: ignore
from sklearn.metrics import confusion_matrix, log_loss, f1_score, roc_curve
from scripts.visualize import plot_training_history


def create_mlp_model(input_dim: int, l2_reg: float = 0.001, dropout_rate: float = 0.3) -> keras.Model:
    """
    Create a simple MLP model for binary classification with regularization.
    
    Architecture:
    - Input layer: size = input_dim
    - Hidden layer 1: Dense(64, activation="relu") + L2 regularization + Dropout
    - Hidden layer 2: Dense(32, activation="relu") + L2 regularization + Dropout
    - Output layer: Dense(1, activation="sigmoid")
    
    Args:
        input_dim: Number of input features
        l2_reg: L2 regularization strength
        dropout_rate: Dropout rate for regularization
        
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu', 
                    kernel_regularizer=regularizers.l2(l2_reg),
                    name='hidden1'),
        layers.Dropout(dropout_rate, name='dropout1'),
        layers.Dense(32, activation='relu', 
                    kernel_regularizer=regularizers.l2(l2_reg),
                    name='hidden2'),
        layers.Dropout(dropout_rate, name='dropout2'),
        layers.Dense(1, activation='sigmoid', name='output')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    return model


def train_model(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    epochs: int = 100,
    batch_size: int = 64,
    patience: int = 5,
    l2_reg: float = 0.001,
    dropout_rate: float = 0.3,
    model_save_path: str = "models/stage_a1_mlp.keras"
) -> tuple:
    """
    Train the MLP model with early stopping and regularization.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        epochs: Maximum number of epochs
        batch_size: Batch size for training
        patience: Early stopping patience (default: 5)
        l2_reg: L2 regularization strength
        dropout_rate: Dropout rate
        model_save_path: Path to save the best model
        
    Returns:
        Tuple of (trained_model, history)
    """
    print("Creating model...")
    input_dim = X_train.shape[1]
    model = create_mlp_model(input_dim, l2_reg=l2_reg, dropout_rate=dropout_rate)
    
    print(f"\nModel configuration:")
    print(f"  L2 regularization: {l2_reg}")
    print(f"  Dropout rate: {dropout_rate}")
    print(f"  Early stopping patience: {patience} epochs")
    
    print(f"\nModel architecture:")
    model.summary()
    
    # Set up callbacks
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )
    
    model_checkpoint = callbacks.ModelCheckpoint(
        model_save_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    
    # Train model
    print(f"\nTraining model...")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, model_checkpoint, reduce_lr],
        verbose=1
    )
    
    # Plot training history
    print("\nGenerating training history plots...")
    plot_training_history(history, save_path="outputs/plots/training_history.png", show=False)
    
    return model, history


def find_optimal_threshold(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    method: str = 'f1'
) -> tuple:
    """
    Find optimal decision threshold based on validation set.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        method: Method to use ('f1', 'youden', 'balanced')
            - 'f1': Maximize F1 score
            - 'youden': Maximize Youden's J statistic (sensitivity + specificity - 1)
            - 'balanced': Minimize difference between sensitivity and specificity
    
    Returns:
        Tuple of (optimal_threshold, metric_value)
    """
    thresholds = np.linspace(0.1, 0.9, 81)  # Test thresholds from 0.1 to 0.9
    best_threshold = 0.5
    best_metric = 0.0
    
    if method == 'f1':
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            f1 = f1_score(y_true, y_pred)
            if f1 > best_metric:
                best_metric = f1
                best_threshold = threshold
        print(f"\nOptimal threshold (F1): {best_threshold:.3f} (F1={best_metric:.4f})")
    
    elif method == 'youden':
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred_proba)
        # Youden's J statistic = Sensitivity + Specificity - 1 = TPR - FPR
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_threshold = roc_thresholds[best_idx]
        best_metric = j_scores[best_idx]
        print(f"\nOptimal threshold (Youden): {best_threshold:.3f} (J={best_metric:.4f})")
    
    elif method == 'balanced':
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            cm = confusion_matrix(y_true, y_pred)
            if cm.sum() > 0:
                tn, fp, fn, tp = cm.ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                balance_metric = 1 - abs(sensitivity - specificity)
                if balance_metric > best_metric:
                    best_metric = balance_metric
                    best_threshold = threshold
        print(f"\nOptimal threshold (Balanced): {best_threshold:.3f} (Balance={best_metric:.4f})")
    
    return best_threshold, best_metric


def evaluate_model(
    model: keras.Model,
    X: np.ndarray,
    y: np.ndarray,
    set_name: str = "Test",
    threshold: float = 0.5
) -> dict:
    """
    Evaluate model and return metrics.
    
    Args:
        model: Trained Keras model
        X: Features
        y: True labels
        set_name: Name of the dataset (for printing)
        threshold: Decision threshold (default: 0.5)
        
    Returns:
        Dictionary of metrics
    """
    print(f"\nEvaluating on {set_name} set (threshold={threshold:.3f})...")
    
    # Get predictions
    y_pred_proba = model.predict(X, verbose=0).flatten()
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate metrics
    loss, accuracy, auc = model.evaluate(X, y, verbose=0)
    
    # Calculate additional metrics
    cm = confusion_matrix(y, y_pred)
    logloss = log_loss(y, y_pred_proba)
    
    metrics = {
        'loss': float(loss),
        'accuracy': float(accuracy),
        'auc': float(auc),
        'log_loss': float(logloss),
        'confusion_matrix': cm.tolist(),
        'num_samples': len(y),
        'positive_rate': float(y.mean()),
        'threshold': float(threshold),
        'f1_score': float(f1_score(y, y_pred))
    }
    
    # Print results
    print(f"{set_name} Results:")
    print(f"  Loss: {loss:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  AUC: {auc:.4f}")
    print(f"  F1 Score: {metrics['f1_score']:.4f}")
    print(f"  Log Loss: {logloss:.4f}")
    print(f"  Confusion Matrix:")
    print(f"    {cm}")
    
    return metrics


def save_results(
    train_metrics: dict,
    val_metrics: dict,
    test_metrics: dict,
    history: keras.callbacks.History,
    optimal_threshold: float = 0.5,
    threshold_metric: str = "f1",
    output_path: str = "outputs/results.json"
):
    """Save training and evaluation results."""
    results = {
        'timestamp': datetime.now().isoformat(),
        'optimal_threshold': float(optimal_threshold),
        'threshold_metric': threshold_metric,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'training_history': {
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']],
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']],
        },
        'epochs_trained': len(history.history['loss'])
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    # Load preprocessed data
    print("Loading preprocessed data...")
    X_train = np.load("outputs/X_train.npy")
    y_train = np.load("outputs/y_train.npy")
    X_val = np.load("outputs/X_val.npy")
    y_val = np.load("outputs/y_val.npy")
    X_test = np.load("outputs/X_test.npy")
    y_test = np.load("outputs/y_test.npy")
    
    # Ensure models directory exists
    Path("models").mkdir(parents=True, exist_ok=True)
    
    # Train model
    model, history = train_model(X_train, y_train, X_val, y_val)
    
    # Evaluate on all sets
    train_metrics = evaluate_model(model, X_train, y_train, "Train")
    val_metrics = evaluate_model(model, X_val, y_val, "Validation")
    test_metrics = evaluate_model(model, X_test, y_test, "Test")
    
    # Save results
    save_results(train_metrics, val_metrics, test_metrics, history)
    
    print("\nTraining complete!")
