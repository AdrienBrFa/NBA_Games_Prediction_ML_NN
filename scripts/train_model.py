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
from tensorflow.keras import layers, callbacks
from sklearn.metrics import confusion_matrix, log_loss


def create_mlp_model(input_dim: int) -> keras.Model:
    """
    Create a simple MLP model for binary classification.
    
    Architecture (as specified in README):
    - Input layer: size = input_dim
    - Hidden layer 1: Dense(64, activation="relu")
    - Hidden layer 2: Dense(32, activation="relu")
    - Output layer: Dense(1, activation="sigmoid")
    
    Args:
        input_dim: Number of input features
        
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu', name='hidden1'),
        layers.Dense(32, activation='relu', name='hidden2'),
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
    patience: int = 10,
    model_save_path: str = "models/stage_a1_mlp.keras"
) -> tuple:
    """
    Train the MLP model with early stopping.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        epochs: Maximum number of epochs
        batch_size: Batch size for training
        patience: Early stopping patience
        model_save_path: Path to save the best model
        
    Returns:
        Tuple of (trained_model, history)
    """
    print("Creating model...")
    input_dim = X_train.shape[1]
    model = create_mlp_model(input_dim)
    
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
    
    # Train model
    print(f"\nTraining model...")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, model_checkpoint],
        verbose=1
    )
    
    return model, history


def evaluate_model(
    model: keras.Model,
    X: np.ndarray,
    y: np.ndarray,
    set_name: str = "Test"
) -> dict:
    """
    Evaluate model and return metrics.
    
    Args:
        model: Trained Keras model
        X: Features
        y: True labels
        set_name: Name of the dataset (for printing)
        
    Returns:
        Dictionary of metrics
    """
    print(f"\nEvaluating on {set_name} set...")
    
    # Get predictions
    y_pred_proba = model.predict(X, verbose=0).flatten()
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
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
        'positive_rate': float(y.mean())
    }
    
    # Print results
    print(f"{set_name} Results:")
    print(f"  Loss: {loss:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  AUC: {auc:.4f}")
    print(f"  Log Loss: {logloss:.4f}")
    print(f"  Confusion Matrix:")
    print(f"    {cm}")
    
    return metrics


def save_results(
    train_metrics: dict,
    val_metrics: dict,
    test_metrics: dict,
    history: keras.callbacks.History,
    output_path: str = "outputs/results.json"
):
    """Save training and evaluation results."""
    results = {
        'timestamp': datetime.now().isoformat(),
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
