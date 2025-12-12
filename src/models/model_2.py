"""
Model 2: Enhanced MLP with Team Embeddings for NBA game prediction.

This module implements an improved neural network architecture that learns
team representations through embeddings, combined with an enhanced MLP.

Architecture:
- Shared team embedding layer (default: 16-dimensional)
- Embedding combination: concat(home_emb, away_emb, home_emb - away_emb)
- Enhanced MLP with BatchNorm, Dropout, and L2 regularization
- Output: sigmoid probability P(home_win)

Compatible with:
- Model 2A: Stage A1 data + team embeddings
- Model 2B-intermediate: Stage B1 intermediate features + team embeddings
- Model 2B-full: Stage B1 full features + team embeddings
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from typing import Tuple


def create_model_2(
    num_features: int,
    num_teams: int,
    embedding_dim: int = 16,
    hidden_units: Tuple[int, ...] = (128, 64, 32),
    dropout_rate: float = 0.3,
    l2_reg: float = 1e-3,
) -> keras.Model:
    """
    Create Model 2: Enhanced MLP with team embeddings.
    
    The model accepts three inputs:
    1. Numeric features (from Stage A1 or B1)
    2. Home team ID (integer index)
    3. Away team ID (integer index)
    
    Team embeddings are learned during training and capture team characteristics
    that may not be fully represented in aggregated statistics.
    
    Args:
        num_features: Number of numeric input features
        num_teams: Number of unique teams (for embedding layer)
        embedding_dim: Dimension of team embedding vectors (default: 16)
        hidden_units: Tuple of hidden layer sizes (default: (128, 64, 32))
        dropout_rate: Dropout rate for regularization (default: 0.3)
        l2_reg: L2 regularization strength (default: 1e-3)
        
    Returns:
        Compiled Keras Model with functional API
        
    Example:
        >>> model = create_model_2(num_features=50, num_teams=30)
        >>> # Inputs: [X_numeric, home_ids, away_ids]
        >>> # Output: probabilities of shape (batch_size, 1)
    """
    # Define inputs
    numeric_input = layers.Input(shape=(num_features,), name='numeric_features')
    home_team_input = layers.Input(shape=(1,), dtype='int32', name='home_team_id')
    away_team_input = layers.Input(shape=(1,), dtype='int32', name='away_team_id')
    
    # Shared team embedding layer
    # Add 1 to num_teams to handle potential UNK token at index 0
    team_embedding = layers.Embedding(
        input_dim=num_teams + 1,
        output_dim=embedding_dim,
        embeddings_regularizer=regularizers.l2(l2_reg),
        name='team_embedding'
    )
    
    # Lookup embeddings for home and away teams
    home_embedding = team_embedding(home_team_input)  # Shape: (batch, 1, embedding_dim)
    away_embedding = team_embedding(away_team_input)  # Shape: (batch, 1, embedding_dim)
    
    # Flatten embeddings
    home_embedding = layers.Flatten()(home_embedding)  # Shape: (batch, embedding_dim)
    away_embedding = layers.Flatten()(away_embedding)  # Shape: (batch, embedding_dim)
    
    # Combine embeddings: [home_emb, away_emb, home_emb - away_emb]
    # This captures both team identities and their relative strength
    embedding_diff = layers.Subtract(name='embedding_diff')([home_embedding, away_embedding])
    combined_embeddings = layers.Concatenate(name='combined_embeddings')([
        home_embedding,
        away_embedding,
        embedding_diff
    ])
    
    # Concatenate combined embeddings with numeric features
    full_features = layers.Concatenate(name='full_features')([
        numeric_input,
        combined_embeddings
    ])
    
    # Enhanced MLP with BatchNorm and Dropout
    x = full_features
    for i, units in enumerate(hidden_units, start=1):
        x = layers.Dense(
            units,
            kernel_regularizer=regularizers.l2(l2_reg),
            name=f'dense_{i}'
        )(x)
        x = layers.BatchNormalization(name=f'batch_norm_{i}')(x)
        x = layers.Activation('relu', name=f'relu_{i}')(x)
        x = layers.Dropout(dropout_rate, name=f'dropout_{i}')(x)
    
    # Output layer
    output = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    # Create model
    model = keras.Model(
        inputs=[numeric_input, home_team_input, away_team_input],
        outputs=output,
        name='model_2_team_embeddings'
    )
    
    return model


def compile_model_2(model: keras.Model, learning_rate: float = 0.001) -> keras.Model:
    """
    Compile Model 2 with appropriate loss, optimizer, and metrics.
    
    Args:
        model: Keras model to compile
        learning_rate: Learning rate for Adam optimizer (default: 0.001)
        
    Returns:
        Compiled model
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.AUC(name='auc'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
    )
    return model


def extract_team_embeddings(model: keras.Model) -> tf.Tensor:
    """
    Extract learned team embedding matrix from trained Model 2.
    
    This function retrieves the weight matrix from the team embedding layer,
    which can be used for visualization and interpretation.
    
    Args:
        model: Trained Model 2 instance
        
    Returns:
        Tensor of shape (num_teams+1, embedding_dim) containing team embeddings
        
    Raises:
        ValueError: If model doesn't have a 'team_embedding' layer
        
    Example:
        >>> embeddings = extract_team_embeddings(trained_model)
        >>> print(embeddings.shape)  # (31, 16) for 30 teams + UNK
    """
    try:
        embedding_layer = model.get_layer('team_embedding')
        embeddings = embedding_layer.get_weights()[0]
        return tf.convert_to_tensor(embeddings)
    except ValueError:
        raise ValueError(
            "Model doesn't have a 'team_embedding' layer. "
            "Ensure this is a Model 2 instance."
        )


if __name__ == '__main__':
    # Simple test to verify model creation
    print("Testing Model 2 creation...")
    
    test_model = create_model_2(
        num_features=50,
        num_teams=30,
        embedding_dim=16
    )
    
    test_model = compile_model_2(test_model)
    
    print("\nModel Summary:")
    test_model.summary()
    
    print("\n✓ Model 2 created successfully!")
    print(f"  - Total parameters: {test_model.count_params():,}")
    print(f"  - Embedding layer shape: {test_model.get_layer('team_embedding').output_shape}")
