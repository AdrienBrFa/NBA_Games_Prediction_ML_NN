"""
Unit tests for Model 2 forward pass.

Tests the Model 2 architecture and forward pass functionality.
"""

import sys
from pathlib import Path
import numpy as np
import tensorflow as tf

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.model_2 import create_model_2, compile_model_2, extract_team_embeddings


def test_model_creation():
    """Test Model 2 can be created with correct architecture."""
    print("Test 1: Creating Model 2...")
    
    model = create_model_2(
        num_features=50,
        num_teams=30,
        embedding_dim=16
    )
    
    # Verify inputs
    assert len(model.inputs) == 3, "Model should have 3 inputs"
    assert model.inputs[0].shape[1] == 50, "Numeric input should have correct shape"
    assert model.inputs[1].shape[1] == 1, "Home team input should be scalar"
    assert model.inputs[2].shape[1] == 1, "Away team input should be scalar"
    
    # Verify output
    assert model.outputs[0].shape[1] == 1, "Output should be single probability"
    
    # Verify has embedding layer
    layer_names = [layer.name for layer in model.layers]
    assert 'team_embedding' in layer_names, "Should have team_embedding layer"
    
    print(f"✓ Test 1 passed - Model has {model.count_params():,} parameters")


def test_forward_pass():
    """Test forward pass with synthetic data."""
    print("\nTest 2: Testing forward pass...")
    
    # Create model
    model = create_model_2(
        num_features=50,
        num_teams=30,
        embedding_dim=16
    )
    model = compile_model_2(model)
    
    # Create synthetic inputs
    batch_size = 32
    X_numeric = np.random.randn(batch_size, 50).astype(np.float32)
    home_ids = np.random.randint(1, 31, size=(batch_size, 1))
    away_ids = np.random.randint(1, 31, size=(batch_size, 1))
    
    # Forward pass
    predictions = model.predict([X_numeric, home_ids, away_ids], verbose=0)
    
    # Verify output
    assert predictions.shape == (batch_size, 1), f"Expected shape (32, 1), got {predictions.shape}"
    assert np.all((predictions >= 0) & (predictions <= 1)), "Predictions should be in [0, 1]"
    
    print("✓ Test 2 passed - Forward pass successful")


def test_different_configurations():
    """Test model with different hyperparameters."""
    print("\nTest 3: Testing different configurations...")
    
    configs = [
        {'num_features': 50, 'num_teams': 30, 'embedding_dim': 8},
        {'num_features': 100, 'num_teams': 30, 'embedding_dim': 16},
        {'num_features': 177, 'num_teams': 30, 'embedding_dim': 32},
    ]
    
    for i, config in enumerate(configs, 1):
        model = create_model_2(**config)
        model = compile_model_2(model)
        
        # Test forward pass
        X = np.random.randn(10, config['num_features']).astype(np.float32)
        home = np.random.randint(1, 31, size=(10, 1))
        away = np.random.randint(1, 31, size=(10, 1))
        
        predictions = model.predict([X, home, away], verbose=0)
        assert predictions.shape == (10, 1), f"Config {i} failed"
        
        print(f"  Config {i}: {config['num_features']} features, {config['embedding_dim']}D embeddings - ✓")
    
    print("✓ Test 3 passed")


def test_embedding_extraction():
    """Test extracting team embeddings from trained model."""
    print("\nTest 4: Testing embedding extraction...")
    
    # Create model
    model = create_model_2(
        num_features=50,
        num_teams=30,
        embedding_dim=16
    )
    model = compile_model_2(model)
    
    # Extract embeddings (before training - should be random)
    embeddings = extract_team_embeddings(model)
    
    # Verify shape
    expected_shape = (31, 16)  # 30 teams + 1 UNK
    assert embeddings.shape == expected_shape, f"Expected shape {expected_shape}, got {embeddings.shape}"
    
    # Verify values are reasonable (not all zeros, not NaN)
    assert not np.all(embeddings == 0), "Embeddings should not be all zeros"
    assert not np.any(np.isnan(embeddings)), "Embeddings should not contain NaN"
    
    print(f"✓ Test 4 passed - Extracted embeddings shape: {embeddings.shape}")


def test_gradient_flow():
    """Test that gradients flow through the model."""
    print("\nTest 5: Testing gradient flow...")
    
    # Create model
    model = create_model_2(
        num_features=50,
        num_teams=30,
        embedding_dim=16
    )
    model = compile_model_2(model)
    
    # Create synthetic training data
    X = np.random.randn(64, 50).astype(np.float32)
    home = np.random.randint(1, 31, size=(64, 1))
    away = np.random.randint(1, 31, size=(64, 1))
    y = np.random.randint(0, 2, size=(64, 1)).astype(np.float32)
    
    # Train for 1 epoch
    initial_loss = model.evaluate([X, home, away], y, verbose=0)[0]
    history = model.fit(
        [X, home, away], y,
        epochs=2,
        batch_size=32,
        verbose=0
    )
    final_loss = history.history['loss'][-1]
    
    # Loss should decrease (learning is happening)
    assert final_loss < initial_loss, "Loss should decrease after training"
    
    print(f"✓ Test 5 passed - Loss: {initial_loss:.4f} → {final_loss:.4f}")


def test_embedding_uniqueness():
    """Test that different teams get different embeddings."""
    print("\nTest 6: Testing embedding uniqueness...")
    
    # Create and train model briefly
    model = create_model_2(
        num_features=50,
        num_teams=30,
        embedding_dim=16
    )
    model = compile_model_2(model)
    
    # Extract embeddings
    embeddings = extract_team_embeddings(model)
    
    # Check that not all embeddings are identical
    # (after random initialization, they should be different)
    unique_rows = np.unique(embeddings.numpy(), axis=0)
    assert len(unique_rows) > 1, "Embeddings should be unique (not all identical)"
    
    print("✓ Test 6 passed - Embeddings are unique")


if __name__ == '__main__':
    print("="*60)
    print("Testing Model 2 Forward Pass")
    print("="*60)
    
    test_model_creation()
    test_forward_pass()
    test_different_configurations()
    test_embedding_extraction()
    test_gradient_flow()
    test_embedding_uniqueness()
    
    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60)
