"""
Unit tests for embedding extraction functionality.

Tests the extraction and validation of team embeddings from Model 2.
"""

import sys
from pathlib import Path
import numpy as np
import tensorflow as tf

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.model_2 import create_model_2, compile_model_2, extract_team_embeddings


def test_embedding_extraction_shape():
    """Test that extracted embeddings have correct shape."""
    print("Test 1: Verifying embedding shape...")
    
    configs = [
        (30, 8),
        (30, 16),
        (30, 32),
    ]
    
    for num_teams, emb_dim in configs:
        model = create_model_2(
            num_features=50,
            num_teams=num_teams,
            embedding_dim=emb_dim
        )
        
        embeddings = extract_team_embeddings(model)
        expected_shape = (num_teams + 1, emb_dim)  # +1 for UNK token
        
        assert embeddings.shape == expected_shape, \
            f"Expected {expected_shape}, got {embeddings.shape}"
        
        print(f"  {num_teams} teams, {emb_dim}D: {embeddings.shape} ✓")
    
    print("✓ Test 1 passed")


def test_embedding_values():
    """Test that embedding values are reasonable."""
    print("\nTest 2: Verifying embedding values...")
    
    model = create_model_2(
        num_features=50,
        num_teams=30,
        embedding_dim=16
    )
    model = compile_model_2(model)
    
    embeddings = extract_team_embeddings(model)
    
    # Check no NaN or Inf
    assert not np.any(np.isnan(embeddings.numpy())), "Embeddings contain NaN"
    assert not np.any(np.isinf(embeddings.numpy())), "Embeddings contain Inf"
    
    # Check not all zeros
    assert not np.all(embeddings.numpy() == 0), "Embeddings are all zeros"
    
    # Check reasonable magnitude (typical init is ~N(0, 0.05))
    abs_mean = np.abs(embeddings.numpy()).mean()
    assert abs_mean < 1.0, f"Embeddings have unusually large values (mean={abs_mean:.3f})"
    
    print(f"  Mean absolute value: {abs_mean:.4f} ✓")
    print("✓ Test 2 passed")


def test_embedding_changes_after_training():
    """Test that embeddings change after training."""
    print("\nTest 3: Verifying embeddings change after training...")
    
    # Create model
    model = create_model_2(
        num_features=50,
        num_teams=30,
        embedding_dim=16
    )
    model = compile_model_2(model)
    
    # Get initial embeddings
    initial_embeddings = extract_team_embeddings(model).numpy().copy()
    
    # Train briefly
    X = np.random.randn(128, 50).astype(np.float32)
    home = np.random.randint(1, 31, size=(128, 1))
    away = np.random.randint(1, 31, size=(128, 1))
    y = np.random.randint(0, 2, size=(128, 1)).astype(np.float32)
    
    model.fit([X, home, away], y, epochs=5, batch_size=32, verbose=0)
    
    # Get trained embeddings
    trained_embeddings = extract_team_embeddings(model).numpy()
    
    # Verify they changed
    diff = np.abs(trained_embeddings - initial_embeddings).mean()
    assert diff > 0.001, f"Embeddings barely changed (mean diff={diff:.6f})"
    
    print(f"  Mean change: {diff:.4f} ✓")
    print("✓ Test 3 passed")


def test_embedding_layer_access():
    """Test direct access to embedding layer."""
    print("\nTest 4: Testing embedding layer access...")
    
    model = create_model_2(
        num_features=50,
        num_teams=30,
        embedding_dim=16
    )
    
    # Get embedding layer
    embedding_layer = model.get_layer('team_embedding')
    
    # Verify properties
    config = embedding_layer.get_config()
    assert config['input_dim'] == 31, "Input dim should be num_teams + 1"
    assert config['output_dim'] == 16, "Output dim should match embedding_dim"
    
    # Get weights directly
    weights = embedding_layer.get_weights()[0]
    assert weights.shape == (31, 16), f"Weight shape should be (31, 16), got {weights.shape}"
    
    print("✓ Test 4 passed")


def test_embedding_consistency():
    """Test that same input produces same embedding."""
    print("\nTest 5: Testing embedding consistency...")
    
    model = create_model_2(
        num_features=50,
        num_teams=30,
        embedding_dim=16
    )
    model = compile_model_2(model)
    
    # Same input twice
    X = np.random.randn(10, 50).astype(np.float32)
    home = np.array([[5], [5], [10], [10], [15], [15], [20], [20], [25], [25]])
    away = np.array([[10], [10], [5], [5], [20], [20], [15], [15], [30], [30]])
    
    pred1 = model.predict([X, home, away], verbose=0)
    pred2 = model.predict([X, home, away], verbose=0)
    
    # Predictions should be identical
    np.testing.assert_array_almost_equal(pred1, pred2, decimal=6)
    
    print("✓ Test 5 passed - Predictions are deterministic")


def test_unk_token_embedding():
    """Test that UNK token (index 0) has an embedding."""
    print("\nTest 6: Testing UNK token embedding...")
    
    model = create_model_2(
        num_features=50,
        num_teams=30,
        embedding_dim=16
    )
    
    embeddings = extract_team_embeddings(model).numpy()
    
    # Get UNK embedding (first row)
    unk_embedding = embeddings[0]
    
    # Verify it exists and is not all zeros
    assert len(unk_embedding) == 16, "UNK embedding should have correct dimension"
    assert not np.all(unk_embedding == 0), "UNK embedding should not be all zeros"
    
    print("✓ Test 6 passed - UNK token has valid embedding")


if __name__ == '__main__':
    print("="*60)
    print("Testing Embedding Extraction")
    print("="*60)
    
    test_embedding_extraction_shape()
    test_embedding_values()
    test_embedding_changes_after_training()
    test_embedding_layer_access()
    test_embedding_consistency()
    test_unk_token_embedding()
    
    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60)
