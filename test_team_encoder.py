"""
Unit tests for team encoding utilities.

Tests the team ID encoding functionality used in Model 2.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.team_encoding import (
    fit_team_encoder,
    transform_team_ids,
    get_inverse_mapping,
    UNK_ID,
    UNK_TOKEN
)


def test_fit_team_encoder():
    """Test fitting team encoder on training data."""
    print("Test 1: Fitting team encoder...")
    
    # Create sample data
    df = pd.DataFrame({
        'Home': ['LAL', 'GSW', 'LAL', 'BOS'],
        'Away': ['GSW', 'BOS', 'BOS', 'LAL']
    })
    
    # Fit encoder
    mapping = fit_team_encoder(df)
    
    # Verify
    assert UNK_TOKEN in mapping, "UNK token should be in mapping"
    assert mapping[UNK_TOKEN] == UNK_ID, "UNK token should be mapped to UNK_ID"
    assert len(mapping) == 4, f"Expected 4 mappings (UNK + 3 teams), got {len(mapping)}"
    assert all(team in mapping for team in ['LAL', 'GSW', 'BOS']), "All teams should be mapped"
    
    # Check IDs are contiguous
    ids = sorted([v for k, v in mapping.items() if k != UNK_TOKEN])
    assert ids == list(range(1, len(mapping))), "IDs should be contiguous starting from 1"
    
    print("✓ Test 1 passed")


def test_transform_team_ids():
    """Test transforming team names to IDs."""
    print("\nTest 2: Transforming team names to IDs...")
    
    # Create mapping
    mapping = {UNK_TOKEN: 0, 'LAL': 1, 'GSW': 2, 'BOS': 3}
    
    # Create test data
    df = pd.DataFrame({
        'Home': ['LAL', 'GSW', 'UNKNOWN'],
        'Away': ['GSW', 'BOS', 'LAL']
    })
    
    # Transform
    home_ids, away_ids = transform_team_ids(df, mapping)
    
    # Verify
    assert isinstance(home_ids, np.ndarray), "Should return numpy array"
    assert isinstance(away_ids, np.ndarray), "Should return numpy array"
    assert len(home_ids) == 3, "Should have 3 home IDs"
    assert len(away_ids) == 3, "Should have 3 away IDs"
    
    # Check specific values
    assert home_ids[0] == 1, "LAL should map to 1"
    assert home_ids[1] == 2, "GSW should map to 2"
    assert home_ids[2] == 0, "Unknown team should map to UNK_ID (0)"
    assert away_ids[2] == 1, "LAL should map to 1"
    
    print("✓ Test 2 passed")


def test_inverse_mapping():
    """Test inverse mapping creation."""
    print("\nTest 3: Creating inverse mapping...")
    
    # Create mapping
    mapping = {UNK_TOKEN: 0, 'LAL': 1, 'GSW': 2, 'BOS': 3}
    
    # Get inverse
    inverse = get_inverse_mapping(mapping)
    
    # Verify
    assert len(inverse) == len(mapping), "Inverse should have same size"
    assert inverse[0] == UNK_TOKEN, "ID 0 should map to UNK"
    assert inverse[1] == 'LAL', "ID 1 should map to LAL"
    assert inverse[2] == 'GSW', "ID 2 should map to GSW"
    assert inverse[3] == 'BOS', "ID 3 should map to BOS"
    
    print("✓ Test 3 passed")


def test_reproducibility():
    """Test that encoding is reproducible."""
    print("\nTest 4: Testing reproducibility...")
    
    # Same data, fit twice
    df = pd.DataFrame({
        'Home': ['LAL', 'GSW', 'LAL', 'BOS'],
        'Away': ['GSW', 'BOS', 'BOS', 'LAL']
    })
    
    mapping1 = fit_team_encoder(df)
    mapping2 = fit_team_encoder(df)
    
    # Should be identical (teams are sorted)
    assert mapping1 == mapping2, "Encoding should be reproducible"
    
    print("✓ Test 4 passed")


def test_edge_cases():
    """Test edge cases."""
    print("\nTest 5: Testing edge cases...")
    
    # Single team
    df_single = pd.DataFrame({
        'Home': ['LAL'],
        'Away': ['LAL']
    })
    mapping_single = fit_team_encoder(df_single)
    assert len(mapping_single) == 2, "Should have UNK + 1 team"
    
    # All unknown teams
    mapping = {UNK_TOKEN: 0, 'LAL': 1}
    df_unknown = pd.DataFrame({
        'Home': ['XXX', 'YYY'],
        'Away': ['ZZZ', 'WWW']
    })
    home_ids, away_ids = transform_team_ids(df_unknown, mapping)
    assert all(home_ids == UNK_ID), "All unknown home teams should map to UNK_ID"
    assert all(away_ids == UNK_ID), "All unknown away teams should map to UNK_ID"
    
    print("✓ Test 5 passed")


if __name__ == '__main__':
    print("="*60)
    print("Testing Team Encoding Utilities")
    print("="*60)
    
    test_fit_team_encoder()
    test_transform_team_ids()
    test_inverse_mapping()
    test_reproducibility()
    test_edge_cases()
    
    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60)
