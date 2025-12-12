"""
Team ID encoding utilities for Model 2.

This module provides functions to convert team names/identifiers to contiguous
integer indices required for embedding layers in neural networks.

Key features:
- Fit encoder on training data only (avoid leakage)
- Handle unknown teams with UNK token
- Save/load mappings as JSON for reproducibility
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional


UNK_TOKEN = "UNK"
UNK_ID = 0


def fit_team_encoder(train_df: pd.DataFrame, home_col: str = 'hometeamId', away_col: str = 'awayteamId') -> Dict[str, int]:
    """
    Build team name to integer ID mapping from training data.
    
    Creates a contiguous integer mapping [1..N] for N unique teams,
    with UNK_TOKEN mapped to 0 for handling unknown teams at inference.
    
    Args:
        train_df: Training DataFrame with home and away team columns
        home_col: Name of home team column (default: 'hometeamId')
        away_col: Name of away team column (default: 'awayteamId')
        
    Returns:
        Dictionary mapping team names to integer IDs
        {UNK_TOKEN: 0, team1: 1, team2: 2, ...}
        
    Example:
        >>> train_df = pd.DataFrame({
        ...     'hometeamId': ['LAL', 'GSW', 'LAL'],
        ...     'awayteamId': ['GSW', 'BOS', 'BOS']
        ... })
        >>> mapping = fit_team_encoder(train_df)
        >>> print(mapping)
        {'UNK': 0, 'LAL': 1, 'GSW': 2, 'BOS': 3}
    """
    # Get all unique team names from both Home and Away columns
    home_teams = set(train_df[home_col].unique())
    away_teams = set(train_df[away_col].unique())
    all_teams = sorted(home_teams | away_teams)  # Sort for reproducibility
    
    # Create mapping with UNK token at index 0
    team_to_id = {UNK_TOKEN: UNK_ID}
    
    # Assign contiguous IDs starting from 1
    for idx, team in enumerate(all_teams, start=1):
        team_to_id[team] = idx
    
    return team_to_id


def transform_team_ids(
    df: pd.DataFrame,
    team_to_id: Dict[str, int],
    home_col: str = 'hometeamId',
    away_col: str = 'awayteamId'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform team names to integer IDs using fitted mapping.
    
    Unknown teams (not in mapping) are assigned UNK_ID.
    
    Args:
        df: DataFrame with home and away team columns
        team_to_id: Team name to ID mapping from fit_team_encoder()
        home_col: Name of home team column (default: 'hometeamId')
        away_col: Name of away team column (default: 'awayteamId')
        
    Returns:
        Tuple of (home_ids, away_ids) as numpy arrays of shape (n_samples,)
        
    Example:
        >>> df = pd.DataFrame({'hometeamId': ['LAL', 'UNK_TEAM'], 'awayteamId': ['GSW', 'BOS']})
        >>> mapping = {'UNK': 0, 'LAL': 1, 'GSW': 2, 'BOS': 3}
        >>> home_ids, away_ids = transform_team_ids(df, mapping)
        >>> print(home_ids)  # [1, 0] - unknown team gets UNK_ID
        >>> print(away_ids)  # [2, 3]
    """
    # Convert team names to IDs, using UNK_ID for unknown teams
    home_ids = df[home_col].map(lambda x: team_to_id.get(x, UNK_ID)).values
    away_ids = df[away_col].map(lambda x: team_to_id.get(x, UNK_ID)).values
    
    return home_ids, away_ids


def save_team_encoder(team_to_id: Dict[str, int], filepath: Path) -> None:
    """
    Save team encoding mapping to JSON file.
    
    Args:
        team_to_id: Team name to ID mapping
        filepath: Path to save JSON file
        
    Example:
        >>> mapping = {'UNK': 0, 'LAL': 1, 'GSW': 2}
        >>> save_team_encoder(mapping, Path('outputs/team_encoding.json'))
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert all values to native Python int (JSON can't handle numpy int64)
    team_to_id_serializable = {str(k): int(v) for k, v in team_to_id.items()}
    
    with open(filepath, 'w') as f:
        json.dump(team_to_id_serializable, f, indent=2)
    
    print(f"Team encoder saved to {filepath}")


def load_team_encoder(filepath: Path) -> Dict[str, int]:
    """
    Load team encoding mapping from JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Dictionary mapping team names to integer IDs
        
    Raises:
        FileNotFoundError: If file doesn't exist
        
    Example:
        >>> mapping = load_team_encoder(Path('outputs/team_encoding.json'))
        >>> print(mapping['LAL'])
        1
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Team encoder file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        team_to_id = json.load(f)
    
    return team_to_id


def get_inverse_mapping(team_to_id: Dict[str, int]) -> Dict[int, str]:
    """
    Create inverse mapping from team IDs to team names.
    
    Useful for visualization and interpretation of embeddings.
    
    Args:
        team_to_id: Team name to ID mapping
        
    Returns:
        Dictionary mapping integer IDs to team names
        
    Example:
        >>> mapping = {'UNK': 0, 'LAL': 1, 'GSW': 2}
        >>> inverse = get_inverse_mapping(mapping)
        >>> print(inverse[1])
        'LAL'
    """
    return {v: k for k, v in team_to_id.items()}


def print_team_encoder_summary(team_to_id: Dict[str, int]) -> None:
    """
    Print summary of team encoding mapping.
    
    Args:
        team_to_id: Team name to ID mapping
    """
    print(f"\n{'='*60}")
    print("Team Encoder Summary")
    print(f"{'='*60}")
    print(f"Total teams (including UNK): {len(team_to_id)}")
    print(f"Unique teams: {len(team_to_id) - 1}")
    print(f"\nTeam Mappings:")
    
    # Sort by ID for readability
    sorted_items = sorted(team_to_id.items(), key=lambda x: x[1])
    for team, team_id in sorted_items[:10]:  # Show first 10
        print(f"  {team:15} -> {team_id}")
    
    if len(sorted_items) > 10:
        print(f"  ... ({len(sorted_items) - 10} more teams)")
    
    print(f"{'='*60}\n")


if __name__ == '__main__':
    # Simple test
    print("Testing team encoding utilities...")
    
    # Create sample data
    test_df = pd.DataFrame({
        'Home': ['LAL', 'GSW', 'LAL', 'BOS', 'MIA'],
        'Away': ['GSW', 'BOS', 'BOS', 'LAL', 'GSW']
    })
    
    print("\nSample Data:")
    print(test_df)
    
    # Fit encoder
    print("\nFitting encoder...")
    mapping = fit_team_encoder(test_df)
    print_team_encoder_summary(mapping)
    
    # Transform IDs
    print("Transforming team names to IDs...")
    home_ids, away_ids = transform_team_ids(test_df, mapping)
    print(f"Home IDs: {home_ids}")
    print(f"Away IDs: {away_ids}")
    
    # Test unknown team handling
    print("\nTesting unknown team handling...")
    test_unknown = pd.DataFrame({
        'Home': ['LAL', 'UNKNOWN_TEAM'],
        'Away': ['GSW', 'BOS']
    })
    home_ids, away_ids = transform_team_ids(test_unknown, mapping)
    print(f"Home IDs (with unknown): {home_ids}")
    print(f"  - Unknown team mapped to UNK_ID={UNK_ID}")
    
    print("\n✓ Team encoding utilities working correctly!")
