"""
Data loading and basic filtering for NBA game prediction (Stage A1).

This script:
1. Loads Games.csv
2. Parses dates and computes seasonYear
3. Filters for Regular Season games from 1990 onwards
4. Creates the target variable (home team win indicator)
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_and_filter_games(data_path: str = "data/Games.csv") -> pd.DataFrame:
    """
    Load and perform basic filtering on Games.csv.
    
    Args:
        data_path: Path to Games.csv file
        
    Returns:
        Filtered DataFrame with seasonYear and target variable
    """
    print("Loading Games.csv...")
    df = pd.read_csv(data_path)
    print(f"Initial rows: {len(df)}")
    
    # Parse date
    df['gameDateTimeEst'] = pd.to_datetime(df['gameDateTimeEst'])
    
    # Drop rows with missing scores
    df = df.dropna(subset=['homeScore', 'awayScore'])
    print(f"After dropping missing scores: {len(df)}")
    
    # Filter for Regular Season only
    df = df[df['gameType'] == 'Regular Season']
    print(f"After filtering Regular Season: {len(df)}")
    
    # Compute seasonYear
    df['year'] = df['gameDateTimeEst'].dt.year
    df['month'] = df['gameDateTimeEst'].dt.month
    df['day'] = df['gameDateTimeEst'].dt.day
    
    # seasonYear logic: if month >= 8, seasonYear = year, else seasonYear = year - 1
    df['seasonYear'] = df.apply(
        lambda row: row['year'] if row['month'] >= 8 else row['year'] - 1,
        axis=1
    )
    
    # Filter for seasonYear >= 1990
    df = df[df['seasonYear'] >= 1990]
    print(f"After filtering seasonYear >= 1990: {len(df)}")
    
    # Create target variable: y = 1 if home team wins, 0 otherwise
    df['y'] = (df['homeScore'] > df['awayScore']).astype(int)
    
    # Sort by date for chronological processing
    df = df.sort_values('gameDateTimeEst').reset_index(drop=True)
    
    print(f"Final dataset size: {len(df)} games")
    print(f"Season range: {df['seasonYear'].min()} to {df['seasonYear'].max()}")
    print(f"Home team win rate: {df['y'].mean():.3f}")
    
    return df


if __name__ == "__main__":
    df = load_and_filter_games()
    output_path = Path("outputs/filtered_games.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nFiltered data saved to {output_path}")
