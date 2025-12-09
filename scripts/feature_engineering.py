"""
Feature engineering for NBA game prediction (Stage A1).

This script computes all features defined in the README:
- Team season statistics (games played, wins, losses, win %)
- Team last 5 games statistics
- Days since last game
- Delta features (home - away)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from utils import parse_datetime_column


def compute_team_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute historical features for each team at each game.
    
    This function processes games chronologically for each team and season,
    computing features based only on previous games (no data leakage).
    
    Args:
        df: DataFrame with filtered games
        
    Returns:
        DataFrame with added feature columns
    """
    print("Computing team historical features...")
    print(f"Processing {len(df)} games...")
    
    # Create a long format dataset where each row is a team-game combination
    home_games = df[['gameDateTimeEst', 'seasonYear', 'hometeamId', 'homeScore', 'awayScore']].copy()
    home_games.columns = ['date', 'season', 'team', 'team_score', 'opp_score']
    home_games['is_home'] = 1
    
    away_games = df[['gameDateTimeEst', 'seasonYear', 'awayteamId', 'awayScore', 'homeScore']].copy()
    away_games.columns = ['date', 'season', 'team', 'team_score', 'opp_score']
    away_games['is_home'] = 0
    
    # Combine and sort
    all_team_games = pd.concat([home_games, away_games], ignore_index=True)
    all_team_games = all_team_games.sort_values(['team', 'season', 'date']).reset_index(drop=True)
    
    # Compute team-level stats
    all_team_games['won'] = (all_team_games['team_score'] > all_team_games['opp_score']).astype(int)
    all_team_games['point_diff'] = all_team_games['team_score'] - all_team_games['opp_score']
    
    # Compute cumulative stats (shifted to avoid leakage)
    all_team_games['games_played'] = all_team_games.groupby(['team', 'season']).cumcount()
    all_team_games['season_wins'] = all_team_games.groupby(['team', 'season'])['won'].cumsum() - all_team_games['won']
    all_team_games['season_games'] = all_team_games['games_played']
    all_team_games['season_losses'] = all_team_games['season_games'] - all_team_games['season_wins']
    
    # Win percentage with default
    all_team_games['season_win_pct'] = all_team_games['season_wins'] / all_team_games['season_games']
    all_team_games.loc[all_team_games['season_games'] == 0, 'season_win_pct'] = 0.5
    
    # Last 5 games statistics using rolling windows
    all_team_games['last5_wins'] = (
        all_team_games.groupby(['team', 'season'])['won']
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum())
    )
    all_team_games['last5_games'] = (
        all_team_games.groupby(['team', 'season'])['won']
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).count())
    )
    all_team_games['last5_point_diff'] = (
        all_team_games.groupby(['team', 'season'])['point_diff']
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )
    
    # Fill NaN for first game of season
    all_team_games['last5_games'] = all_team_games['last5_games'].fillna(0)
    all_team_games['last5_wins'] = all_team_games['last5_wins'].fillna(0)
    all_team_games['last5_point_diff'] = all_team_games['last5_point_diff'].fillna(0.0)
    
    # Win percentage for last 5
    all_team_games['last5_win_pct'] = all_team_games['last5_wins'] / all_team_games['last5_games']
    all_team_games.loc[all_team_games['last5_games'] == 0, 'last5_win_pct'] = 0.5
    
    # Days since last game
    all_team_games['prev_date'] = all_team_games.groupby(['team', 'season'])['date'].shift(1)
    all_team_games['days_since_last'] = (all_team_games['date'] - all_team_games['prev_date']).dt.days
    
    # Split back to home and away
    home_stats = all_team_games[all_team_games['is_home'] == 1].copy()
    away_stats = all_team_games[all_team_games['is_home'] == 0].copy()
    
    # Merge back to original dataframe
    df = df.sort_values('gameDateTimeEst').reset_index(drop=True)
    home_stats = home_stats.sort_values('date').reset_index(drop=True)
    away_stats = away_stats.sort_values('date').reset_index(drop=True)
    
    # Add home features
    df['home_season_games_played'] = home_stats['season_games'].values
    df['home_season_wins'] = home_stats['season_wins'].values
    df['home_season_losses'] = home_stats['season_losses'].values
    df['home_season_win_pct'] = home_stats['season_win_pct'].values
    df['home_last5_games_played'] = home_stats['last5_games'].values
    df['home_last5_win_pct'] = home_stats['last5_win_pct'].values
    df['home_last5_avg_point_diff'] = home_stats['last5_point_diff'].values
    df['home_days_since_last_game'] = home_stats['days_since_last'].values
    
    # Add away features
    df['away_season_games_played'] = away_stats['season_games'].values
    df['away_season_wins'] = away_stats['season_wins'].values
    df['away_season_losses'] = away_stats['season_losses'].values
    df['away_season_win_pct'] = away_stats['season_win_pct'].values
    df['away_last5_games_played'] = away_stats['last5_games'].values
    df['away_last5_win_pct'] = away_stats['last5_win_pct'].values
    df['away_last5_avg_point_diff'] = away_stats['last5_point_diff'].values
    df['away_days_since_last_game'] = away_stats['days_since_last'].values
    
    print("Team features computed successfully")
    return df


def add_delta_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add delta features (home - away differences).
    
    Args:
        df: DataFrame with home and away features
        
    Returns:
        DataFrame with delta features added
    """
    print("Computing delta features...")
    
    df['delta_season_win_pct'] = df['home_season_win_pct'] - df['away_season_win_pct']
    df['delta_last5_win_pct'] = df['home_last5_win_pct'] - df['away_last5_win_pct']
    df['delta_last5_point_diff'] = df['home_last5_avg_point_diff'] - df['away_last5_avg_point_diff']
    df['delta_days_rest'] = df['home_days_since_last_game'] - df['away_days_since_last_game']
    
    print("Delta features computed successfully")
    return df


def engineer_features(input_path: str = "outputs/filtered_games.csv") -> pd.DataFrame:
    """
    Main function to engineer all features.
    
    Args:
        input_path: Path to filtered games CSV
        
    Returns:
        DataFrame with all engineered features
    """
    df = pd.read_csv(input_path, low_memory=False)
    df = parse_datetime_column(df, 'gameDateTimeEst')
    
    # Compute team features
    df = compute_team_features(df)
    
    # Add delta features
    df = add_delta_features(df)
    
    # Print feature summary
    print("\nFeature engineering complete!")
    print(f"Total games: {len(df)}")
    print(f"Missing values in key features:")
    key_features = ['home_season_win_pct', 'away_season_win_pct', 
                   'home_last5_win_pct', 'away_last5_win_pct',
                   'home_days_since_last_game', 'away_days_since_last_game']
    for feat in key_features:
        missing = df[feat].isna().sum()
        print(f"  {feat}: {missing} ({missing/len(df)*100:.2f}%)")
    
    return df


if __name__ == "__main__":
    df = engineer_features()
    output_path = Path("outputs/features_engineered.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nFeatures saved to {output_path}")
