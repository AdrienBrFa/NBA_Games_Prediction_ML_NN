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
    
    # Initialize feature columns
    feature_cols = [
        'home_season_games_played', 'home_season_wins', 'home_season_losses', 'home_season_win_pct',
        'home_last5_games_played', 'home_last5_win_pct', 'home_last5_avg_point_diff',
        'home_days_since_last_game',
        'away_season_games_played', 'away_season_wins', 'away_season_losses', 'away_season_win_pct',
        'away_last5_games_played', 'away_last5_win_pct', 'away_last5_avg_point_diff',
        'away_days_since_last_game'
    ]
    
    for col in feature_cols:
        df[col] = np.nan
    
    # Process each team and season separately
    teams = pd.concat([df['hometeamId'], df['awayteamId']]).unique()
    
    for team_id in teams:
        for season in df['seasonYear'].unique():
            # Get all games for this team in this season
            team_mask_home = (df['hometeamId'] == team_id) & (df['seasonYear'] == season)
            team_mask_away = (df['awayteamId'] == team_id) & (df['seasonYear'] == season)
            team_games_indices = df[team_mask_home | team_mask_away].index.tolist()
            
            # Sort by date
            team_games_indices.sort(key=lambda idx: df.loc[idx, 'gameDateTimeEst'])
            
            # Process each game chronologically
            for i, game_idx in enumerate(team_games_indices):
                # Get previous games (games before current game in chronological order)
                prev_indices = team_games_indices[:i]
                
                if len(prev_indices) == 0:
                    # No previous games - use default values
                    games_played = 0
                    wins = 0
                    losses = 0
                    win_pct = 0.5
                    last5_games_played = 0
                    last5_win_pct = 0.5
                    last5_avg_point_diff = 0.0
                    days_since_last = np.nan
                else:
                    # Compute season statistics from all previous games
                    games_played = len(prev_indices)
                    wins = 0
                    point_diffs = []
                    
                    for prev_idx in prev_indices:
                        prev_game = df.loc[prev_idx]
                        # Determine if team won and compute point differential
                        if prev_game['hometeamId'] == team_id:
                            team_score = prev_game['homeScore']
                            opp_score = prev_game['awayScore']
                        else:
                            team_score = prev_game['awayScore']
                            opp_score = prev_game['homeScore']
                        
                        if team_score > opp_score:
                            wins += 1
                        point_diffs.append(team_score - opp_score)
                    
                    losses = games_played - wins
                    win_pct = wins / games_played
                    
                    # Compute last 5 statistics
                    last5_indices = prev_indices[-5:]
                    last5_games_played = len(last5_indices)
                    last5_wins = 0
                    last5_point_diffs = []
                    
                    for prev_idx in last5_indices:
                        prev_game = df.loc[prev_idx]
                        if prev_game['hometeamId'] == team_id:
                            team_score = prev_game['homeScore']
                            opp_score = prev_game['awayScore']
                        else:
                            team_score = prev_game['awayScore']
                            opp_score = prev_game['homeScore']
                        
                        if team_score > opp_score:
                            last5_wins += 1
                        last5_point_diffs.append(team_score - opp_score)
                    
                    last5_win_pct = last5_wins / last5_games_played if last5_games_played > 0 else 0.5
                    last5_avg_point_diff = np.mean(last5_point_diffs) if last5_point_diffs else 0.0
                    
                    # Days since last game
                    last_game_idx = prev_indices[-1]
                    current_date = df.loc[game_idx, 'gameDateTimeEst']
                    last_date = df.loc[last_game_idx, 'gameDateTimeEst']
                    days_since_last = (current_date - last_date).days
                
                # Assign features based on whether team is home or away
                current_game = df.loc[game_idx]
                if current_game['hometeamId'] == team_id:
                    df.loc[game_idx, 'home_season_games_played'] = games_played
                    df.loc[game_idx, 'home_season_wins'] = wins
                    df.loc[game_idx, 'home_season_losses'] = losses
                    df.loc[game_idx, 'home_season_win_pct'] = win_pct
                    df.loc[game_idx, 'home_last5_games_played'] = last5_games_played
                    df.loc[game_idx, 'home_last5_win_pct'] = last5_win_pct
                    df.loc[game_idx, 'home_last5_avg_point_diff'] = last5_avg_point_diff
                    df.loc[game_idx, 'home_days_since_last_game'] = days_since_last
                else:
                    df.loc[game_idx, 'away_season_games_played'] = games_played
                    df.loc[game_idx, 'away_season_wins'] = wins
                    df.loc[game_idx, 'away_season_losses'] = losses
                    df.loc[game_idx, 'away_season_win_pct'] = win_pct
                    df.loc[game_idx, 'away_last5_games_played'] = last5_games_played
                    df.loc[game_idx, 'away_last5_win_pct'] = last5_win_pct
                    df.loc[game_idx, 'away_last5_avg_point_diff'] = last5_avg_point_diff
                    df.loc[game_idx, 'away_days_since_last_game'] = days_since_last
    
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
    df = pd.read_csv(input_path)
    df['gameDateTimeEst'] = pd.to_datetime(df['gameDateTimeEst'])
    
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
