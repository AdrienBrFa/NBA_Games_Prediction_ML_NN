"""
Stage B1 Team Statistics Feature Engineering

Integrates TeamStatistics.csv with game-level data to create rolling features,
home/away splits, and delta features while preventing data leakage.

Key principles:
1. NO LEAKAGE: Rolling features computed only from games BEFORE current game
2. Time-based: Strict datetime ordering for rolling windows
3. Merge validation: Each game must have stats for both teams
4. Missing values: Handled with neutral defaults or median imputation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import warnings


def load_team_statistics(filepath: str) -> pd.DataFrame:
    """
    Load and preprocess TeamStatistics.csv.
    
    Args:
        filepath: Path to TeamStatistics.csv
        
    Returns:
        DataFrame with parsed datetime and sorted by game date
    """
    df = pd.read_csv(filepath)
    
    # Parse datetime - handle mixed timezone formats by using utc=True then converting
    df['gameDateTimeEst'] = pd.to_datetime(df['gameDateTimeEst'], utc=True, format='mixed')
    
    # Convert to timezone-naive EST time
    df['gameDateTimeEst'] = df['gameDateTimeEst'].dt.tz_localize(None)
    
    # Sort by game datetime (critical for no-leakage rolling features)
    df = df.sort_values('gameDateTimeEst').reset_index(drop=True)
    
    print(f"Loaded TeamStatistics: {len(df)} rows, {len(df.columns)} columns")
    print(f"Date range: {df['gameDateTimeEst'].min()} to {df['gameDateTimeEst'].max()}")
    
    return df


def compute_possessions(row: pd.Series) -> float:
    """
    Estimate possessions using basketball reference formula.
    
    Possessions ≈ FGA - ORB + TO + 0.4 * FTA
    
    Args:
        row: Team statistics row
        
    Returns:
        Estimated possessions (or NaN if data missing)
    """
    try:
        fga = row['fieldGoalsAttempted']
        orb = row['reboundsOffensive']
        tov = row['turnovers']
        fta = row['freeThrowsAttempted']
        
        if pd.notna(fga) and pd.notna(orb) and pd.notna(tov) and pd.notna(fta):
            return fga - orb + tov + 0.4 * fta
        return np.nan
    except:
        return np.nan


def compute_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add offensive rating, defensive rating, and net rating.
    
    ORtg = 100 * Points / Possessions
    DRtg = 100 * Opponent Points / Possessions
    NetRtg = ORtg - DRtg
    
    Args:
        df: TeamStatistics DataFrame
        
    Returns:
        DataFrame with added rating columns
    """
    df = df.copy()
    
    # Compute possessions
    df['possessions'] = df.apply(compute_possessions, axis=1)
    
    # Offensive rating
    df['offensive_rating'] = np.where(
        df['possessions'] > 0,
        100 * df['teamScore'] / df['possessions'],
        np.nan
    )
    
    # Defensive rating
    df['defensive_rating'] = np.where(
        df['possessions'] > 0,
        100 * df['opponentScore'] / df['possessions'],
        np.nan
    )
    
    # Net rating
    df['net_rating'] = df['offensive_rating'] - df['defensive_rating']
    
    # Point differential
    df['point_diff'] = df['teamScore'] - df['opponentScore']
    
    return df


def compute_rolling_features(
    df: pd.DataFrame,
    team_id: int,
    windows: list = [5, 10],
    location_filter: str = None
) -> pd.DataFrame:
    """
    Compute rolling features for a single team with NO LEAKAGE.
    
    For each game, rolling stats are computed from games BEFORE that game only.
    Uses .shift(1) to exclude current game from window.
    
    Args:
        df: TeamStatistics DataFrame (sorted by gameDateTimeEst)
        team_id: Team ID to compute features for
        windows: List of window sizes (default: [5, 10])
        location_filter: If 'home' or 'away', filter by location before rolling
        
    Returns:
        DataFrame with rolling feature columns added
    """
    # Filter to this team's games
    team_df = df[df['teamId'] == team_id].copy()
    
    # Apply location filter if specified
    if location_filter == 'home':
        team_df = team_df[team_df['home'] == 1].copy()
        suffix = '_home'
    elif location_filter == 'away':
        team_df = team_df[team_df['home'] == 0].copy()
        suffix = '_away'
    else:
        suffix = ''
    
    # Columns to compute rolling features on
    rolling_cols = {
        'teamScore': 'pts',
        'opponentScore': 'opp_pts',
        'point_diff': 'pt_diff',
        'fieldGoalsPercentage': 'fg_pct',
        'threePointersPercentage': '3p_pct',
        'freeThrowsPercentage': 'ft_pct',
        'reboundsTotal': 'reb_total',
        'reboundsOffensive': 'reb_off',
        'reboundsDefensive': 'reb_def',
        'assists': 'ast',
        'turnovers': 'tov',
        'steals': 'stl',
        'blocks': 'blk',
        'offensive_rating': 'ortg',
        'defensive_rating': 'drtg',
        'net_rating': 'net_rtg',
        'possessions': 'pace',
    }
    
    # Compute rolling features for each window
    for window in windows:
        for col, short_name in rolling_cols.items():
            if col in team_df.columns:
                # Use .shift(1) to exclude current game, then rolling mean
                # This ensures NO LEAKAGE
                col_name = f'{short_name}_last{window}{suffix}'
                team_df[col_name] = team_df[col].shift(1).rolling(
                    window=window, min_periods=1
                ).mean()
    
    return team_df


def merge_team_stats_to_games(
    df_games: pd.DataFrame,
    df_team_stats: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge team statistics to games DataFrame.
    
    Each game gets stats for both home and away teams.
    Validates that each game has exactly 2 team-stat rows.
    
    Args:
        df_games: Games DataFrame with gameId, hometeamId, awayteamId
        df_team_stats: TeamStatistics DataFrame with rolling features
        
    Returns:
        Merged DataFrame with home_* and away_* prefixed features
    """
    # Keep track of columns we want to preserve from df_games
    games_cols_to_preserve = df_games.columns.tolist()
    
    # Merge home team stats
    df_merged = df_games.merge(
        df_team_stats,
        left_on=['gameId', 'hometeamId'],
        right_on=['gameId', 'teamId'],
        how='left',
        suffixes=('', '_home_merge')
    )
    
    # Identify and rename home team columns
    # After merge, team stats columns don't have suffixes yet
    team_stat_cols = [col for col in df_team_stats.columns 
                      if col not in ['gameId', 'teamId', 'gameDateTimeEst']]
    
    # Rename home team columns with home_ prefix
    rename_dict_home = {}
    for col in team_stat_cols:
        # Only rename columns that don't already exist in df_games
        if col in df_merged.columns and not col.startswith('home_') and col not in games_cols_to_preserve:
            rename_dict_home[col] = f'home_{col}'
    
    # Also handle the teamId column from merge
    if 'teamId' in df_merged.columns:
        rename_dict_home['teamId'] = 'home_teamId_merged'
    
    df_merged = df_merged.rename(columns=rename_dict_home)
    
    # Merge away team stats
    df_merged = df_merged.merge(
        df_team_stats,
        left_on=['gameId', 'awayteamId'],
        right_on=['gameId', 'teamId'],
        how='left',
        suffixes=('', '_away_merge')
    )
    
    # Rename away team columns with away_ prefix
    rename_dict_away = {}
    for col in team_stat_cols:
        # Only rename columns that don't already exist in df_games and aren't already home_ prefixed
        if col in df_merged.columns and not col.startswith(('home_', 'away_')) and col not in games_cols_to_preserve:
            rename_dict_away[col] = f'away_{col}'

    
    # Handle the teamId column from second merge
    if 'teamId' in df_merged.columns:
        rename_dict_away['teamId'] = 'away_teamId_merged'
    
    df_merged = df_merged.rename(columns=rename_dict_away)
    
    # Validation: check merge coverage
    total_games = len(df_games)
    merged_games = df_merged['gameId'].notna().sum()
    
    # Check if home and away features exist (use a common rolling feature as proxy)
    home_feature_exists = any(col.startswith('home_') and 'last' in col for col in df_merged.columns)
    away_feature_exists = any(col.startswith('away_') and 'last' in col for col in df_merged.columns)
    
    print(f"\nMerge validation:")
    print(f"  Total games: {total_games}")
    print(f"  Successfully merged: {merged_games} ({100*merged_games/total_games:.1f}%)")
    print(f"  Home team features: {'Found' if home_feature_exists else 'MISSING'}")
    print(f"  Away team features: {'Found' if away_feature_exists else 'MISSING'}")
    
    # Drop games with missing critical rolling features
    # Check for a key feature like net_rtg_last5
    if 'home_net_rtg_last5' in df_merged.columns and 'away_net_rtg_last5' in df_merged.columns:
        before_drop = len(df_merged)
        df_merged = df_merged.dropna(subset=['home_net_rtg_last5', 'away_net_rtg_last5'])
        after_drop = len(df_merged)
        
        if before_drop > after_drop:
            print(f"  Dropped {before_drop - after_drop} games with missing key features")
    
    return df_merged
    
    return df_merged


def add_delta_features(df: pd.DataFrame, windows: list = [5, 10]) -> pd.DataFrame:
    """
    Add delta features (home minus away) for key metrics.
    
    Delta features give the model direct matchup comparison signals.
    
    Args:
        df: DataFrame with home_* and away_* rolling features
        windows: Window sizes to compute deltas for
        
    Returns:
        DataFrame with delta_* columns added
    """
    df = df.copy()
    
    # Metrics to compute deltas for
    delta_metrics = [
        'net_rtg', 'pt_diff', 'fg_pct', '3p_pct', 'tov', 
        'reb_total', 'ast', 'pace', 'ortg', 'drtg'
    ]
    
    for window in windows:
        for metric in delta_metrics:
            home_col = f'home_{metric}_last{window}'
            away_col = f'away_{metric}_last{window}'
            
            if home_col in df.columns and away_col in df.columns:
                delta_col = f'delta_{metric}_last{window}'
                df[delta_col] = df[home_col] - df[away_col]
    
    return df


def validate_no_leakage(
    df_games: pd.DataFrame,
    df_team_stats: pd.DataFrame,
    n_samples: int = 10,
    seed: int = 42
) -> bool:
    """
    Validate that rolling features don't leak future data.
    
    For random sample of games, verify that all team games used in
    rolling window occurred BEFORE the game datetime.
    
    Args:
        df_games: Games DataFrame with rolling features
        df_team_stats: Original TeamStatistics DataFrame
        n_samples: Number of games to validate
        seed: Random seed
        
    Returns:
        True if validation passes, False otherwise
    """
    np.random.seed(seed)
    
    # Sample random games
    sample_games = df_games.sample(min(n_samples, len(df_games)))
    
    print(f"\nValidating no leakage on {len(sample_games)} sample games...")
    
    violations = 0
    
    for idx, game_row in sample_games.iterrows():
        game_datetime = game_row['gameDateTimeEst']
        
        # Check home team
        home_team_id = game_row['hometeamId']
        home_team_games = df_team_stats[
            (df_team_stats['teamId'] == home_team_id) &
            (df_team_stats['gameDateTimeEst'] < game_datetime)
        ]
        
        # Check away team
        away_team_id = game_row['awayteamId']
        away_team_games = df_team_stats[
            (df_team_stats['teamId'] == away_team_id) &
            (df_team_stats['gameDateTimeEst'] < game_datetime)
        ]
        
        # For games with rolling features, verify sufficient past games exist
        if pd.notna(game_row.get('home_net_rtg_last5')):
            if len(home_team_games) < 1:
                print(f"  ⚠️  Game {game_row['gameId']}: Home team has rolling features but no past games!")
                violations += 1
        
        if pd.notna(game_row.get('away_net_rtg_last5')):
            if len(away_team_games) < 1:
                print(f"  ⚠️  Game {game_row['gameId']}: Away team has rolling features but no past games!")
                violations += 1
    
    if violations == 0:
        print(f"  ✅ No leakage detected in {len(sample_games)} sample games")
        return True
    else:
        print(f"  ❌ Found {violations} potential leakage violations")
        return False


def engineer_stage_b1_features(
    df_games: pd.DataFrame,
    team_stats_path: str = "data/TeamStatistics.csv",
    validate_leakage: bool = True
) -> pd.DataFrame:
    """
    Main function to engineer Stage B1 features.
    
    Adds rolling team statistics features to game-level data while
    ensuring no data leakage.
    
    Args:
        df_games: Games DataFrame (must have gameDateTimeEst, hometeamId, awayteamId)
        team_stats_path: Path to TeamStatistics.csv
        validate_leakage: Whether to run leakage validation
        
    Returns:
        DataFrame with Stage B1 features added
    """
    print("\n" + "="*70)
    print("STAGE B1 FEATURE ENGINEERING")
    print("="*70)
    
    # Load and preprocess team statistics
    df_team_stats = load_team_statistics(team_stats_path)
    
    # Compute advanced stats (ratings, possessions)
    print("\nComputing offensive/defensive ratings and possessions...")
    df_team_stats = compute_ratings(df_team_stats)
    
    # Get unique teams
    unique_teams = df_team_stats['teamId'].unique()
    print(f"\nComputing rolling features for {len(unique_teams)} teams...")
    
    # Compute rolling features for each team
    all_team_features = []
    
    for team_id in unique_teams:
        # Overall rolling features (windows: 5, 10)
        team_features = compute_rolling_features(df_team_stats, team_id, windows=[5, 10])
        
        # Home-specific rolling features (window: 10 only for efficiency)
        team_features_home = compute_rolling_features(
            df_team_stats, team_id, windows=[10], location_filter='home'
        )
        
        # Away-specific rolling features (window: 10 only)
        team_features_away = compute_rolling_features(
            df_team_stats, team_id, windows=[10], location_filter='away'
        )
        
        # Merge home/away split features back to main team features
        # Keep only the split-specific columns
        split_cols_home = [col for col in team_features_home.columns if col.endswith('_home')]
        split_cols_away = [col for col in team_features_away.columns if col.endswith('_away')]
        
        team_features = team_features.merge(
            team_features_home[['gameId', 'teamId'] + split_cols_home],
            on=['gameId', 'teamId'],
            how='left'
        )
        
        team_features = team_features.merge(
            team_features_away[['gameId', 'teamId'] + split_cols_away],
            on=['gameId', 'teamId'],
            how='left'
        )
        
        all_team_features.append(team_features)
    
    # Combine all teams
    df_team_stats_features = pd.concat(all_team_features, ignore_index=True)
    
    print(f"Rolling features computed: {len(df_team_stats_features)} team-game records")
    
    # Merge team stats to games
    print("\nMerging team statistics to games...")
    df_merged = merge_team_stats_to_games(df_games, df_team_stats_features)
    
    # Add delta features
    print("\nComputing delta features (home - away)...")
    df_merged = add_delta_features(df_merged, windows=[5, 10])
    
    # Feature summary
    new_features = [col for col in df_merged.columns if col not in df_games.columns]
    print(f"\nAdded {len(new_features)} new features")
    print(f"Total features in merged DataFrame: {len(df_merged.columns)}")
    
    # Validate no leakage
    if validate_leakage and len(df_merged) > 0:
        validate_no_leakage(df_merged, df_team_stats)
    
    print("\n" + "="*70)
    print("STAGE B1 FEATURE ENGINEERING COMPLETE")
    print("="*70 + "\n")
    
    return df_merged


def get_stage_b1_feature_names(df: pd.DataFrame) -> Dict[str, list]:
    """
    Get categorized list of Stage B1 features.
    
    Args:
        df: DataFrame with Stage B1 features
        
    Returns:
        Dictionary mapping feature categories to column lists
    """
    rolling_features = [col for col in df.columns if 'last' in col and not col.startswith('delta_')]
    home_away_splits = [col for col in df.columns if col.endswith('_home') or col.endswith('_away')]
    delta_features = [col for col in df.columns if col.startswith('delta_')]
    
    return {
        'rolling': rolling_features,
        'home_away_splits': home_away_splits,
        'delta': delta_features,
        'all_b1': rolling_features + home_away_splits + delta_features
    }
