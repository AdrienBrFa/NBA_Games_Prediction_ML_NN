"""
Stage B1 Feature Engineering with Seasonal Reset

Enhanced version with configurable feature sets and seasonal reset option.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
import sys

# Import base functions
sys.path.insert(0, str(Path(__file__).parent))
from stage_b_teamstats import (
    load_team_statistics,
    compute_ratings,
    merge_team_stats_to_games
)
from stage_b_config import (
    INTERMEDIATE_ROLLING_METRICS,
    INTERMEDIATE_WINDOWS,
    INTERMEDIATE_DELTA_METRICS,
    get_intermediate_feature_list,
    print_intermediate_feature_summary
)


def compute_rolling_features_with_season_reset(
    df: pd.DataFrame,
    team_id: int,
    windows: list = [5, 10],
    location_filter: str = None,
    reset_by_season: bool = True,
    feature_set: str = 'intermediate'
) -> pd.DataFrame:
    """
    Compute rolling features with optional seasonal reset.
    
    Args:
        df: TeamStatistics DataFrame with seasonYear column
        team_id: Team ID to compute features for
        windows: List of window sizes
        location_filter: 'home', 'away', or None
        reset_by_season: If True, rolling windows reset at season boundaries
        feature_set: 'full' or 'intermediate' - which metrics to compute
        
    Returns:
        DataFrame with rolling features
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
    
    # Select metrics based on feature set
    if feature_set == 'intermediate':
        rolling_cols = INTERMEDIATE_ROLLING_METRICS
    else:  # full
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
    
    # Compute rolling features
    if reset_by_season and 'seasonYear' in team_df.columns:
        # Group by season and compute rolling within each season
        for window in windows:
            for col, short_name in rolling_cols.items():
                if col in team_df.columns:
                    col_name = f'{short_name}_last{window}{suffix}'
                    # Compute rolling per season group
                    team_df[col_name] = team_df.groupby('seasonYear')[col].transform(
                        lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
                    )
    else:
        # No seasonal reset - continuous rolling across seasons
        for window in windows:
            for col, short_name in rolling_cols.items():
                if col in team_df.columns:
                    col_name = f'{short_name}_last{window}{suffix}'
                    team_df[col_name] = team_df[col].shift(1).rolling(
                        window=window, min_periods=1
                    ).mean()
    
    return team_df


def add_delta_features_selective(
    df: pd.DataFrame, 
    windows: list = [5, 10],
    feature_set: str = 'intermediate'
) -> pd.DataFrame:
    """
    Add delta features with optional filtering.
    
    Args:
        df: DataFrame with home_* and away_* features
        windows: Window sizes
        feature_set: 'full' or 'intermediate'
        
    Returns:
        DataFrame with delta features
    """
    df = df.copy()
    
    # Select metrics based on feature set
    if feature_set == 'intermediate':
        delta_metrics = INTERMEDIATE_DELTA_METRICS
    else:  # full
        delta_metrics = [
            'net_rtg', 'pt_diff', 'fg_pct', '3p_pct', 'tov',
            'reb_total', 'ast', 'pace', 'ortg', 'drtg',
            'pts', 'opp_pts', 'reb_off', 'reb_def', 'stl', 'blk'
        ]
    
    for window in windows:
        for metric in delta_metrics:
            home_col = f'home_{metric}_last{window}'
            away_col = f'away_{metric}_last{window}'
            
            if home_col in df.columns and away_col in df.columns:
                delta_col = f'delta_{metric}_last{window}'
                df[delta_col] = df[home_col] - df[away_col]
    
    return df


def engineer_stage_b1_features_configurable(
    df_games: pd.DataFrame,
    team_stats_path: str = "data/TeamStatistics.csv",
    feature_set: str = 'intermediate',
    reset_by_season: bool = True,
    validate_leakage: bool = False
) -> pd.DataFrame:
    """
    Main function to engineer Stage B1 features with configuration options.
    
    Args:
        df_games: Games DataFrame
        team_stats_path: Path to TeamStatistics.csv
        feature_set: 'full' or 'intermediate'
        reset_by_season: Whether to reset rolling windows at season boundaries
        validate_leakage: Whether to run leakage validation
        
    Returns:
        DataFrame with configured features
    """
    print("\n" + "="*70)
    print(f"STAGE B1 FEATURE ENGINEERING - {feature_set.upper()} VARIANT")
    print(f"Rolling window reset by season: {reset_by_season}")
    print("="*70)
    
    # Load and preprocess team statistics
    df_team_stats = load_team_statistics(team_stats_path)
    
    # Add seasonYear to team stats if not present
    if 'seasonYear' not in df_team_stats.columns:
        df_team_stats['year'] = df_team_stats['gameDateTimeEst'].dt.year
        df_team_stats['month'] = df_team_stats['gameDateTimeEst'].dt.month
        df_team_stats['seasonYear'] = np.where(
            df_team_stats['month'] >= 8,
            df_team_stats['year'],
            df_team_stats['year'] - 1
        )
    
    # Compute advanced stats
    print("\nComputing offensive/defensive ratings and possessions...")
    df_team_stats = compute_ratings(df_team_stats)
    
    # Get unique teams
    unique_teams = df_team_stats['teamId'].unique()
    print(f"\nComputing rolling features for {len(unique_teams)} teams...")
    print(f"Feature set: {feature_set}")
    
    # Select windows based on feature set
    if feature_set == 'intermediate':
        windows = INTERMEDIATE_WINDOWS
    else:
        windows = [5, 10]
    
    # Compute rolling features for each team
    all_team_features = []
    
    for team_id in unique_teams:
        # Overall rolling features
        team_features = compute_rolling_features_with_season_reset(
            df_team_stats, team_id, 
            windows=windows,
            reset_by_season=reset_by_season,
            feature_set=feature_set
        )
        
        # Skip home/away splits for intermediate to keep feature count down
        if feature_set == 'full':
            # Home-specific rolling features
            team_features_home = compute_rolling_features_with_season_reset(
                df_team_stats, team_id,
                windows=[10],
                location_filter='home',
                reset_by_season=reset_by_season,
                feature_set=feature_set
            )
            
            # Away-specific rolling features
            team_features_away = compute_rolling_features_with_season_reset(
                df_team_stats, team_id,
                windows=[10],
                location_filter='away',
                reset_by_season=reset_by_season,
                feature_set=feature_set
            )
            
            # Merge split features
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
    df_merged = add_delta_features_selective(df_merged, windows=windows, feature_set=feature_set)
    
    # Feature summary
    new_features = [col for col in df_merged.columns if col not in df_games.columns]
    print(f"\nAdded {len(new_features)} new features")
    print(f"Total columns in merged DataFrame: {len(df_merged.columns)}")
    
    if feature_set == 'intermediate':
        print_intermediate_feature_summary()
    
    print("\n" + "="*70)
    print(f"STAGE B1 {feature_set.upper()} FEATURE ENGINEERING COMPLETE")
    print("="*70 + "\n")
    
    return df_merged
