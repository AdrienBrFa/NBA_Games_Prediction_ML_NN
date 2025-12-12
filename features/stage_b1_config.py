"""
Stage B1 Intermediate Feature Configuration

Defines reduced feature set for Stage B1 intermediate variant.
Target: 40-60 features (vs 177 in full variant)
"""

# Feature set selection
INTERMEDIATE_ROLLING_METRICS = {
    # Core performance metrics
    'offensive_rating': 'ortg',
    'defensive_rating': 'drtg', 
    'net_rating': 'net_rtg',
    'possessions': 'pace',
    
    # Shooting efficiency
    'fieldGoalsPercentage': 'fg_pct',
    'threePointersPercentage': '3p_pct',
    'freeThrowsPercentage': 'ft_pct',
    
    # Key stats
    'turnovers': 'tov',
    'reboundsTotal': 'reb_total',
    'assists': 'ast',
}

# Windows to use for intermediate
INTERMEDIATE_WINDOWS = [5, 10]

# Delta features for intermediate (home - away comparisons)
INTERMEDIATE_DELTA_METRICS = [
    'net_rtg', 'ortg', 'drtg', 'fg_pct', '3p_pct', 
    'tov', 'reb_total', 'ast', 'pace'
]


def get_intermediate_feature_list():
    """
    Get exact list of intermediate features.
    
    Returns:
        List of feature names that should be kept
    """
    features = []
    
    # Stage A1 features (always included)
    stage_a1 = [
        'seasonYear',
        'home_season_games_played', 'home_season_wins', 'home_season_losses', 'home_season_win_pct',
        'away_season_games_played', 'away_season_wins', 'away_season_losses', 'away_season_win_pct',
        'home_last5_games_played', 'home_last5_win_pct', 'home_last5_avg_point_diff',
        'away_last5_games_played', 'away_last5_win_pct', 'away_last5_avg_point_diff',
        'home_days_since_last_game', 'away_days_since_last_game',
        'delta_season_win_pct', 'delta_last5_win_pct', 'delta_last5_point_diff', 'delta_days_rest'
    ]
    features.extend(stage_a1)
    
    # Rolling features for home team
    for metric_short in INTERMEDIATE_ROLLING_METRICS.values():
        for window in INTERMEDIATE_WINDOWS:
            features.append(f'home_{metric_short}_last{window}')
    
    # Rolling features for away team
    for metric_short in INTERMEDIATE_ROLLING_METRICS.values():
        for window in INTERMEDIATE_WINDOWS:
            features.append(f'away_{metric_short}_last{window}')
    
    # Delta features
    for metric_short in INTERMEDIATE_DELTA_METRICS:
        for window in INTERMEDIATE_WINDOWS:
            features.append(f'delta_{metric_short}_last{window}')
    
    return features


def print_intermediate_feature_summary():
    """Print summary of intermediate feature set."""
    features = get_intermediate_feature_list()
    
    stage_a1_count = sum(1 for f in features if not f.startswith(('home_', 'away_', 'delta_')) or 'season' in f or 'last5' in f and 'last' not in f.replace('last5', ''))
    home_rolling = sum(1 for f in features if f.startswith('home_') and '_last' in f)
    away_rolling = sum(1 for f in features if f.startswith('away_') and '_last' in f)
    delta_features = sum(1 for f in features if f.startswith('delta_') and '_last' in f)
    
    print(f"\n{'='*70}")
    print("INTERMEDIATE FEATURE SET SUMMARY")
    print(f"{'='*70}")
    print(f"Stage A1 historical features: {stage_a1_count}")
    print(f"Home team rolling features: {home_rolling}")
    print(f"Away team rolling features: {away_rolling}")
    print(f"Delta features: {delta_features}")
    print(f"Total features: {len(features)}")
    print(f"{'='*70}\n")
    
    return len(features)
