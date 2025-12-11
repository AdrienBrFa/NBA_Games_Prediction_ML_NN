"""
Preprocessing pipeline for NBA game prediction (Stage A1).

This script:
1. Splits data into train/validation/test sets
2. One-hot encodes team IDs
3. Handles missing values
4. Scales numeric features
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from typing import Tuple, Dict
from utils import parse_datetime_column


def split_train_val_test(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data by seasonYear.
    
    Train: 1990-2015
    Validation: 2016-2019
    Test: 2020-2023
    
    Args:
        df: Full dataset with features
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    print("Splitting data by seasonYear...")
    
    train_df = df[(df['seasonYear'] >= 1990) & (df['seasonYear'] <= 2015)].copy()
    val_df = df[(df['seasonYear'] >= 2016) & (df['seasonYear'] <= 2019)].copy()
    test_df = df[(df['seasonYear'] >= 2020) & (df['seasonYear'] <= 2023)].copy()
    
    print(f"Train set: {len(train_df)} games ({train_df['seasonYear'].min()}-{train_df['seasonYear'].max()})")
    print(f"Validation set: {len(val_df)} games ({val_df['seasonYear'].min()}-{val_df['seasonYear'].max()})")
    print(f"Test set: {len(test_df)} games ({test_df['seasonYear'].min()}-{test_df['seasonYear'].max()})")
    
    return train_df, val_df, test_df


def preprocess_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Preprocess train/val/test data.
    
    Steps:
    1. Extract target variable (y)
    2. One-hot encode team IDs
    3. Select and order numeric features
    4. Impute missing values (using training set statistics)
    5. Scale features (using training set statistics)
    
    Args:
        train_df: Training set
        val_df: Validation set
        test_df: Test set
        
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test, preprocessing_objects)
    """
    print("\nPreprocessing data...")
    
    # Extract target
    y_train = train_df['y'].values
    y_val = val_df['y'].values
    y_test = test_df['y'].values
    
    # Identify all numeric features dynamically (exclude ID columns and target)
    exclude_cols = [
        'gameId', 'gameDateTimeEst', 'hometeamId', 'awayteamId', 
        'hometeamCity', 'hometeamName', 'awayteamCity', 'awayteamName',
        'homeScore', 'awayScore', 'winner', 'y',
        'home_teamCity', 'home_teamName', 'away_teamCity', 'away_teamName',
        'home_opponentTeamCity', 'home_opponentTeamName', 'away_opponentTeamCity', 'away_opponentTeamName',
        'home_teamId_merged', 'away_teamId_merged', 'home_opponentTeamId', 'away_opponentTeamId',
        'home_home', 'away_home', 'home_win', 'away_win', 'home_coachId', 'away_coachId'
    ]
    
    # Get all columns that are numeric and not in exclude list
    numeric_features = []
    for col in train_df.columns:
        if col not in exclude_cols and pd.api.types.is_numeric_dtype(train_df[col]):
            numeric_features.append(col)
    
    print(f"Detected {len(numeric_features)} numeric features")
    
    # Stage A1 core features (for reference/validation)
    stage_a1_features = [
        'seasonYear',
        'home_season_games_played', 'home_season_wins', 'home_season_losses', 'home_season_win_pct',
        'away_season_games_played', 'away_season_wins', 'away_season_losses', 'away_season_win_pct',
        'home_last5_games_played', 'home_last5_win_pct', 'home_last5_avg_point_diff',
        'away_last5_games_played', 'away_last5_win_pct', 'away_last5_avg_point_diff',
        'home_days_since_last_game', 'away_days_since_last_game',
        'delta_season_win_pct', 'delta_last5_win_pct', 'delta_last5_point_diff', 'delta_days_rest'
    ]
    
    # One-hot encode team IDs
    print("One-hot encoding team IDs...")
    
    # Combine train/val/test to get all unique teams
    all_teams_home = pd.concat([
        train_df['hometeamId'],
        val_df['hometeamId'],
        test_df['hometeamId']
    ]).unique()
    
    all_teams_away = pd.concat([
        train_df['awayteamId'],
        val_df['awayteamId'],
        test_df['awayteamId']
    ]).unique()
    
    all_teams = sorted(set(all_teams_home) | set(all_teams_away))
    
    # Create one-hot encoded features
    for team in all_teams:
        train_df[f'home_team_{team}'] = (train_df['hometeamId'] == team).astype(int)
        train_df[f'away_team_{team}'] = (train_df['awayteamId'] == team).astype(int)
        
        val_df[f'home_team_{team}'] = (val_df['hometeamId'] == team).astype(int)
        val_df[f'away_team_{team}'] = (val_df['awayteamId'] == team).astype(int)
        
        test_df[f'home_team_{team}'] = (test_df['hometeamId'] == team).astype(int)
        test_df[f'away_team_{team}'] = (test_df['awayteamId'] == team).astype(int)
    
    # Get one-hot feature names
    onehot_features = [f'home_team_{team}' for team in all_teams] + [f'away_team_{team}' for team in all_teams]
    
    # Combine all features
    all_features = numeric_features + onehot_features
    
    print(f"Total features: {len(all_features)} ({len(numeric_features)} numeric + {len(onehot_features)} one-hot)")
    
    # Extract feature matrices
    X_train_raw = train_df[all_features].values
    X_val_raw = val_df[all_features].values
    X_test_raw = test_df[all_features].values
    
    # Impute missing values (using median from training set for numeric features)
    print("Imputing missing values...")
    imputer = SimpleImputer(strategy='median')
    
    # Only impute numeric features (one-hot features have no missing values)
    X_train_numeric = X_train_raw[:, :len(numeric_features)]
    X_val_numeric = X_val_raw[:, :len(numeric_features)]
    X_test_numeric = X_test_raw[:, :len(numeric_features)]
    
    X_train_numeric_imputed = imputer.fit_transform(X_train_numeric)
    X_val_numeric_imputed = imputer.transform(X_val_numeric)
    X_test_numeric_imputed = imputer.transform(X_test_numeric)
    
    # Scale numeric features (using StandardScaler on training set)
    print("Scaling numeric features...")
    scaler = StandardScaler()
    
    X_train_numeric_scaled = scaler.fit_transform(X_train_numeric_imputed)
    X_val_numeric_scaled = scaler.transform(X_val_numeric_imputed)
    X_test_numeric_scaled = scaler.transform(X_test_numeric_imputed)
    
    # Combine scaled numeric features with one-hot features
    X_train_onehot = X_train_raw[:, len(numeric_features):]
    X_val_onehot = X_val_raw[:, len(numeric_features):]
    X_test_onehot = X_test_raw[:, len(numeric_features):]
    
    X_train = np.concatenate([X_train_numeric_scaled, X_train_onehot], axis=1)
    X_val = np.concatenate([X_val_numeric_scaled, X_val_onehot], axis=1)
    X_test = np.concatenate([X_test_numeric_scaled, X_test_onehot], axis=1)
    
    # Store preprocessing objects for later use
    preprocessing_objects = {
        'imputer': imputer,
        'scaler': scaler,
        'numeric_features': numeric_features,
        'onehot_features': onehot_features,
        'all_features': all_features,
        'all_teams': all_teams
    }
    
    print(f"Final shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"Class balance - Train: {y_train.mean():.3f}, Val: {y_val.mean():.3f}, Test: {y_test.mean():.3f}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, preprocessing_objects


def save_preprocessed_data(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    preprocessing_objects: Dict,
    output_dir: str = "outputs"
):
    """Save preprocessed data and preprocessing objects."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save arrays
    np.save(output_path / "X_train.npy", X_train)
    np.save(output_path / "y_train.npy", y_train)
    np.save(output_path / "X_val.npy", X_val)
    np.save(output_path / "y_val.npy", y_val)
    np.save(output_path / "X_test.npy", X_test)
    np.save(output_path / "y_test.npy", y_test)
    
    # Save preprocessing objects
    with open(output_path / "preprocessing_objects.pkl", "wb") as f:
        pickle.dump(preprocessing_objects, f)
    
    print(f"\nPreprocessed data saved to {output_dir}/")


if __name__ == "__main__":
    # Load engineered features
    df = pd.read_csv("outputs/features_engineered.csv", low_memory=False)
    df = parse_datetime_column(df, 'gameDateTimeEst')
    
    # Split data
    train_df, val_df, test_df = split_train_val_test(df)
    
    # Preprocess
    X_train, y_train, X_val, y_val, X_test, y_test, preprocessing_objects = preprocess_data(
        train_df, val_df, test_df
    )
    
    # Save
    save_preprocessed_data(X_train, y_train, X_val, y_val, X_test, y_test, preprocessing_objects)
