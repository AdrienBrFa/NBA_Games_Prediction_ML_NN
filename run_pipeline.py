"""
Main pipeline for NBA game prediction - Stage A1.

This script runs the complete pipeline from raw data to trained model:
1. Load and filter data
2. Engineer features
3. Preprocess data
4. Train model
5. Evaluate and save results
"""

import sys
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from scripts.load_data import load_and_filter_games
from scripts.feature_engineering import compute_team_features, add_delta_features
from scripts.preprocessing import split_train_val_test, preprocess_data, save_preprocessed_data
from scripts.train_model import train_model, evaluate_model, save_results


def main():
    """Run the complete Stage A1 pipeline."""
    print("="*80)
    print("NBA Game Prediction - Stage A1 Pipeline")
    print("="*80)
    
    # Step 1: Load and filter data
    print("\n" + "="*80)
    print("STEP 1: Loading and filtering data")
    print("="*80)
    df_filtered = load_and_filter_games("data/Games.csv")
    
    # Save filtered data
    output_path = Path("outputs")
    output_path.mkdir(parents=True, exist_ok=True)
    df_filtered.to_csv(output_path / "filtered_games.csv", index=False)
    print(f"Filtered data saved to {output_path / 'filtered_games.csv'}")
    
    # Step 2: Engineer features
    print("\n" + "="*80)
    print("STEP 2: Engineering features")
    print("="*80)
    df_features = compute_team_features(df_filtered.copy())
    df_features = add_delta_features(df_features)
    df_features.to_csv(output_path / "features_engineered.csv", index=False)
    print(f"Engineered features saved to {output_path / 'features_engineered.csv'}")
    
    # Step 3: Split and preprocess data
    print("\n" + "="*80)
    print("STEP 3: Splitting and preprocessing data")
    print("="*80)
    train_df, val_df, test_df = split_train_val_test(df_features)
    X_train, y_train, X_val, y_val, X_test, y_test, preprocessing_objects = preprocess_data(
        train_df, val_df, test_df
    )
    save_preprocessed_data(X_train, y_train, X_val, y_val, X_test, y_test, preprocessing_objects)
    
    # Step 4: Train model
    print("\n" + "="*80)
    print("STEP 4: Training MLP model")
    print("="*80)
    Path("models").mkdir(parents=True, exist_ok=True)
    model, history = train_model(X_train, y_train, X_val, y_val)
    
    # Step 5: Evaluate model
    print("\n" + "="*80)
    print("STEP 5: Evaluating model")
    print("="*80)
    train_metrics = evaluate_model(model, X_train, y_train, "Train")
    val_metrics = evaluate_model(model, X_val, y_val, "Validation")
    test_metrics = evaluate_model(model, X_test, y_test, "Test")
    
    # Save results
    save_results(train_metrics, val_metrics, test_metrics, history)
    
    # Summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETE - SUMMARY")
    print("="*80)
    print(f"Total games processed: {len(df_filtered)}")
    print(f"Training set: {len(X_train)} games")
    print(f"Validation set: {len(X_val)} games")
    print(f"Test set: {len(X_test)} games")
    print(f"\nFinal Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Final Test AUC: {test_metrics['auc']:.4f}")
    print(f"\nModel saved to: models/stage_a1_mlp.keras")
    print(f"Results saved to: outputs/results.json")
    print("="*80)


if __name__ == "__main__":
    main()
