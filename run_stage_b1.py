"""
Main pipeline for NBA game prediction - Stage B1.

This script runs the complete Stage B1 pipeline from raw data to trained model:
1. Load and filter data (Games.csv + TeamStatistics.csv)
2. Engineer historical features + team statistics features
3. Preprocess data (one-hot encoding, scaling)
4. Train MLP model with regularization
5. Evaluate and optimize decision threshold
6. Generate comprehensive visualizations
7. Archive results for comparison

Stage B1 extends Stage A1 by incorporating team statistics data
(offensive/defensive ratings, shooting efficiency, etc.).
"""

import sys
from pathlib import Path

# Add scripts and features directories to path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent / "features"))

from scripts.load_data import load_and_filter_games
from scripts.feature_engineering import compute_team_features, add_delta_features
from scripts.preprocessing import split_train_val_test, preprocess_data, save_preprocessed_data
from scripts.train_model import train_model, evaluate_model, save_results, find_optimal_threshold
from scripts.visualize import generate_comprehensive_report
from scripts.archive_manager import archive_previous_results
from features.stage_b1_teamstats import engineer_stage_b1_features
import numpy as np

# Stage configuration
STAGE_NAME = "stage_b1"


def main():
    """Run the complete Stage B1 pipeline."""
    print("="*80)
    print(f"NBA Game Prediction - {STAGE_NAME.upper()} Pipeline")
    print("="*80)
    print("\nStage B1: Team Statistics Integration")
    print("  - Stage A1 historical features")
    print("  - Rolling team statistics (last 5, 10 games)")
    print("  - Home/away split features")
    print("  - Delta features (home - away)")
    print("="*80)
    
    # Step 1: Load and filter data
    print("\n" + "="*80)
    print("STEP 1: Loading and filtering data")
    print("="*80)
    df_filtered = load_and_filter_games("data/Games.csv")
    
    # Save filtered data
    output_path = Path("outputs") / STAGE_NAME
    output_path.mkdir(parents=True, exist_ok=True)
    df_filtered.to_csv(output_path / "filtered_games.csv", index=False)
    print(f"Filtered data saved to {output_path / 'filtered_games.csv'}")
    
    # Step 2: Engineer Stage A1 features (historical)
    print("\n" + "="*80)
    print("STEP 2A: Engineering Stage A1 historical features")
    print("="*80)
    df_features = compute_team_features(df_filtered.copy())
    df_features = add_delta_features(df_features)
    
    stage_a1_features = len(df_features.columns) - len(df_filtered.columns)
    print(f"Stage A1 features added: {stage_a1_features}")
    
    # Step 2B: Engineer Stage B1 team statistics features
    print("\n" + "="*80)
    print("STEP 2B: Engineering Stage B1 team statistics features")
    print("="*80)
    df_features = engineer_stage_b1_features(
        df_features,
        team_stats_path="data/TeamStatistics.csv",
        validate_leakage=True
    )
    
    total_features = len(df_features.columns) - len(df_filtered.columns)
    stage_b1_features = total_features - stage_a1_features
    print(f"\nFeature summary:")
    print(f"  Stage A1 features: {stage_a1_features}")
    print(f"  Stage B1 features: {stage_b1_features}")
    print(f"  Total features: {total_features}")
    
    df_features.to_csv(output_path / "features_engineered.csv", index=False)
    print(f"\nEngineered features saved to {output_path / 'features_engineered.csv'}")
    
    # Step 3: Split and preprocess data
    print("\n" + "="*80)
    print("STEP 3: Splitting and preprocessing data")
    print("="*80)
    train_df, val_df, test_df = split_train_val_test(df_features)
    X_train, y_train, X_val, y_val, X_test, y_test, preprocessing_objects = preprocess_data(
        train_df, val_df, test_df
    )
    save_preprocessed_data(
        X_train, y_train, X_val, y_val, X_test, y_test, preprocessing_objects,
        output_dir=str(output_path)
    )
    
    # Step 4: Train model
    print("\n" + "="*80)
    print("STEP 4: Training MLP model")
    print("="*80)
    models_path = Path("models") / STAGE_NAME
    models_path.mkdir(parents=True, exist_ok=True)
    model_save_path = str(models_path / "mlp.keras")
    model, history = train_model(X_train, y_train, X_val, y_val, model_save_path=model_save_path)
    
    # Step 5: Evaluate model and optimize threshold
    print("\n" + "="*80)
    print("STEP 5: Evaluating model and optimizing decision threshold")
    print("="*80)
    
    # First, find optimal threshold on validation set
    print("\n--- Finding optimal threshold on validation set ---")
    y_val_pred_proba = model.predict(X_val, verbose=0).flatten()
    optimal_threshold, _ = find_optimal_threshold(y_val, y_val_pred_proba, method='f1')
    
    # Evaluate with default threshold (0.5)
    print("\n--- Evaluation with default threshold (0.5) ---")
    train_metrics_default = evaluate_model(model, X_train, y_train, "Train", threshold=0.5)
    val_metrics_default = evaluate_model(model, X_val, y_val, "Validation", threshold=0.5)
    test_metrics_default = evaluate_model(model, X_test, y_test, "Test", threshold=0.5)
    
    # Evaluate with optimal threshold
    print(f"\n--- Evaluation with optimal threshold ({optimal_threshold:.3f}) ---")
    train_metrics = evaluate_model(model, X_train, y_train, "Train", threshold=optimal_threshold)
    val_metrics = evaluate_model(model, X_val, y_val, "Validation", threshold=optimal_threshold)
    test_metrics = evaluate_model(model, X_test, y_test, "Test", threshold=optimal_threshold)
    
    # Save results
    save_results(
        train_metrics, val_metrics, test_metrics, history,
        optimal_threshold=optimal_threshold,
        output_path=str(output_path / "results.json")
    )
    
    # Step 6: Generate comprehensive visualization report
    print("\n" + "="*80)
    print("STEP 6: Generating comprehensive visualization report")
    print("="*80)
    
    # Get feature names from preprocessing objects
    feature_names = preprocessing_objects['all_features']
    
    plots_path = output_path / "plots"
    generate_comprehensive_report(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        history=history,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        feature_names=feature_names,
        output_dir=str(plots_path),
        show=False
    )
    
    # Summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETE - SUMMARY")
    print("="*80)
    print(f"Stage: {STAGE_NAME.upper()}")
    print(f"Total games processed: {len(df_filtered)}")
    print(f"Training set: {len(X_train)} games")
    print(f"Validation set: {len(X_val)} games")
    print(f"Test set: {len(X_test)} games")
    print(f"\nFinal Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Final Test F1 Score: {test_metrics['f1_score']:.4f}")
    print(f"Final Test AUC: {test_metrics['auc']:.4f}")
    print(f"Optimal Decision Threshold: {optimal_threshold:.3f}")
    print(f"\nModel saved to: {model_save_path}")
    print(f"Results saved to: {output_path / 'results.json'}")
    print(f"Visualizations saved to: {plots_path}/")
    print("="*80)
    
    # Archive results at the end of the run
    print("\n")
    archive_previous_results(
        outputs_dir=str(output_path),
        archive_base_dir=f"archives/{STAGE_NAME}"
    )
    
    print("\n" + "="*80)
    print(f"{STAGE_NAME.upper()} PIPELINE COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
