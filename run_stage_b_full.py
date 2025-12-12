"""
Main pipeline for NBA game prediction - Stage B1 Full.

This variant uses the complete feature set with all rolling metrics:
- Stage A1 historical features (20)
- All rolling metrics (offensive/defensive/net rating, all shooting stats, rebounds, etc.)
- Windows: last 5, last 10 games  
- Home/away split features
- Optional seasonal reset for rolling windows

Configuration:
- FEATURE_SET = 'full' (177 features)
- ROLLING_RESET_BY_SEASON = True (reset at season boundaries)
"""

import sys
import argparse
from pathlib import Path

# Add directories to path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent / "features"))

from scripts.load_data import load_and_filter_games
from scripts.feature_engineering import compute_team_features, add_delta_features
from scripts.preprocessing import split_train_val_test, preprocess_data, save_preprocessed_data
from scripts.train_model import train_model, evaluate_model, save_results
from scripts.visualize import generate_comprehensive_report
from scripts.archive_manager import archive_previous_results
from features.stage_b_enhanced import engineer_stage_b1_features_configurable
from src.utils.thresholds import find_optimal_threshold
from src.utils.team_encoding import fit_team_encoder, transform_team_ids, save_team_encoder
from src.utils.embedding_viz import visualize_model_2_embeddings
import numpy as np

# Stage configuration
STAGE_NAME = "stage_b_full"
FEATURE_SET = "full"  # 'intermediate' or 'full'
ROLLING_RESET_BY_SEASON = True  # Reset rolling windows at season boundaries


def main(threshold_metric="f1", model_version=1, embedding_dim=16):
    """
    Run the complete Stage B Full pipeline.
    
    Args:
        threshold_metric: Metric to optimize threshold
        model_version: Model version (1: original MLP, 2: with team embeddings)
        embedding_dim: Embedding dimension for Model 2
    """
    print("="*80)
    print(f"NBA Game Prediction - {STAGE_NAME.upper()} Pipeline")
    print(f"Model Version: {model_version}")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Feature Set: {FEATURE_SET}")
    print(f"  Rolling Window Reset: {'By Season' if ROLLING_RESET_BY_SEASON else 'Continuous'}")
    print(f"  Target Features: ~177")
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
    
    # Step 2: Engineer Stage A features (historical)
    print("\n" + "="*80)
    print("STEP 2A: Engineering Stage A historical features")
    print("="*80)
    df_features = compute_team_features(df_filtered.copy())
    df_features = add_delta_features(df_features)
    
    stage_a_features = len(df_features.columns) - len(df_filtered.columns)
    print(f"Stage A features added: {stage_a_features}")
    
    # Step 2B: Engineer Stage B team statistics features (full)
    print("\n" + "="*80)
    print("STEP 2B: Engineering Stage B team statistics features")
    print("="*80)
    df_features = engineer_stage_b1_features_configurable(
        df_features,
        team_stats_path="data/TeamStatistics.csv",
        feature_set=FEATURE_SET,
        reset_by_season=ROLLING_RESET_BY_SEASON,
        validate_leakage=False
    )
    
    total_features = len(df_features.columns) - len(df_filtered.columns)
    stage_b_features = total_features - stage_a_features
    print(f"\nFeature summary:")
    print(f"  Stage A features: {stage_a_features}")
    print(f"  Stage B features: {stage_b_features}")
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
    
    # Step 3b: Prepare team encodings for Model 2 (if needed)
    team_to_id = None
    home_train_ids = home_val_ids = home_test_ids = None
    away_train_ids = away_val_ids = away_test_ids = None
    num_teams = None
    
    if model_version == 2:
        print("\n" + "="*80)
        print("STEP 3b: Preparing team encodings for Model 2")
        print("="*80)
        
        # Fit team encoder on training data only
        team_to_id = fit_team_encoder(train_df)
        num_teams = len(team_to_id) - 1  # Exclude UNK token
        print(f"Number of unique teams: {num_teams}")
        
        # Transform team names to IDs
        home_train_ids, away_train_ids = transform_team_ids(train_df, team_to_id)
        home_val_ids, away_val_ids = transform_team_ids(val_df, team_to_id)
        home_test_ids, away_test_ids = transform_team_ids(test_df, team_to_id)
        
        # Save team encoding
        team_encoding_path = output_path / "team_encoding.json"
        save_team_encoder(team_to_id, team_encoding_path)
        print(f"Team encoding saved to {team_encoding_path}")
    
    # Step 4: Train model
    print("\n" + "="*80)
    print(f"STEP 4: Training MLP model (Model {model_version})")
    print("="*80)
    models_path = Path("models") / STAGE_NAME
    models_path.mkdir(parents=True, exist_ok=True)
    model_save_path = str(models_path / "mlp.keras")
    
    model, history = train_model(
        X_train, y_train, X_val, y_val,
        model_save_path=model_save_path,
        model_version=model_version,
        home_train_ids=home_train_ids,
        away_train_ids=away_train_ids,
        home_val_ids=home_val_ids,
        away_val_ids=away_val_ids,
        num_teams=num_teams,
        embedding_dim=embedding_dim
    )
    
    # Step 5: Evaluate model and optimize threshold
    print("\n" + "="*80)
    print(f"STEP 5: Evaluating model and optimizing decision threshold (metric: {threshold_metric})")
    print("="*80)
    
    # Find optimal threshold on validation set
    print(f"\n--- Finding optimal threshold on validation set (optimizing {threshold_metric}) ---")
    if model_version == 1:
        y_val_pred_proba = model.predict(X_val, verbose=0).flatten()
    else:  # model_version == 2
        y_val_pred_proba = model.predict(
            [X_val, home_val_ids.reshape(-1, 1), away_val_ids.reshape(-1, 1)],
            verbose=0
        ).flatten()
    
    optimal_threshold, metric_values = find_optimal_threshold(y_val, y_val_pred_proba, metric=threshold_metric)
    
    # Evaluate with default threshold (0.5)
    print("\n--- Evaluation with default threshold (0.5) ---")
    train_metrics_default = evaluate_model(
        model, X_train, y_train, "Train", threshold=0.5,
        model_version=model_version,
        home_ids=home_train_ids,
        away_ids=away_train_ids
    )
    val_metrics_default = evaluate_model(
        model, X_val, y_val, "Validation", threshold=0.5,
        model_version=model_version,
        home_ids=home_val_ids,
        away_ids=away_val_ids
    )
    test_metrics_default = evaluate_model(
        model, X_test, y_test, "Test", threshold=0.5,
        model_version=model_version,
        home_ids=home_test_ids,
        away_ids=away_test_ids
    )
    
    # Evaluate with optimal threshold
    print(f"\n--- Evaluation with optimal threshold ({optimal_threshold:.3f}) ---")
    train_metrics = evaluate_model(
        model, X_train, y_train, "Train", threshold=optimal_threshold,
        model_version=model_version,
        home_ids=home_train_ids,
        away_ids=away_train_ids
    )
    val_metrics = evaluate_model(
        model, X_val, y_val, "Validation", threshold=optimal_threshold,
        model_version=model_version,
        home_ids=home_val_ids,
        away_ids=away_val_ids
    )
    test_metrics = evaluate_model(
        model, X_test, y_test, "Test", threshold=optimal_threshold,
        model_version=model_version,
        home_ids=home_test_ids,
        away_ids=away_test_ids
    )
    
    # Save results
    save_results(
        train_metrics, val_metrics, test_metrics, history,
        optimal_threshold=optimal_threshold,
        threshold_metric=threshold_metric,
        output_path=str(output_path / "results.json")
    )
    
    # Step 5b: Visualize team embeddings (Model 2 only)
    if model_version == 2:
        print("\n" + "="*80)
        print("STEP 5b: Visualizing team embeddings")
        print("="*80)
        visualize_model_2_embeddings(
            model=model,
            team_to_id=team_to_id,
            stage_name="2B-full",
            embedding_dim=embedding_dim,
            threshold_metric=threshold_metric,
            output_dir=output_path,
            reduction_method='pca',
            show=False
        )
    
    # Step 6: Generate comprehensive visualization report (skip for Model 2 due to complex input handling)
    if model_version == 1:
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
    else:
        print("\n" + "="*80)
        print("STEP 6: Skipping comprehensive visualization report for Model 2")
        print("="*80)
        print("(Model 2 requires special input handling for visualization)")
        plots_path = output_path / "plots"
        plots_path.mkdir(parents=True, exist_ok=True)
    
    # Summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETE - SUMMARY")
    print("="*80)
    print(f"Stage: {STAGE_NAME.upper()}")
    print(f"Feature Set: {FEATURE_SET}")
    print(f"Rolling Reset: {'By Season' if ROLLING_RESET_BY_SEASON else 'Continuous'}")
    print(f"Total games processed: {len(df_filtered)}")
    print(f"Training set: {len(X_train)} games")
    print(f"Validation set: {len(X_val)} games")
    print(f"Test set: {len(X_test)} games")
    print(f"Final features used: {X_train.shape[1]}")
    print(f"\nFinal Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Final Test F1 Score: {test_metrics['f1_score']:.4f}")
    print(f"Final Test AUC: {test_metrics['auc']:.4f}")
    print(f"Optimal Decision Threshold: {optimal_threshold:.3f} (optimized for {threshold_metric})")
    print(f"\nModel saved to: {model_save_path}")
    print(f"Results saved to: {output_path / 'results.json'}")
    print(f"Visualizations saved to: {plots_path}/")
    print("="*80)
    
    # Archive results at the end of the run
    print("\n")
    model_suffix = f"_model{model_version}" if model_version == 2 else ""
    archive_previous_results(
        outputs_dir=str(output_path),
        archive_base_dir=f"archives/{STAGE_NAME}",
        run_suffix=f"threshold-{threshold_metric}{model_suffix}"
    )
    
    print("\n" + "="*80)
    print(f"{STAGE_NAME.upper()} PIPELINE COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Stage B1 Full pipeline with configurable threshold optimization and model version")
    parser.add_argument(
        "--threshold_metric",
        type=str,
        default="f1",
        choices=["f1", "accuracy", "balanced_accuracy"],
        help="Metric to optimize when selecting decision threshold (default: f1)"
    )
    parser.add_argument(
        "--model_version",
        type=int,
        default=1,
        choices=[1, 2],
        help="Model version: 1 for original MLP, 2 for MLP with team embeddings (default: 1)"
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=16,
        help="Embedding dimension for Model 2 (default: 16, ignored for Model 1)"
    )
    args = parser.parse_args()
    
    main(
        threshold_metric=args.threshold_metric,
        model_version=args.model_version,
        embedding_dim=args.embedding_dim
    )
