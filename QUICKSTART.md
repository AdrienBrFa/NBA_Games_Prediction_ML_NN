# Quick Start Guide

This guide helps you get started with the NBA Game Prediction Stage A1 pipeline.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/AdrienBrFa/NBA_Games_Prediction_ML_NN.git
cd NBA_Games_Prediction_ML_NN
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Pipeline

### Complete Pipeline

To run the entire pipeline from data loading to model training:

```bash
python run_pipeline.py
```

This will:
1. Load and filter Games.csv
2. Compute team-level features
3. Split data into train/val/test sets
4. Preprocess features (one-hot encoding, imputation, scaling)
5. Train a simple MLP model
6. Evaluate and save results

Expected runtime: ~5-10 minutes on a standard laptop.

### Individual Steps

You can also run individual steps:

```bash
# Step 1: Load and filter data
python scripts/load_data.py

# Step 2: Engineer features
python scripts/feature_engineering.py

# Step 3: Preprocess data
python scripts/preprocessing.py

# Step 4: Train model
python scripts/train_model.py
```

## Output Files

After running the pipeline, you'll find:

- **models/stage_a1_mlp.keras** - Trained neural network model
- **outputs/results.json** - Performance metrics (accuracy, AUC, confusion matrix)
- **outputs/filtered_games.csv** - Filtered and cleaned games
- **outputs/features_engineered.csv** - Games with computed features
- **outputs/*.npy** - Preprocessed train/val/test arrays
- **outputs/preprocessing_objects.pkl** - Scalers and encoders

## Expected Results

The Stage A1 model achieves:
- **Test Accuracy**: ~57-60%
- **Test AUC**: ~0.58-0.61
- **Validation Accuracy**: ~60-63%

This is better than random guessing (50%) and demonstrates the model can capture patterns like home court advantage and team strength.

## Detailed Documentation

For detailed information about the project structure and how each component works, see:
- [Repository Structure Documentation](docs/repo_structure.md)
- [Main README](README.md) - Full Stage A1 specification

## Troubleshooting

**Issue**: Missing Games.csv
- **Solution**: Ensure `data/Games.csv` exists with the correct structure

**Issue**: Out of memory during feature engineering
- **Solution**: The script processes 40k+ games. Use a machine with at least 8GB RAM.

**Issue**: TensorFlow not found
- **Solution**: Install TensorFlow: `pip install tensorflow>=2.13.0`

## Next Steps

After successfully running Stage A1:
- Review the results in `outputs/results.json`
- Examine the model architecture in `scripts/train_model.py`
- Consider implementing Stage B (adding TeamStatistics.csv)
