# Repository Structure Documentation

This document explains the structure of the NBA Game Prediction project and how each file/script/notebook fits together to build and run the Stage A1 model.

## Project Overview

This repository implements Stage A1 of an NBA game prediction system using a neural network. Stage A1 uses only `Games.csv` data with a simple MLP (Multi-Layer Perceptron) model to predict whether the home team will win a game.

## Directory Structure

```
NBA_Games_Prediction_ML_NN/
├── data/                       # Raw data files
│   ├── Games.csv              # Main dataset: historical NBA games
│   ├── TeamStatistics.csv     # Team statistics (for future stages)
│   ├── Players.csv            # Player information (for future stages)
│   ├── TeamHistories.csv      # Team historical data
│   ├── LeagueSchedule24_25.csv # Schedule for 2024-25 season
│   └── LeagueSchedule25_26.csv # Schedule for 2025-26 season
│
├── scripts/                    # Python scripts for each pipeline step
│   ├── load_data.py           # Step 1: Load and filter raw data
│   ├── feature_engineering.py # Step 2: Compute team features
│   ├── preprocessing.py       # Step 3: Split, encode, impute, scale
│   └── train_model.py         # Step 4: Define, train, and evaluate model
│
├── models/                     # Saved trained models (created during execution)
│   └── stage_a1_mlp.keras     # Best MLP model from training
│
├── outputs/                    # Intermediate and final outputs (created during execution)
│   ├── filtered_games.csv     # Filtered games (Regular Season 1990+)
│   ├── features_engineered.csv # Games with all computed features
│   ├── X_train.npy            # Training features (preprocessed)
│   ├── y_train.npy            # Training labels
│   ├── X_val.npy              # Validation features (preprocessed)
│   ├── y_val.npy              # Validation labels
│   ├── X_test.npy             # Test features (preprocessed)
│   ├── y_test.npy             # Test labels
│   ├── preprocessing_objects.pkl # Scaler, imputer, feature names
│   └── results.json           # Training metrics and results
│
├── docs/                       # Documentation
│   └── repo_structure.md      # This file
│
├── run_pipeline.py             # Main script: runs complete pipeline
├── requirements.txt            # Python dependencies
├── README.md                   # Project specification and guidelines
└── .gitignore                  # Git ignore patterns
```

## Data Flow and Pipeline

The complete pipeline follows these steps:

### Step 1: Data Loading and Filtering (`scripts/load_data.py`)

**Purpose**: Load raw games data and perform basic filtering.

**Input**: `data/Games.csv`

**Output**: `outputs/filtered_games.csv`

**What it does**:
- Loads the Games.csv file
- Parses game dates
- Filters for Regular Season games only
- Computes `seasonYear` (season starting year)
- Filters for games from season 1990-91 onwards
- Creates target variable `y` (1 if home team wins, 0 otherwise)
- Drops games with missing scores

**Key logic**:
- `seasonYear = year` if month >= 8 (August-December)
- `seasonYear = year - 1` if month < 8 (January-July)

### Step 2: Feature Engineering (`scripts/feature_engineering.py`)

**Purpose**: Compute historical statistics for each team at each game.

**Input**: `outputs/filtered_games.csv`

**Output**: `outputs/features_engineered.csv`

**What it does**:
- For each team and season, processes games chronologically
- Computes team-level features based only on previous games (no data leakage):
  - **Season stats**: games played, wins, losses, win percentage
  - **Last 5 games**: games played, win percentage, average point differential
  - **Rest**: days since last game
- Computes delta features (home - away differences)

**Features computed** (21 numeric features total):
1. `seasonYear`
2-8. Home team season stats (games played, wins, losses, win %, etc.)
9-15. Away team season stats
16-17. Days since last game (home and away)
18-21. Delta features (differences between home and away)

Plus 2 categorical features (team IDs) that will be one-hot encoded.

### Step 3: Preprocessing (`scripts/preprocessing.py`)

**Purpose**: Prepare data for model training.

**Input**: `outputs/features_engineered.csv`

**Outputs**: 
- `outputs/X_train.npy`, `outputs/y_train.npy`
- `outputs/X_val.npy`, `outputs/y_val.npy`
- `outputs/X_test.npy`, `outputs/y_test.npy`
- `outputs/preprocessing_objects.pkl`

**What it does**:
1. **Time-based split**:
   - Train: seasons 1990-2015
   - Validation: seasons 2016-2019
   - Test: seasons 2020-2023

2. **One-hot encoding**:
   - Creates binary features for each team (e.g., `home_team_LAL`, `away_team_LAL`)
   - All unique teams across train/val/test are encoded

3. **Missing value imputation**:
   - Uses median strategy on training set
   - Applies same imputation to val/test

4. **Feature scaling**:
   - StandardScaler on numeric features (fit on training set)
   - Applies same scaling to val/test
   - One-hot features are not scaled

5. **Saves preprocessing objects** for future use (inference on new data)

### Step 4: Model Training (`scripts/train_model.py`)

**Purpose**: Define, train, and evaluate the neural network model.

**Input**: Preprocessed numpy arrays from Step 3

**Outputs**:
- `models/stage_a1_mlp.keras` (best model)
- `outputs/results.json` (metrics and training history)

**What it does**:
1. **Model architecture** (Simple MLP):
   ```
   Input → Dense(64, relu) → Dense(32, relu) → Dense(1, sigmoid)
   ```

2. **Training setup**:
   - Loss: Binary Cross-Entropy
   - Optimizer: Adam
   - Metrics: Accuracy, AUC
   - Early stopping on validation loss (patience=10)
   - Saves best model based on validation loss

3. **Evaluation**:
   - Computes metrics on train, validation, and test sets
   - Reports: accuracy, AUC, log loss, confusion matrix
   - Saves all results to JSON

## How to Run the Pipeline

### Prerequisites

1. Install Python 3.8+ and dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Ensure `data/Games.csv` is present in the data directory.

### Running the Complete Pipeline

Execute the main pipeline script:

```bash
python run_pipeline.py
```

This will:
1. Load and filter data
2. Engineer features
3. Preprocess and split data
4. Train the MLP model
5. Evaluate and save results

The entire pipeline takes approximately 10-30 minutes depending on hardware.

### Running Individual Steps

You can also run individual scripts:

```bash
# Step 1: Load data
python scripts/load_data.py

# Step 2: Engineer features
python scripts/feature_engineering.py

# Step 3: Preprocess
python scripts/preprocessing.py

# Step 4: Train model
python scripts/train_model.py
```

Each script expects the outputs from previous steps to be in the `outputs/` directory.

## Expected Results

After running the pipeline, you should have:

- **Model file**: `models/stage_a1_mlp.keras` (trained neural network)
- **Results file**: `outputs/results.json` (performance metrics)
- **Preprocessed data**: numpy arrays in `outputs/` directory
- **Intermediate CSVs**: filtered data and engineered features

Typical performance for Stage A1:
- Test Accuracy: ~60-65% (better than random guessing at 50%)
- AUC: ~0.60-0.65
- The model captures basic patterns like home court advantage and team strength

## Future Stages (Not Yet Implemented)

The README outlines future development stages:

- **Stage B**: Incorporate `TeamStatistics.csv` for richer features
- **Stage C**: Add `PlayerStatistics.csv` aggregated at team level
- **Advanced Models**: LSTM/GRU for sequence modeling, embeddings for teams

These stages will follow the same structure with additional scripts in the `scripts/` directory.

## Key Design Principles

1. **No data leakage**: All features computed using only past games
2. **Time-based splits**: Train/val/test separated by season years
3. **Reproducibility**: All random seeds set, preprocessing objects saved
4. **Modularity**: Each step is a separate script that can be run independently
5. **Clear documentation**: Code comments and this structure document

## Troubleshooting

**Issue**: Missing data files
- **Solution**: Ensure `data/Games.csv` exists with the correct structure

**Issue**: Out of memory
- **Solution**: The feature engineering step can be memory-intensive. Consider processing smaller batches or using a machine with more RAM.

**Issue**: Model not training
- **Solution**: Check that TensorFlow is properly installed and GPU drivers (if using GPU)

## Contact and Contribution

For questions or improvements, please refer to the main README.md for project guidelines and contribution instructions.
