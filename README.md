# NBA_Games_Prediction_ML_NN
This is a Machine Learning Model using a Neural Network to predict NBA Games outcomes based on historical data

## 🚀 Quick Start

**Current Status**: Stage A1 is complete and optimized. Stage B1 is ready for team statistics integration.

```bash
# Run Stage A1 (historical features only)
python run_stage_a1.py

# Run Stage B1 (team statistics - coming soon)
python run_stage_b1.py

# Analyze results and compare runs
python analyze_results.py
```

**Documentation**:
- [Stage A1 Analysis & Results](docs/stage_a1_analysis.md)
- [Repository Structure](docs/repo_structure.md)
- [Model Improvements Guide](docs/model_improvements.md)
- [Archiving System](docs/archiving_system.md)

---

# NBA Game Outcome Prediction with Neural Networks

> IMPORTANT NOTE  
> In addition to this README, the repository includes comprehensive documentation  
> in the `docs/` folder that explains each component and how they fit together.

This project is a learning-oriented machine learning pipeline to predict NBA game outcomes
using neural networks based on historical data.

The work is structured in **stages** along two axes:

1. **Data complexity**  
   - **Stage A**: `Games.csv` only ✅ **COMPLETE**
   - **Stage B**: `Games.csv` + `TeamStatistics.csv` 🔄 **IN PROGRESS**
   - **Stage C**: `Games.csv` + `TeamStatistics.csv` + `PlayerStatistics.csv` 📋 **PLANNED**

2. **Model complexity**  
   - **Model 1**: Simple MLP (baseline) ✅ **COMPLETE**
   - **Model 2**: MLP with embeddings and regularization 📋 **PLANNED**
   - **Model 3**: Sequence-based model (LSTM/GRU/CNN 1D) 📋 **PLANNED**

## Current Implementation

### Stage A1 ✅ (Complete)
**Data Stage A (Games only) × Model 1 (Simple MLP with regularization)**

Files:
- `run_stage_a1.py` - Main pipeline for Stage A1
- Results in `outputs/stage_a1/`
- Archives in `archives/stage_a1/`

**Performance** (Run 2 - optimized):
- Test Accuracy: **58.55%**
- Test AUC: **0.5984**
- Test F1-Score: **0.7098**

See [Stage A1 Analysis](docs/stage_a1_analysis.md) for detailed comparison and improvements.

### Stage B1 ✅ (Complete)
**Data Stage B (Games + Team Statistics) × Model 1 (Simple MLP)**

Files:
- `run_stage_b1.py` - Main pipeline for Stage B1
- `features/stage_b1_teamstats.py` - Team statistics feature engineering
- Results in `outputs/stage_b1/`
- Archives in `archives/stage_b1/`

**Stage B1 extends Stage A1 by adding:**
- **Rolling team statistics** (last 5, 10 games):
  - Scoring: points for/against, point differential
  - Shooting: FG%, 3P%, FT%
  - Advanced: offensive/defensive/net rating, pace
  - Rebounds, assists, turnovers, steals, blocks
- **Home/away split features**: Performance in home vs away games
- **Delta features**: Direct matchup comparisons (home - away)
- **NO DATA LEAKAGE**: Rolling features computed only from past games

**Expected Performance Target**: Test AUC 0.62–0.65 (vs 0.598 in Stage A1)

See [Stage B1 Design Document](docs/stage_b1_design.md) for complete feature specifications.

**Run Stage B1:**
```bash
python run_stage_b1.py

# Validate implementation
python test_stage_b1.py
```

---

## Stage A1 Details

The goal of Stage A1 is:
- to build a clean and understandable pipeline end-to-end,
- to use only `Games.csv` (plus simple engineered features),
- to train a basic neural network and evaluate it realistically over time.

---

## 1. Data Scope for Stage A1

### 1.1 Source file

- `Games.csv`  
  Contains all historical NBA games (home team, away team, scores, date, etc.).

### 1.2 Temporal scope

For Stage A1, we use:

- Only games from **season 1990–1991 onwards**.
- Only **Regular Season** games.

Concretely:

1. Parse the game date (`gameDateTimeEst`) to extract:
   - year
   - month
   - day

2. Define a `seasonYear` as:
   - If `month >= 8` (August–December): `seasonYear = year`
   - If `month < 8` (January–July): `seasonYear = year - 1`

Examples:
- 2023-11-10 → season 2023–24 → `seasonYear = 2023`
- 2024-03-05 → still season 2023–24 → `seasonYear = 2023`

3. Keep only rows where:
   - `seasonYear >= 1990`
   - `gameType == "Regular Season"`
   - `homeScore` and `awayScore` are not null.

Preseason, playoffs, play-in tournaments, etc. are excluded for Stage A1.

---

## 2. Target Variable (Label)

We predict whether the **home team wins** the game.

- Let `homeScore` be the final score for the home team.
- Let `awayScore` be the final score for the away team.

Define:

- `y = 1` if `homeScore > awayScore`
- `y = 0` if `homeScore < awayScore`

NBA games cannot end in a tie, so this is a clean binary classification problem.

---

## 3. Feature Set for Stage A1

This section defines the **exact features** to compute for Stage A1:
name, type, description, and calculation logic.

All features are computed **at the moment just before each game**,
using only past games of the same team (no data leakage from future games).

### 3.1 Helper concepts

For each team and each `seasonYear`:

- Sort that team’s games chronologically by `gameDateTimeEst`.
- For a game at time `t`, “previous games” are all games of that team in the same `seasonYear`
  with a date strictly earlier than `t`.

When we say “last 5 games”, we refer to up to 5 such previous games in the current season.

If a team has fewer than 5 previous games, the “last 5” aggregation is computed
on however many games exist (1–4). If there are **no** previous games, we define:
- win percentages as `0.5` (neutral prior),
- average point differential as `0.0`,
- number of games in the window as `0`,
- days since last game as `NaN` (to be imputed in preprocessing).

All of this logic should be implemented in a reproducible way.

### 3.2 Feature list (numeric features)

Below is the exact list of **numeric features** for Stage A1.

#### 1. `seasonYear`
- **Type**: int (later treated as numeric feature)
- **Description**: Season starting year for this game (e.g. 2023 for season 2023–24).
- **Calculation**: As defined above:
  - `year = year(gameDateTimeEst)`
  - `month = month(gameDateTimeEst)`
  - If `month >= 8`: `seasonYear = year`, else `seasonYear = year - 1`.

#### 2. `home_season_games_played`
- **Type**: int
- **Description**: Number of games the home team has already played in the **current season** before this game.
- **Calculation**:
  - Count of previous games for home team with the same `seasonYear`.

#### 3. `home_season_wins`
- **Type**: int
- **Description**: Number of games won by the home team in the current season before this game.
- **Calculation**:
  - Among previous games in the same `seasonYear`, count where home team’s final score > opponent’s final score (from the team’s perspective).

#### 4. `home_season_losses`
- **Type**: int
- **Description**: Number of games lost by the home team in the current season before this game.
- **Calculation**:
  - `home_season_losses = home_season_games_played - home_season_wins`.

#### 5. `home_season_win_pct`
- **Type**: float
- **Description**: Win percentage of the home team in the current season before this game.
- **Calculation**:
  - If `home_season_games_played > 0`:
    - `home_season_win_pct = home_season_wins / home_season_games_played`
  - Else:
    - `home_season_win_pct = 0.5`.

#### 6. `away_season_games_played`
- **Type**: int
- **Description**: Number of games the away team has already played in the current season before this game.
- **Calculation**:
  - Count of previous games for away team with the same `seasonYear`.

#### 7. `away_season_wins`
- **Type**: int
- **Description**: Number of games won by the away team in the current season before this game.
- **Calculation**:
  - Among previous games in the same `seasonYear`, count where away team’s final score (as that team) is higher than the opponent’s.

#### 8. `away_season_losses`
- **Type**: int
- **Description**: Number of games lost by the away team in the current season before this game.
- **Calculation**:
  - `away_season_losses = away_season_games_played - away_season_wins`.

#### 9. `away_season_win_pct`
- **Type**: float
- **Description**: Win percentage of the away team in the current season before this game.
- **Calculation**:
  - If `away_season_games_played > 0`:
    - `away_season_win_pct = away_season_wins / away_season_games_played`
  - Else:
    - `away_season_win_pct = 0.5`.

#### 10. `home_last5_games_played`
- **Type**: int
- **Description**: Number of games considered in the “last 5” window for the home team.
- **Calculation**:
  - Count of up to 5 most recent previous games in the current season.

#### 11. `home_last5_win_pct`
- **Type**: float
- **Description**: Win percentage of the home team over its last 5 games (or fewer if less than 5 played).
- **Calculation**:
  - If `home_last5_games_played > 0`:
    - `home_last5_win_pct = (wins in last5) / home_last5_games_played`
  - Else:
    - `home_last5_win_pct = 0.5`.

#### 12. `home_last5_avg_point_diff`
- **Type**: float
- **Description**: Average point differential for the home team over its last 5 games.
- **Calculation**:
  - For each of the last up to 5 previous games:
    - `point_diff = (team_points_scored) - (team_points_allowed)`
  - `home_last5_avg_point_diff = mean(point_diff)` over these games.
  - If no previous games: `home_last5_avg_point_diff = 0.0`.

#### 13. `away_last5_games_played`
- **Type**: int
- **Description**: Number of games considered in the “last 5” window for the away team.
- **Calculation**:
  - Count of up to 5 most recent previous games in the current season.

#### 14. `away_last5_win_pct`
- **Type**: float
- **Description**: Win percentage of the away team over its last 5 games (or fewer if less than 5 played).
- **Calculation**:
  - If `away_last5_games_played > 0`:
    - `away_last5_win_pct = (wins in last5) / away_last5_games_played`
  - Else:
    - `away_last5_win_pct = 0.5`.

#### 15. `away_last5_avg_point_diff`
- **Type**: float
- **Description**: Average point differential for the away team over its last 5 games.
- **Calculation**:
  - Same as home, but from the away team’s perspective.
  - If no previous games: `away_last5_avg_point_diff = 0.0`.

#### 16. `home_days_since_last_game`
- **Type**: float
- **Description**: Number of days since the home team’s previous game in the same season.
- **Calculation**:
  - If the team has a previous game:
    - `home_days_since_last_game = (current_game_date - previous_game_date).days`
  - Else:
    - `home_days_since_last_game = NaN` (to be imputed later).

#### 17. `away_days_since_last_game`
- **Type**: float
- **Description**: Number of days since the away team’s previous game in the same season.
- **Calculation**:
  - Same logic as for the home team.

#### 18. `delta_season_win_pct`
- **Type**: float
- **Description**: Difference in season win percentage between home and away teams before the game.
- **Calculation**:
  - `delta_season_win_pct = home_season_win_pct - away_season_win_pct`.

#### 19. `delta_last5_win_pct`
- **Type**: float
- **Description**: Difference in last5 win percentage between home and away teams.
- **Calculation**:
  - `delta_last5_win_pct = home_last5_win_pct - away_last5_win_pct`.

#### 20. `delta_last5_point_diff`
- **Type**: float
- **Description**: Difference in last5 average point differential between home and away teams.
- **Calculation**:
  - `delta_last5_point_diff = home_last5_avg_point_diff - away_last5_avg_point_diff`.

#### 21. `delta_days_rest`
- **Type**: float
- **Description**: Difference in rest days between home and away teams before the game.
- **Calculation**:
  - `delta_days_rest = home_days_since_last_game - away_days_since_last_game`.

### 3.3 Categorical features (teams)

We also use the identity of the teams as categorical features:

#### 22. `hometeamId`
- **Type**: categorical (team ID)
- **Description**: Identifier of the home team.
- **Usage in Stage A1**:
  - One-hot encode this column into multiple binary columns (e.g. `home_team_LAL`, `home_team_BOS`, etc.).

#### 23. `awayteamId`
- **Type**: categorical (team ID)
- **Description**: Identifier of the away team.
- **Usage in Stage A1**:
  - One-hot encode this column into multiple binary columns (e.g. `away_team_LAL`, `away_team_BOS`, etc.).

These one-hot features are concatenated with the numeric features to form the model input.

---

## 4. Train/Validation/Test Split (Time-Based)

We split the data **by seasonYear** to respect the time dimension:

- **Train set**:
  - All Regular Season games with `seasonYear` from **1990** to **2015** (inclusive).

- **Validation set**:
  - All Regular Season games with `seasonYear` from **2016** to **2019** (inclusive).

- **Test set**:
  - All Regular Season games with `seasonYear` from **2020** to **2023** (inclusive).

Future use (not implemented in Stage A1, but important for the project):

- **2024–25**: “live simulation” predictions (once the model is trained and tested).
- **2025–26**: main prediction target season, using `LeagueSchedule25_26.csv` (later stages).

---

## 5. Stage A1 Pipeline Specification

This section describes the **end-to-end pipeline** that should be implemented for Stage A1.

The goal is to have this pipeline reproducible in code (Python, with typical ML tools)
and easy to understand.

### 5.1 High-level steps

1. **Load and filter raw data**
2. **Compute seasonYear and filter by season**
3. **Compute team-level historical statistics**
4. **Build match-level feature table**
5. **Split into train/validation/test**
6. **Preprocess features**
7. **Define and train a simple MLP model**
8. **Evaluate and save results**

### 5.2 Detailed pipeline

#### Step 1 – Load and basic filtering

- Load `Games.csv` into a DataFrame.
- Parse `gameDateTimeEst` as a datetime column.
- Drop rows where `homeScore` or `awayScore` is missing.
- Drop rows where `gameType` is not `"Regular Season"` (or where `gameType` is null).

#### Step 2 – Compute `seasonYear` and filter by period

- Compute `seasonYear` for each game as defined in section 2.
- Keep only games with `seasonYear >= 1990`.

#### Step 3 – Compute team-level historical stats

For each team and each `seasonYear`:

- Sort that team’s games by date ascending.
- For each game in chronological order:
  - Look at all previous games in the same `seasonYear`.
  - Compute:
    - season games played / wins / losses / win_pct.
    - last5 games played / last5 win_pct / last5 avg point diff.
    - days since last game.
- Store these values back into the main game-level dataset for:
  - the home team (prefixed with `home_`)
  - the away team (prefixed with `away_`).
- After computing home and away features, compute the `delta_*` features
  (difference home minus away).

Important: for each game, features must only use information from games strictly before that game
(no information from the current or future games).

#### Step 4 – Build match-level feature table

- Keep only the columns needed:
  - Target: `y` (home win indicator).
  - Numeric features listed in section 3.2.
  - Categorical team IDs: `hometeamId`, `awayteamId`.
  - `seasonYear` (for splitting).
- Drop games where critical features are missing and cannot be reasonably imputed.

At this stage, we should have a clean DataFrame with one row per game and all features defined.

#### Step 5 – Train/Validation/Test split

- Based on `seasonYear`:
  - Train: `seasonYear` in [1990, 2015]
  - Validation: `seasonYear` in [2016, 2019]
  - Test: `seasonYear` in [2020, 2023]
- Create:
  - `X_train`, `y_train`
  - `X_val`, `y_val`
  - `X_test`, `y_test`

#### Step 6 – Preprocessing

- One-hot encode `hometeamId` and `awayteamId`.
- Concatenate one-hot features with numeric features.
- Handle missing values:
  - For numeric features (e.g., `home_days_since_last_game`, `away_days_since_last_game`),
    impute using a simple strategy (e.g., median of the training set).
- Scale numeric features (e.g., StandardScaler or MinMaxScaler) based only on the training set.
- Apply the same transformations to validation and test sets.

The final input shape for the MLP should be:
- `(n_samples, n_features)` where `n_features = number_of_numeric_features + number_of_one_hot_features`.

#### Step 7 – MLP Model (Stage A1)

Define a **simple feed-forward neural network** architecture, for example:

- Input layer: size = `n_features`
- Hidden layer 1: Dense(64, activation="relu")
- Hidden layer 2: Dense(32, activation="relu")
- Output layer: Dense(1, activation="sigmoid")

Training setup:

- Loss: Binary Cross-Entropy
- Optimizer: e.g., Adam
- Metrics: accuracy, log loss; optionally AUC/Brier score
- Use early stopping based on validation loss to prevent overfitting.

#### Step 8 – Evaluation and outputs

- Evaluate the trained model on:
  - Validation set: used for model tuning.
  - Test set: used for final Stage A1 performance.
- Report at least:
  - Accuracy
  - Log loss
  - Confusion matrix (optional but helpful)
- Save:
  - Trained model weights (for later reuse)
  - Preprocessing objects (encoders, scalers, etc.)
  - Evaluation metrics (e.g., as JSON or in a results file).

---

## 6. Next Steps (Beyond Stage A1)

After Stage A1 is implemented and understood:

- **Stage B1/B2**:
  - Enrich input with `TeamStatistics.csv`
  - Possibly add embeddings for teams instead of one-hot

- **Stage C1–C3**:
  - Aggregate `PlayerStatistics.csv` at team level for more advanced features
  - Experiment with sequence-based models (LSTM/GRU/CNN 1D) over recent games

These stages are deliberately **out of scope for this README**, which focuses on
a clean, understandable, and working Stage A1.

---

## 6. Running the Pipeline

### Stage A1 (Historical Features Only)

```bash
# Run the complete Stage A1 pipeline
python run_stage_a1.py
```

**What it does**:
1. Loads and filters Games.csv
2. Engineers historical features (win rates, last 5 games, rest days)
3. Preprocesses data (one-hot encoding, scaling)
4. Trains MLP with L2 regularization and Dropout
5. Optimizes decision threshold on validation set
6. Generates comprehensive visualizations
7. Archives results for future comparison

**Outputs**:
- Model: `models/stage_a1/mlp.keras`
- Results: `outputs/stage_a1/results.json`
- Plots: `outputs/stage_a1/plots/`
- Archive: `archives/stage_a1/run_YYYYMMDD_HHMMSS/`

### Stage B1 (Team Statistics) - Coming Soon

```bash
# Run the complete Stage B1 pipeline
python run_stage_b1.py
```

**Status**: Pipeline structure is ready. Currently uses Stage A1 features.  
**Next step**: Integrate TeamStatistics.csv for offensive/defensive ratings, shooting efficiency, etc.

**Outputs** (when implemented):
- Model: `models/stage_b1/mlp.keras`
- Results: `outputs/stage_b1/results.json`
- Plots: `outputs/stage_b1/plots/`
- Archive: `archives/stage_b1/run_YYYYMMDD_HHMMSS/`

### Analyzing Results

```bash
# Analyze current results and compare with previous runs
python analyze_results.py

# List all archived runs
python scripts/archive_manager.py --list

# Compare specific runs
python scripts/archive_manager.py --compare archives/stage_a1/run_20251209_193450
```

### File Organization

```
NBA_Games_Predictions_ML_NN/
├── run_stage_a1.py          # Stage A1 pipeline (complete)
├── run_stage_b1.py          # Stage B1 pipeline (structure ready)
├── run_pipeline.py          # Deprecated - redirects to Stage A1
├── analyze_results.py       # Results analysis and comparison
├── outputs/
│   ├── stage_a1/           # Stage A1 results
│   │   ├── results.json
│   │   ├── plots/
│   │   ├── X_train.npy, y_train.npy, etc.
│   │   └── preprocessing_objects.pkl
│   └── stage_b1/           # Stage B1 results (when implemented)
├── archives/
│   ├── stage_a1/           # Stage A1 archived runs
│   │   ├── run_20251209_191510/
│   │   └── run_20251209_193450/
│   └── stage_b1/           # Stage B1 archived runs
├── models/
│   ├── stage_a1/           # Stage A1 models
│   │   └── mlp.keras
│   └── stage_b1/           # Stage B1 models
└── docs/
    ├── stage_a1_analysis.md    # Complete Stage A1 analysis
    ├── model_improvements.md    # Improvement documentation
    ├── archiving_system.md      # Archive system guide
    └── repo_structure.md        # Code structure
```

---

## 7. Next Steps (Beyond Stage A1)

After Stage A1 is implemented and understood:

- **Stage B1**:
  - ✅ Pipeline structure created
  - 🔄 Integrate `TeamStatistics.csv`
  - Add offensive/defensive ratings, shooting efficiency, pace
  - Add rolling averages for team stats
  - Expected improvement: Test AUC 0.598 → 0.62–0.65

- **Stage B2** (optional):
  - Replace one-hot encoding with team embeddings
  - Learn team representations

- **Stage C1–C3**:
  - Aggregate `PlayerStatistics.csv` at team level
  - Injury reports, player availability
  - Experiment with sequence-based models (LSTM/GRU/CNN 1D) over recent games

---
