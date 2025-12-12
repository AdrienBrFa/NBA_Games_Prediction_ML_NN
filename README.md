# NBA Game Prediction with Neural Networks

**A pedagogical machine learning project for learning neural networks through NBA game outcome prediction.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/tensorflow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Data Sources](#-data-sources)
- [Pipeline Stages](#-pipeline-stages)
- [Results Summary](#-results-summary)
- [Quick Start](#-quick-start)
- [Detailed Usage](#-detailed-usage)
- [Reproducibility & Archiving](#-reproducibility--archiving)
- [Project Structure](#-project-structure)
- [Roadmap](#-roadmap)
- [Documentation](#-documentation)

---

## 🎯 Project Overview

### Goal
Learn neural network fundamentals through a **real-world binary classification problem**: predicting NBA game outcomes (Home Win vs Away Win).

### Key Characteristics

**Binary Classification**
- Target: `1` if home team wins, `0` if away team wins
- No draws in NBA basketball (overtime until winner)

**Time-Based Evaluation**
- Training: 1990-2015 seasons
- Validation: 2016-2020 seasons  
- Test: 2021-2024 seasons
- **No random shuffling** - prevents data leakage and simulates real-world deployment

**Pedagogical Approach**
- Start simple (historical features only)
- Progressively add complexity (team stats, player stats)
- Systematic experimentation and documentation
- Focus on understanding over performance maximization

### Why NBA Games?

✅ **Rich dataset** - decades of historical data  
✅ **Binary outcome** - clean classification problem  
✅ **Domain knowledge accessible** - basketball statistics are well-documented  
✅ **Inherently unpredictable** - ~60-70% accuracy ceiling keeps expectations realistic  
✅ **Temporal structure** - teaches proper time-series handling

---

## 📊 Data Sources

### Primary Datasets

#### `Games.csv` (72,307 games)
Core game information from 1946 to 2024.

**Key columns**:
- `gameId` - unique identifier
- `gameDateTimeEst` - game timestamp (for temporal ordering)
- `seasonYear` - NBA season (e.g., 2023 = 2023-24 season)
- `hometeamId`, `awayteamId` - team identifiers
- `homeScore`, `awayScore` - final scores
- Derived: `winner` (home=1, away=0)

**Preprocessing**:
- Filter to regular season games only (no playoffs/preseason)
- Filter to seasons ≥ 1990 (modern NBA era with 3-point line)
- Final dataset: **40,970 games**

#### `TeamStatistics.csv` (144,614 rows)
Detailed team performance statistics for each game.

**Key statistics** (48 columns per team per game):
- Basic: points, field goals, 3-pointers, free throws, rebounds, assists, turnovers, steals, blocks
- Advanced: offensive rating, defensive rating, net rating, possessions (pace)
- Each game has 2 rows: one for home team, one for away team

**Usage**: Used in Stage B1 to compute rolling averages and team performance trends.

#### `PlayerStatistics.csv` ⚠️ NOT USED YET
Individual player statistics (points, assists, etc.).

**Status**: Available but postponed to Stage C due to:
- Dataset size (millions of rows)
- Complexity of player-level aggregation
- Need to establish baseline performance first

---

## 🏗️ Pipeline Stages

The project is organized along **two axes**:

### Data Complexity Axis

| Stage | Data Sources | Status | Description |
|-------|-------------|--------|-------------|
| **A** | Games.csv only | ✅ Complete | Historical game outcomes and scheduling |
| **B** | + TeamStatistics.csv | ✅ Complete | Team performance metrics and trends |
| **C** | + PlayerStatistics.csv | 📋 Planned | Individual player contributions |

### Model Complexity Axis

| Model | Architecture | Status | Description |
|-------|-------------|--------|-------------|
| **1** | Simple MLP | ✅ Complete | Baseline with regularization |
| **2** | Enhanced MLP | 📋 Planned | Embeddings, deeper architecture |
| **3** | Sequence Model | 📋 Future | LSTM/GRU/CNN-1D for temporal patterns |

### Current Implementation

Each combination of (Data Stage × Model) is a distinct experiment:

- **Stage A1**: Games only × Simple MLP ✅ **COMPLETE**
- **Stage B1 Intermediate**: Games + Team Stats (10 metrics, seasonal reset) × Simple MLP ✅ **COMPLETE**
- **Stage B1 Full**: Games + Team Stats (17 metrics, continuous) × Simple MLP ✅ **COMPLETE**

---

## 📈 Results Summary

### Performance Comparison

All models use **optimal threshold** tuned on validation set (F1 score maximization).

| Stage | Features | Rolling Windows | Season Reset | Test Accuracy | Test AUC | Test F1 | Improvement |
|-------|----------|----------------|--------------|---------------|----------|---------|-------------|
| **A1** | 20 | last 5 games | Yes | 58.6% | 0.598 | 0.710 | Baseline |
| **B1 Intermediate** | 145 | last 5, 10 | Yes | 62.1% | 0.658 | 0.718 | +6.0% AUC |
| **B1 Full** | 203 | last 5, 10 | No | 63.0% | 0.674 | 0.701 | +7.6% AUC |

### Key Insights

#### 1. Stage B1 vs Stage A1 (+6-7.6% AUC)

**Why B1 improves over A1:**
- **Richer features**: Offensive/defensive ratings capture team quality better than win-loss record
- **Rolling statistics**: Recent form (last 5-10 games) predicts better than season-long averages
- **Advanced metrics**: Pace, shooting efficiency, net rating provide deeper insights
- **Home/away splits**: Teams perform differently at home vs on the road

**Trade-off**: More features require more data and increase overfitting risk (mitigated by regularization).

#### 2. B1 Intermediate vs B1 Full

**B1 Intermediate** (Recommended):
- **97.7% of Full's AUC** with **28.6% fewer features** (145 vs 203)
- **Higher F1 score** (0.718 vs 0.701) - better precision/recall balance
- **Seasonal reset** aligns with NBA reality (teams change between seasons)
- **Faster inference** and easier interpretation

**B1 Full** (Benchmark):
- **Best AUC** (0.674) - maximum predictive power
- **Comprehensive features** - all available team statistics
- **Continuous rolling** - carries momentum across season boundaries
- Kept as reference for feature importance analysis

**Recommendation**: Use **B1 Intermediate** for production due to better robustness, efficiency, and F1 score.

#### 3. Performance Ceiling

NBA game prediction is **inherently uncertain** (~60-70% accuracy ceiling):
- Injuries, player rest, coaching decisions
- Referee calls, momentum shifts, clutch performance
- Travel fatigue, schedule compression

Our results (62-63% test accuracy) are **realistic and on par with domain expectations**.

---

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.10 or higher
python --version

# Install dependencies
pip install -r requirements.txt
```

### Run Pipelines

```bash
# Stage A1 - Baseline (Games only)
python run_stage_a1.py

# Stage B1 Intermediate - Recommended (Team stats, 10 metrics, seasonal reset)
python run_stage_b1_intermediate.py

# Stage B1 Full - Benchmark (Team stats, 17 metrics, continuous)
python run_stage_b1_full.py
```

### Threshold Optimization Strategies

All pipelines support multiple threshold optimization strategies:

```bash
# Default: F1-optimized threshold (balances precision and recall)
python run_stage_a1.py

# Accuracy-optimized threshold (maximizes overall correctness)
python run_stage_a1.py --threshold_metric accuracy

# Balanced accuracy-optimized threshold (handles class imbalance)
python run_stage_a1.py --threshold_metric balanced_accuracy

# Compare different strategies
python compare_thresholds.py
```

**See [Threshold Optimization Guide](docs/threshold_optimization.md) for details.**

### Pipeline Workflow

Each pipeline:
1. Loads and filters data
2. Engineers features
3. Trains MLP model with regularization
4. Evaluates on test set with optimal threshold
5. Generates visualizations
6. Archives results for comparison

**Runtime**: ~5-15 minutes per pipeline (depending on hardware)

**Output locations**:
- Models: `models/stage_xxx/mlp.keras`
- Results: `outputs/stage_xxx/results.json`
- Plots: `outputs/stage_xxx/plots/`
- Archives: `archives/stage_xxx/run_YYYYMMDD_HHMMSS_threshold-{metric}/`

---

## 📖 Detailed Usage

### Stage A1 - Historical Features Only

**What it does**:
- Computes team historical performance (win %, win streak, rest days)
- Uses last 5 games rolling window (reset at season boundaries)
- Simple MLP: 80-40-20-1 with L2 regularization, dropout, early stopping

**Features** (20 total):
- **Home team**: win%, last5_win%, win_streak, days_rest, home_win%, home_streak
- **Away team**: win%, last5_win%, win_streak, days_rest, away_win%, away_streak  
- **Deltas**: win%_delta, last5_win%_delta, streak_delta, rest_delta, home_advantage_delta
- **Context**: is_back_to_back_home, is_back_to_back_away

**Results**: Test AUC 0.598, Accuracy 58.6%

**Run**:
```bash
python run_stage_a1.py
```

See [Stage A1 Analysis](docs/stage_a1_analysis.md) for detailed results and improvement history.

---

### Stage B1 Intermediate - Team Statistics (Recommended)

**What it adds to A1**:
- **10 key team metrics** from TeamStatistics.csv:
  - Offensive/Defensive/Net Rating
  - Pace (possessions per game)
  - Shooting: FG%, 3P%, FT%
  - Rebounds (total), Assists, Turnovers
- **Rolling windows**: last 5, last 10 games
- **Home/away splits**: Separate rolling features for home vs away performance
- **Delta features**: Home minus away comparisons (9 deltas)
- **Seasonal reset**: Rolling windows restart at each season boundary

**Total features**: 145 (20 A1 + ~62 rolling + ~60 one-hot + 9 deltas)

**Why seasonal reset?**
- Teams change significantly between seasons (trades, draft, coaching)
- Prevents carryover of outdated information
- Aligns with NBA reality

**Results**: Test AUC 0.658, Accuracy 62.1%, **F1 0.718**

**Run**:
```bash
python run_stage_b1_intermediate.py
```

---

### Stage B1 Full - Comprehensive Team Statistics (Benchmark)

**What it adds to Intermediate**:
- **7 additional metrics**: points scored/allowed, point differential, steals, blocks, off/def rebounds
- **No seasonal reset**: Rolling windows continue across season boundaries (captures momentum)
- **More delta features**: 16 comparisons

**Total features**: 203 (20 A1 + ~177 rolling + ~60 one-hot)

**Trade-offs**:
- **+2.3% AUC** over Intermediate (0.674 vs 0.658)
- **-2.4% F1** (0.701 vs 0.718) - slightly worse precision/recall balance
- **+28.6% more features** - longer training, potential overfitting
- Continuous rolling may carry stale information across seasons

**Use case**: Research, feature importance analysis, upper-bound benchmark

**Results**: Test AUC 0.674, Accuracy 63.0%, F1 0.701

**Run**:
```bash
python run_stage_b1_full.py
```

See [Stage B1 Variants Comparison](docs/stage_b1_variants_comparison.md) for detailed analysis.

---

### Data Leakage Prevention

**Critical**: All rolling features use `.shift(1)` to ensure only **past games** are included.

**Validation**:
```bash
python test_stage_b1.py
```

This script:
- Samples 20 random games
- Manually recomputes rolling features from raw data
- Verifies no future data is used
- Checks merge correctness

See [Stage B1 Leakage Audit](docs/stage_b1_leakage_audit.md) for the comprehensive audit that identified and fixed 101 leakage columns.

---

## 🔄 Reproducibility & Archiving

### Archive System

Every pipeline run is **automatically archived** with:
- Timestamp: `run_YYYYMMDD_HHMMSS`
- Results JSON (all metrics, optimal threshold)
- Trained model (.keras)
- All visualizations (8 plots)
- Feature engineering outputs

**Archive structure**:
```
archives/
├── stage_a1/
│   ├── run_20251209_193450/  # Latest successful run
│   └── run_20251209_191510/  # Previous run (baseline)
├── stage_b1_intermediate/
│   └── run_20251212_011004/
└── stage_b1_full/
    └── run_20251212_XXXXXX/
```

### Comparing Runs

**Using analyze_results.py**:
```bash
python analyze_results.py
```

Shows:
- Current run metrics
- All archived runs
- Performance comparisons

**Manual comparison**:
```bash
# View archived results
cat archives/stage_a1/run_20251209_193450/results.json

# Compare two runs
diff archives/stage_a1/run_20251209_191510/results.json \
     archives/stage_a1/run_20251209_193450/results.json
```

### Archive Manager API

```python
from scripts.archive_manager import list_archives, compare_archives

# List all archives for a stage
archives = list_archives("archives/stage_a1")

# Compare two specific runs
comparison = compare_archives(
    "archives/stage_a1/run_20251209_191510",
    "archives/stage_a1/run_20251209_193450"
)
```

See [Archiving System Documentation](docs/archiving_system.md) for details.

---

## 📂 Project Structure

```
NBA_Games_Predictions_ML_NN/
│
├── data/                          # Raw CSV datasets
│   ├── Games.csv                 # 72,307 games (1946-2024)
│   ├── TeamStatistics.csv        # 144,614 rows (team performance)
│   └── PlayerStatistics.csv      # Not used yet
│
├── scripts/                       # Core pipeline modules
│   ├── load_data.py              # Data loading and filtering
│   ├── feature_engineering.py    # Stage A1 features
│   ├── preprocessing.py          # One-hot encoding, scaling, splitting
│   ├── train_model.py            # MLP training, threshold optimization
│   ├── visualize.py              # Comprehensive visualization suite
│   ├── archive_manager.py        # Result archiving and comparison
│   └── utils.py                  # Helper functions
│
├── features/                      # Stage B1 feature engineering
│   ├── stage_b1_teamstats.py     # Original full implementation
│   ├── stage_b1_enhanced.py      # Enhanced with seasonal reset
│   └── stage_b1_config.py        # Intermediate variant configuration
│
├── run_stage_a1.py               # Stage A1 pipeline
├── run_stage_b1_intermediate.py  # Stage B1 Intermediate pipeline
├── run_stage_b1_full.py          # Stage B1 Full pipeline
│
├── outputs/                       # Current run outputs (not version controlled)
│   ├── stage_a1/
│   ├── stage_b1_intermediate/
│   └── stage_b1_full/
│
├── models/                        # Trained models (not version controlled)
│   ├── stage_a1/
│   ├── stage_b1_intermediate/
│   └── stage_b1_full/
│
├── archives/                      # Historical runs (not version controlled)
│   ├── stage_a1/
│   │   ├── run_20251209_193450/
│   │   └── run_20251209_191510/
│   ├── stage_b1_intermediate/
│   │   └── run_20251212_011004/
│   └── stage_b1_full/
│
├── docs/                          # Comprehensive documentation
│   ├── stage_a1_analysis.md      # A1 results and improvements
│   ├── stage_b1_design.md        # B1 feature specifications
│   ├── stage_b1_variants_comparison.md  # Intermediate vs Full
│   ├── stage_b1_leakage_audit.md # Data leakage audit and fix
│   ├── archiving_system.md       # Archive system documentation
│   ├── repo_cleanup.md           # Repository cleanup log
│   └── repo_structure.md         # Detailed structure explanation
│
├── test_stage_b1.py              # Data leakage validation tests
├── analyze_results.py            # Results comparison utility
├── audit_stage_b1_leakage.py     # Historical audit reference
│
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore patterns
└── README.md                     # This file
```

---

## 🗺️ Roadmap

### ✅ Completed

- **Stage A1**: Games-only baseline (Test AUC 0.598)
- **Stage B1 Intermediate**: Team stats with 10 metrics, seasonal reset (Test AUC 0.658)
- **Stage B1 Full**: Team stats with 17 metrics, continuous rolling (Test AUC 0.674)
- **Data Leakage Audit**: Comprehensive validation and fix (101 columns identified)
- **Archive System**: Reproducible experiment tracking
- **Documentation**: Comprehensive guides for all stages

### 🔄 In Progress

- **Model 2 Architecture Design**: Enhanced MLP with embeddings and deeper layers

### 📋 Planned

#### Model 2: Enhanced Architecture (Same Data)

**Goal**: Improve model capacity while keeping B1 data

**Variants**:
- **Model 2A-base**: Baseline B1 data with new architecture
- **Model 2A-intermediate**: B1 Intermediate data + new architecture
- **Model 2A-full**: B1 Full data + new architecture

**Planned improvements**:
- Team embeddings (learned representations)
- Deeper architecture (more hidden layers)
- Batch normalization
- Advanced regularization (L1+L2, dropout tuning)
- Attention mechanisms (optional)

**Expected**: +2-4% AUC over Model 1

See [Model 2 Plan](docs/model_2_plan.md) for detailed design.

#### Stage C: Player Statistics

**Goal**: Add player-level features

**Challenges**:
- Large dataset (millions of rows)
- Player aggregation complexity
- Lineup combinations
- Injuries and rest management

**Approach**:
- Start with simple aggregations (top 5 players per team)
- Add position-based features
- Consider player embeddings

**Expected**: +1-3% AUC (diminishing returns due to inherent unpredictability)

**Status**: Postponed until Model 2 experiments complete

#### Model 3: Sequence Models

**Goal**: Capture temporal patterns

**Architectures**:
- LSTM/GRU: Sequence of recent games
- CNN-1D: Convolutional temporal features
- Transformer: Attention over game history

**Expected**: Marginal gains (+1-2% AUC), high computational cost

**Status**: Research phase

---

## 📚 Documentation

### Main Documents

- **[Stage A1 Analysis](docs/stage_a1_analysis.md)** - Detailed A1 results, improvements, and comparison
- **[Stage B1 Design](docs/stage_b1_design.md)** - Complete B1 feature specifications
- **[Stage B1 Variants Comparison](docs/stage_b1_variants_comparison.md)** - Intermediate vs Full analysis
- **[Stage B1 Leakage Audit](docs/stage_b1_leakage_audit.md)** - Data leakage discovery and fix
- **[Threshold Optimization Guide](docs/threshold_optimization.md)** - Multi-metric threshold selection
- **[Threshold Comparison](docs/threshold_comparison.md)** - Auto-generated results comparison
- **[Archiving System](docs/archiving_system.md)** - How to use the archive system
- **[Repository Cleanup](docs/repo_cleanup.md)** - Files removed and why
- **[Repository Structure](docs/repo_structure.md)** - Detailed structure explanation
- **[Model 2 Plan](docs/model_2_plan.md)** - Future architecture improvements

### Quick References

- **Run A1**: `python run_stage_a1.py`
- **Run B1 Intermediate**: `python run_stage_b1_intermediate.py`
- **Run B1 Full**: `python run_stage_b1_full.py`
- **Validate B1**: `python test_stage_b1.py`
- **Compare Results**: `python analyze_results.py`
- **Compare Thresholds**: `python compare_thresholds.py`

---

## 🤝 Contributing

This is a learning project, but suggestions and improvements are welcome!

**Areas for contribution**:
- Feature engineering ideas
- Model architecture improvements
- Documentation enhancements
- Bug reports and fixes

---

## 📝 License

MIT License - see LICENSE file for details

---

## 🙏 Acknowledgments

- **NBA**: For providing comprehensive historical data
- **Basketball-Reference.com**: For advanced basketball statistics
- **TensorFlow/Keras**: For the deep learning framework
- **Python Community**: For the excellent data science ecosystem

---

## 📞 Contact

For questions or discussions, please open an issue on GitHub.

---

**Last Updated**: December 12, 2025  
**Current Stage**: Stage B1 Complete (Intermediate & Full)  
**Next Milestone**: Model 2 Architecture Design
