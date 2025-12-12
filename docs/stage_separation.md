# Stage Separation Guide

## Overview

The project is now organized into **separate stages** to facilitate experimentation and comparison:

- **Stage A**: Historical features only (Games.csv)
- **Stage B**: Historical + team statistics (Games.csv + TeamStatistics.csv)
- **Stage C1+**: Player-level features (future)

Each stage has its own:
- Entry point script
- Output directory
- Model directory
- Archive directory

This ensures **no cross-contamination** between experiments and allows for **clean comparisons**.

---

## File Structure

### Entry Points

| File | Purpose | Status |
|------|---------|--------|
| `run_stage_a.py` | Stage A pipeline (historical features only) | ✅ Complete |
| `run_stage_b.py` | Stage B pipeline (+ team statistics) | 🔄 Structure ready |
| `run_pipeline.py` | **DEPRECATED** - Redirects to Stage A | ⚠️ Backwards compatibility |

### Output Organization

```
outputs/
├── stage_a/                    # Stage A outputs
│   ├── results.json
│   ├── filtered_games.csv
│   ├── features_engineered.csv
│   ├── plots/
│   ├── X_train.npy, y_train.npy, etc.
│   └── preprocessing_objects.pkl
└── stage_b/                    # Stage B outputs
    ├── results.json
    ├── filtered_games.csv
    ├── features_engineered.csv
    ├── plots/
    └── ...
```

### Model Organization

```
models/
├── stage_a/
│   └── mlp.keras               # Stage A trained model
└── stage_b/
    └── mlp.keras               # Stage B trained model
```

### Archive Organization

```
archives/
├── stage_a/
│   ├── run_20251209_191510/    # First Stage A run (baseline)
│   └── run_20251209_193450/    # Second Stage A run (improved)
└── stage_b/
    └── run_YYYYMMDD_HHMMSS/    # Future Stage B runs
```

---

## Usage

### Running Stage A

```bash
python run_stage_a.py
```

**What happens**:
1. Loads Games.csv
2. Engineers historical features
3. Trains MLP with regularization
4. Optimizes threshold
5. Generates visualizations
6. Saves to `outputs/stage_a/`
7. Saves model to `models/stage_a/mlp.keras`
8. **Archives to `archives/stage_a/run_YYYYMMDD_HHMMSS/`**

### Running Stage B

```bash
python run_stage_b.py
```

**Current status**: Uses same features as A1 (placeholder).  
**Next step**: Integrate TeamStatistics.csv features.

**What will happen** (when implemented):
1. Loads Games.csv + TeamStatistics.csv
2. Engineers historical + team stat features
3. Trains MLP with regularization
4. Optimizes threshold
5. Generates visualizations
6. Saves to `outputs/stage_b/`
7. Saves model to `models/stage_b/mlp.keras`
8. **Archives to `archives/stage_b/run_YYYYMMDD_HHMMSS/`**

### Analyzing Results

```bash
# Analyze most recent results
python analyze_results.py

# List all Stage A archives
ls archives/stage_a/

# List all Stage B archives
ls archives/stage_b/

# Compare Stage A runs
python scripts/archive_manager.py --compare \
  archives/stage_a/run_20251209_191510 \
  archives/stage_a/run_20251209_193450
```

---

## Comparing Stages

To compare Stage A vs Stage B performance:

1. **Run both stages**:
   ```bash
   python run_stage_a.py
   python run_stage_b.py  # Once implemented
   ```

2. **Compare latest runs**:
   ```bash
   # Get latest Stage A archive
   ls -t archives/stage_a/ | head -1
   
   # Get latest Stage B archive
   ls -t archives/stage_b/ | head -1
   
   # Compare
   python scripts/archive_manager.py --compare \
     archives/stage_a/run_YYYYMMDD_HHMMSS \
     archives/stage_b/run_YYYYMMDD_HHMMSS
   ```

3. **Expected improvements** (Stage A → B1):
   - Test Accuracy: 0.585 → 0.60+
   - Test AUC: 0.598 → 0.62–0.65
   - Better calibration
   - More discriminative features

---

## Migration Notes

### For Existing Users

If you were using `run_pipeline.py`:

**Old way** (still works but deprecated):
```bash
python run_pipeline.py  # Redirects to Stage A after 3 seconds
```

**New way** (recommended):
```bash
python run_stage_a.py  # Direct execution
```

**Changes**:
- ✅ All Stage A behavior is **identical**
- ✅ Same features, same model, same archiving
- ✅ Only difference: organized output paths

### Why Separate Stages?

1. **Clean experimentation**: Each stage isolated
2. **Easy comparison**: Compare A1 vs B1 directly
3. **No overwrites**: Archives don't conflict
4. **Scalability**: Ready for Stage C, D, etc.
5. **Reproducibility**: Exact replication of any stage

---

## Implementation Details

### Stage Configuration

Each stage script has a `STAGE_NAME` constant:

```python
# run_stage_a.py
STAGE_NAME = "stage_a"

# run_stage_b.py
STAGE_NAME = "stage_b"
```

This constant controls:
- Output directory: `outputs/{STAGE_NAME}/`
- Model path: `models/{STAGE_NAME}/mlp.keras`
- Archive path: `archives/{STAGE_NAME}/run_YYYYMMDD_HHMMSS/`

### Archive Manager Updates

The `archive_manager.py` now accepts stage-specific paths:

```python
archive_previous_results(
    outputs_dir=f"outputs/{STAGE_NAME}",
    archive_base_dir=f"archives/{STAGE_NAME}"
)
```

This ensures Stage A and Stage B archives never conflict.

---

## Next Steps

### Implementing Stage B Features

1. **Load team statistics**:
   ```python
   # In run_stage_b.py
   df_team_stats = pd.read_csv("data/TeamStatistics.csv")
   ```

2. **Merge with games**:
   ```python
   df_merged = merge_team_stats(df_filtered, df_team_stats)
   ```

3. **Engineer team stat features**:
   - Offensive rating (per 100 possessions)
   - Defensive rating
   - True shooting percentage
   - Turnover rate
   - Rebound percentage
   - Pace
   - Home/away splits
   - Rolling averages (10-game, 20-game)

4. **Update preprocessing** if needed

5. **Run and compare**:
   ```bash
   python run_stage_b.py
   python scripts/archive_manager.py --compare \
     archives/stage_a/run_latest \
     archives/stage_b/run_latest
   ```

### Future Stages

- **Stage C1**: Add player statistics aggregated at team level
- **Stage A2**: Try different model architectures (embeddings)
- **Stage A3**: Sequence models (LSTM/GRU) over recent games

Each will follow the same pattern with its own `run_stage_XX.py` file.

---

## Summary

✅ **Stage A**: Complete, optimized, ready for comparison  
🔄 **Stage B**: Structure ready, awaiting team statistics integration  
📋 **Stage C+**: Planned for future development

All stages are **independent**, **reproducible**, and **comparable**.
