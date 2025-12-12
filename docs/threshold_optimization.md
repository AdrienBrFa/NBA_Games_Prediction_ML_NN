# Threshold Optimization System

**Status:** ✅ Implemented  
**Version:** 1.0  
**Date:** December 2025

## Overview

The NBA game prediction pipeline now supports multiple threshold optimization strategies. This allows you to compare how different decision thresholds affect model performance on the test set.

### Key Features

- **Multiple optimization metrics**: F1, Accuracy, Balanced Accuracy
- **Backward compatible**: Default behavior unchanged (F1 optimization)
- **Automated archiving**: Results stored separately by threshold metric
- **Comparison tool**: Automated analysis across strategies
- **Zero retraining**: Same model weights, different thresholds

## Quick Start

### Run with Different Thresholds

```bash
# Default: F1-optimized threshold (backward compatible)
python run_stage_a.py

# Accuracy-optimized threshold
python run_stage_a.py --threshold_metric accuracy

# Balanced accuracy-optimized threshold
python run_stage_a.py --threshold_metric balanced_accuracy
```

Same syntax works for all pipelines:
- `run_stage_a.py`
- `run_stage_b_intermediate.py`
- `run_stage_b_full.py`

### Compare Results

After running multiple pipelines with different threshold strategies:

```bash
python compare_thresholds.py
```

This generates `docs/threshold_comparison.md` with comprehensive comparisons.

## Architecture

### Components

#### 1. **Threshold Optimizer** (`src/utils/thresholds.py`)

Generic utility for finding optimal classification thresholds.

```python
from src.utils.thresholds import find_optimal_threshold

# Find threshold that maximizes F1
threshold_f1, metrics_f1 = find_optimal_threshold(
    y_true, y_proba, metric="f1"
)

# Find threshold that maximizes accuracy
threshold_acc, metrics_acc = find_optimal_threshold(
    y_true, y_proba, metric="accuracy"
)
```

**Features:**
- Tests 91 thresholds from 0.05 to 0.95 (0.01 step)
- Supports: `f1`, `accuracy`, `balanced_accuracy`
- Tie-breaking: Selects threshold closest to 0.50 for stability
- Input validation and error handling
- Returns: `(best_threshold: float, metric_values: Dict)`

#### 2. **Pipeline Integration**

All three pipelines support the `--threshold_metric` CLI flag:

```python
def main(threshold_metric="f1"):
    # ... training code ...
    
    # Find optimal threshold using specified metric
    optimal_threshold, metric_values = find_optimal_threshold(
        y_val, y_val_pred_proba, metric=threshold_metric
    )
    
    # Save results with threshold metadata
    save_results(
        train_metrics, val_metrics, test_metrics, history,
        optimal_threshold=optimal_threshold,
        threshold_metric=threshold_metric,  # NEW
        output_path=str(output_path / "results.json")
    )
```

#### 3. **Archive Manager** (`scripts/archive_manager.py`)

Updated to include threshold metric in archive naming:

```python
archive_previous_results(
    outputs_dir=str(output_path),
    archive_base_dir=f"archives/{STAGE_NAME}",
    run_suffix=f"threshold-{threshold_metric}"  # NEW
)
```

**Archive structure:**
```
archives/
├── stage_a/
│   ├── run_20251212_143022_threshold-f1/
│   │   ├── results.json  # Contains threshold_metric: "f1"
│   │   ├── plots/
│   │   └── archive_info.json
│   ├── run_20251212_145533_threshold-accuracy/
│   │   ├── results.json  # Contains threshold_metric: "accuracy"
│   │   ├── plots/
│   │   └── archive_info.json
│   └── ...
├── stage_b_intermediate/
│   └── ...
└── stage_b_full/
    └── ...
```

#### 4. **Comparison Script** (`compare_thresholds.py`)

Automatically analyzes and compares results:

**Features:**
- Finds latest archive for each threshold metric per stage
- Generates comparison tables with key metrics
- Shows confusion matrices
- Calculates metric differences (Δ Accuracy, Δ F1, etc.)
- Provides interpretation guidance

**Output:** `docs/threshold_comparison.md`

## Implementation Details

### Threshold Selection Algorithm

From `src/utils/thresholds.py`:

```python
# Test 91 thresholds
threshold_grid = np.arange(0.05, 0.96, 0.01)

# For each threshold, compute metric
for threshold in threshold_grid:
    y_pred = (y_proba >= threshold).astype(int)
    score = metric_function(y_true, y_pred)
    
    # Track best (with tie-breaking)
    if score > best_score:
        best_score = score
        best_threshold = threshold
    elif score == best_score:
        # Tie-breaking: prefer threshold closest to 0.50
        if abs(threshold - 0.5) < abs(best_threshold - 0.5):
            best_threshold = threshold
```

### Results Storage

Updated `results.json` format:

```json
{
  "timestamp": "2025-12-12T14:30:22.123456",
  "optimal_threshold": 0.390,
  "threshold_metric": "f1",  // NEW FIELD
  "train_metrics": { ... },
  "val_metrics": { ... },
  "test_metrics": {
    "accuracy": 0.6214,
    "f1_score": 0.7184,
    "auc": 0.6582,
    "log_loss": 0.6548,
    "confusion_matrix": [[417, 1721], [193, 2442]]
  },
  "training_history": { ... }
}
```

## Usage Examples

### Example 1: Compare F1 vs Accuracy Optimization

```bash
# Run Stage A with F1 optimization (default)
python run_stage_a.py

# Run Stage A with Accuracy optimization
python run_stage_a.py --threshold_metric accuracy

# Compare results
python compare_thresholds.py
```

**Expected outcome:**
- Two archives in `archives/stage_a/`
- `docs/threshold_comparison.md` shows side-by-side comparison
- Metrics table shows differences: Δ Accuracy, Δ F1, etc.

### Example 2: Full Pipeline Comparison

```bash
# Run all stages with F1 optimization
python run_stage_a.py
python run_stage_b_intermediate.py
python run_stage_b_full.py

# Run all stages with Accuracy optimization
python run_stage_a.py --threshold_metric accuracy
python run_stage_b_intermediate.py --threshold_metric accuracy
python run_stage_b_full.py --threshold_metric accuracy

# Generate comprehensive comparison
python compare_thresholds.py
```

**Expected outcome:**
- 6 total archives (3 stages × 2 threshold metrics)
- `docs/threshold_comparison.md` with 3 stage sections
- Key findings section showing which strategy performs best per stage

### Example 3: Custom Analysis

```python
from src.utils.thresholds import find_optimal_threshold, compare_thresholds

# Load predictions
y_true = ...  # Ground truth labels
y_proba = ...  # Model predictions (probabilities)

# Find optimal thresholds for different metrics
thresh_f1, _ = find_optimal_threshold(y_true, y_proba, metric="f1")
thresh_acc, _ = find_optimal_threshold(y_true, y_proba, metric="accuracy")
thresh_bal, _ = find_optimal_threshold(y_true, y_proba, metric="balanced_accuracy")

# Compare all thresholds
comparison = compare_thresholds(y_true, y_proba, {
    "F1-optimized": thresh_f1,
    "Accuracy-optimized": thresh_acc,
    "Balanced-optimized": thresh_bal,
    "Default": 0.5
})

# Results: dict of metrics for each threshold
for name, metrics in comparison.items():
    print(f"{name}: Accuracy={metrics['accuracy']:.3f}, F1={metrics['f1']:.3f}")
```

## Design Decisions

### Why Not Just Use 0.5?

The default threshold of 0.5 assumes:
1. Balanced class distribution
2. Equal costs for false positives and false negatives
3. Well-calibrated probabilities

NBA games often have:
- **Imbalanced outcomes** (home team wins more often)
- **Cost asymmetry** (depending on betting strategy)
- **Miscalibrated probabilities** (model outputs aren't true probabilities)

**Optimizing the threshold on validation data improves test performance.**

### Why Optimize on Validation Set?

Threshold optimization uses validation set to prevent:
1. **Data leakage** (if we used test set)
2. **Overfitting** (if we used training set)
3. **Inconsistent evaluation** (test set never seen until final evaluation)

Standard ML practice: Train model on train set → Optimize hyperparameters (including threshold) on validation set → Evaluate on test set.

### Why Multiple Metrics?

Different use cases have different objectives:

| Metric | When to Use | Advantage |
|--------|-------------|-----------|
| **F1** | Balanced precision/recall | Good for imbalanced data |
| **Accuracy** | Overall correctness | Simple, interpretable |
| **Balanced Accuracy** | Severe class imbalance | Treats classes equally |

Comparing strategies helps identify the best approach for your specific goals.

### Why Tie-Breaking Toward 0.5?

When multiple thresholds achieve the same metric value:
- Prefer threshold closer to 0.5 for **stability**
- Extreme thresholds (0.05, 0.95) are less robust
- Central thresholds generalize better

## Testing

### Manual Testing

Test the CLI interface:

```bash
# Test help message
python run_stage_a.py --help

# Test different metrics
python run_stage_a.py --threshold_metric f1
python run_stage_a.py --threshold_metric accuracy
python run_stage_a.py --threshold_metric balanced_accuracy

# Verify archives created with correct suffixes
ls archives/stage_a/
```

### Unit Testing

Test the threshold optimizer:

```bash
# Run built-in tests
python src/utils/thresholds.py
```

Expected output:
```
Testing threshold optimizer...
F1: Best threshold: 0.050, Best score: 0.707
ACCURACY: Best threshold: 0.330, Best score: 0.570
BALANCED_ACCURACY: Best threshold: 0.490, Best score: 0.561
✓ Threshold optimizer working correctly
```

### Integration Testing

```bash
# Run comparison script (should work even with 1 archive per stage)
python compare_thresholds.py

# Check output exists
cat docs/threshold_comparison.md
```

## Troubleshooting

### Issue: Archives Not Found

**Symptom:** `compare_thresholds.py` shows "No archived results found"

**Solution:**
1. Check archives exist: `ls archives/stage_a/`
2. Verify `results.json` exists in each archive
3. Ensure archives follow naming convention: `run_YYYYMMDD_HHMMSS_threshold-{metric}/`

### Issue: Import Error for `src.utils.thresholds`

**Symptom:** `ModuleNotFoundError: No module named 'src.utils'`

**Solution:**
1. Verify `src/utils/__init__.py` exists
2. Verify `src/utils/thresholds.py` exists
3. Run from project root directory

### Issue: Comparison Shows N/A

**Symptom:** Comparison table shows "N/A" for all metrics

**Solution:**
1. Need at least 2 archives per stage (one for each threshold metric)
2. Run pipelines with different `--threshold_metric` values
3. Comparison requires both F1 and accuracy archives to show differences

## Future Enhancements

### Potential Improvements

1. **Additional Metrics**
   - Precision-optimized threshold
   - Recall-optimized threshold
   - Custom cost functions

2. **Visualization**
   - Threshold sweep plots (metric vs threshold)
   - ROC curves with optimal threshold markers
   - Precision-recall curves

3. **Automated Experiments**
   - Script to run all stages with all threshold metrics
   - Parallel execution for faster comparison
   - Automated email/notification of results

4. **Statistical Testing**
   - Bootstrap confidence intervals for metric differences
   - Statistical significance tests (McNemar's test)
   - Cross-validation for threshold selection

## References

### Related Documentation

- `README.md` - Main project documentation
- `docs/threshold_comparison.md` - Generated comparison report (after running comparison script)
- `docs/repo_structure.md` - Complete file structure
- `docs/model_2_plan.md` - Future model architecture plans

### External Resources

- [Threshold Selection in Binary Classification](https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/)
- [Optimizing Classification Thresholds](https://developers.google.com/machine-learning/crash-course/classification/thresholding)
- [Scikit-learn Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)

---

**Last Updated:** December 2025  
**Maintainer:** Repository Owner  
**Status:** Production Ready
