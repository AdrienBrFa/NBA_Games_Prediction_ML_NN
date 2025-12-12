# Threshold Optimization - Quick Reference

## Basic Usage

```bash
# Default: F1-optimized (backward compatible)
python run_stage_a1.py

# Accuracy-optimized
python run_stage_a1.py --threshold_metric accuracy

# Balanced accuracy-optimized
python run_stage_a1.py --threshold_metric balanced_accuracy
```

## Full Comparison Experiment

```bash
# Run all stages with both F1 and Accuracy
python run_stage_a1.py --threshold_metric f1
python run_stage_a1.py --threshold_metric accuracy

python run_stage_b1_intermediate.py --threshold_metric f1
python run_stage_b1_intermediate.py --threshold_metric accuracy

python run_stage_b1_full.py --threshold_metric f1
python run_stage_b1_full.py --threshold_metric accuracy

# Generate comparison report
python compare_thresholds.py

# View results
cat docs/threshold_comparison.md
```

## Python API

```python
from src.utils.thresholds import find_optimal_threshold, compare_thresholds

# Find optimal threshold
threshold, metrics = find_optimal_threshold(
    y_true, y_proba, 
    metric="f1"  # or "accuracy" or "balanced_accuracy"
)

# Compare multiple thresholds
results = compare_thresholds(y_true, y_proba, {
    "F1-opt": 0.390,
    "Acc-opt": 0.520,
    "Default": 0.5
})
```

## Metrics Explained

| Metric | Optimizes For | Use When |
|--------|---------------|----------|
| **f1** | Balance precision & recall | General use, imbalanced data |
| **accuracy** | Overall correctness | Simple goal: max correct predictions |
| **balanced_accuracy** | Equal class performance | Severe class imbalance |

## Output Locations

- **Archives:** `archives/{stage}/run_YYYYMMDD_HHMMSS_threshold-{metric}/`
- **Results:** `results.json` (includes `threshold_metric` field)
- **Comparison:** `docs/threshold_comparison.md`

## Help

```bash
python run_stage_a1.py --help
```

## Documentation

- **Complete Guide:** `docs/threshold_optimization.md`
- **Implementation Details:** `THRESHOLD_SYSTEM_COMPLETE.md`
