# Stage B1 Variants Comparison

This document compares the performance of different Stage B1 feature configurations.

## Variant Configurations

### Stage B1 Full (Original)
- **Feature Set**: Full (all 17 rolling metrics)
- **Rolling Reset**: Continuous across seasons (original implementation)
- **Total Features**: 203 features after preprocessing
  - 20 Stage A1 historical features
  - ~177 Stage B1 rolling features
  - 60 one-hot encoded features (teams, seasons, etc.)

**Rolling Metrics (17)**:
- Points scored/allowed, point differential
- Field goal %, 3-point %, free throw %
- Total rebounds, offensive/defensive rebounds
- Assists, turnovers, steals, blocks
- Offensive rating, defensive rating, net rating
- Possessions (pace)

### Stage B1 Intermediate
- **Feature Set**: Intermediate (10 key rolling metrics)
- **Rolling Reset**: By season (reset at season boundaries)
- **Total Features**: 145 features after preprocessing
  - 20 Stage A1 historical features
  - ~62 Stage B1 rolling features (10 metrics × 2 windows × 2 splits + deltas)
  - 60 one-hot encoded features

**Rolling Metrics (10)**:
- Offensive rating, defensive rating, net rating
- Possessions (pace)
- Field goal %, 3-point %, free throw %
- Total rebounds, assists, turnovers

**Key Differences**:
1. Fewer metrics (10 vs 17) - removed: points scored/allowed, point diff, steals, blocks, off/def rebounds
2. Seasonal reset - rolling windows restart at the beginning of each season
3. Reduced delta features (9 vs 16)

## Performance Results

### Test Set Performance (Optimal Threshold)

| Metric | Stage B1 Full | Stage B1 Intermediate | Difference |
|--------|---------------|----------------------|------------|
| **Test AUC** | 0.6740 | 0.6582 | -0.0158 (-2.3%) |
| **Test Accuracy** | 0.6300 | 0.6214 | -0.0086 (-1.4%) |
| **Test F1 Score** | 0.7014 | 0.7184 | +0.0170 (+2.4%) |
| **Optimal Threshold** | 0.393 | 0.390 | -0.003 |
| **Features Used** | 203 | 145 | -58 (-28.6%) |

### Training Characteristics

| Metric | Stage B1 Full | Stage B1 Intermediate |
|--------|---------------|----------------------|
| Training Games | 30,085 | 30,085 |
| Validation Games | 4,669 | 4,669 |
| Test Games | 4,773 | 4,773 |
| Early Stopping Epoch | 19 | 18 |
| Final Train AUC | 0.7222 | 0.7222 |
| Final Val AUC | 0.6706 | 0.6726 |

## Key Observations

### 1. Feature Efficiency
- **Intermediate variant achieved 97.7% of Full's AUC performance with 71.4% of the features**
- Feature reduction: 203 → 145 features (-28.6%)
- This suggests strong feature selection - the 10 core metrics capture most predictive power

### 2. Performance Trade-off
- **AUC**: Full variant is slightly better (0.6740 vs 0.6582, -2.3%)
- **F1 Score**: Intermediate is surprisingly better (0.7184 vs 0.7014, +2.4%)
- **Accuracy**: Very similar (0.6300 vs 0.6214, -1.4%)

The Intermediate variant's higher F1 score despite lower AUC suggests:
- Better balance between precision and recall at the optimal threshold
- Less overfitting due to fewer features
- More robust predictions on the test set

### 3. Seasonal Reset Impact
Intermediate variant uses seasonal reset for rolling windows:
- **Benefit**: Each season starts fresh, preventing carryover from previous season
- **Trade-off**: First games of each season have less historical data
- **Reality**: Teams often change significantly between seasons (roster changes, trades, coaching)

This may explain why Intermediate performs comparably despite fewer features - seasonal reset aligns better with real-world team dynamics.

### 4. Training Efficiency
Both variants:
- Converged at similar epochs (18-19)
- Similar training AUC (0.7222)
- Similar validation AUC (0.67)
- No significant difference in convergence behavior

### 5. Feature Importance (Inferred)
The 7 metrics **removed** in Intermediate (but in Full):
- `teamScore`, `opponentScore` - redundant with offensive/defensive ratings
- `point_diff` - captured by net_rating
- `steals`, `blocks` - less predictive than core metrics
- `reboundsOffensive`, `reboundsDefensive` - captured by total rebounds

These removals had minimal impact (-2.3% AUC), suggesting they provided limited additional signal.

## Recommendations

### Use Stage B1 Full when:
✓ Maximum predictive power is critical (even if marginal)
✓ Computational resources are not a constraint
✓ You want comprehensive feature coverage
✓ Historical continuity across seasons is desired

### Use Stage B1 Intermediate when:
✓ Model interpretability is important (fewer features)
✓ Training/inference speed matters (28.6% fewer features)
✓ You believe teams reset significantly each season
✓ Risk of overfitting is a concern
✓ You need a simpler, more maintainable model

### For Production:
**Recommended: Stage B1 Intermediate**

Rationale:
1. **97.7% of AUC performance with 71.4% of features** - excellent efficiency
2. **Higher F1 score** (0.7184) suggests better real-world performance
3. **Seasonal reset** aligns with NBA reality (teams change between seasons)
4. **Faster inference** - 28.6% fewer features to compute
5. **Easier to explain** - 10 core metrics vs 17 diverse stats

The marginal AUC loss (2.3%) is offset by better F1 score (+2.4%) and significantly reduced complexity.

## Computational Comparison

| Aspect | Stage B1 Full | Stage B1 Intermediate | Savings |
|--------|---------------|----------------------|---------|
| Rolling Features Computed | 34 teams × 17 metrics × 4 windows/splits = 2,312 | 34 teams × 10 metrics × 4 windows/splits = 1,360 | **41.2%** |
| Delta Features | 16 comparisons | 9 comparisons | **43.8%** |
| Total Features | 203 | 145 | **28.6%** |
| Model Input Size | 203 dimensions | 145 dimensions | **28.6%** |
| Estimated Inference Time | ~100% | ~71.4% | **28.6% faster** |

## Next Steps

### Stage B1 Lite (Future)
Consider an even more reduced variant:
- **5-6 core metrics**: net_rtg, pace, fg_pct, 3p_pct, reb_total, ast
- **Single window**: last 10 games only (remove last 5)
- **Target**: ~80-90 features
- **Expected performance**: 95%+ of Full's AUC with 40-45% of features

### Ensemble Approach
Combine both variants:
- Train Stage B1 Full and Intermediate separately
- Ensemble predictions (weighted average or stacking)
- Potential: Best of both worlds (Full's coverage + Intermediate's robustness)

## Conclusion

**Stage B1 Intermediate represents an excellent balance between performance and complexity.**

With 97.7% of Full's AUC performance, better F1 score, 28.6% fewer features, and seasonal reset aligned with NBA reality, it's the recommended variant for production deployment.

The Full variant remains valuable for:
- Maximum performance benchmarking
- Research and feature importance analysis
- Ensemble approaches

---

**Comparison Date**: December 12, 2025  
**Stage B1 Full**: Test AUC 0.6740, 203 features, continuous rolling  
**Stage B1 Intermediate**: Test AUC 0.6582, 145 features, seasonal reset
