# Repository Refactoring Summary

**Date:** December 12, 2025  
**Purpose:** Separate data stages (A, B, C) from model versions (1, 2, 3)

## ✅ Completed Changes

### 1. Pipeline Scripts Renamed
- `run_stage_a1.py` → `run_stage_a.py`
- `run_stage_b1_intermediate.py` → `run_stage_b_intermediate.py`
- `run_stage_b1_full.py` → `run_stage_b_full.py`

**Usage:**
```bash
# Choose data stage (A, B-intermediate, B-full)
# Choose model version (1 or 2) with --model_version flag

# Stage A with Model 1 (default)
python run_stage_a.py

# Stage A with Model 2
python run_stage_a.py --model_version 2

# Stage B Intermediate with Model 2, 8D embeddings
python run_stage_b_intermediate.py --model_version 2 --embedding_dim 8

# Stage B Full with Model 1
python run_stage_b_full.py --model_version 1
```

### 2. Directory Structure Updated
```
outputs/
├── stage_a/              (was stage_a1/)
├── stage_b_intermediate/ (was stage_b1_intermediate/)
└── stage_b_full/         (was stage_b1_full/)

models/
├── stage_a/              (was stage_a1/)
├── stage_b_intermediate/ (was stage_b1_intermediate/)
└── stage_b_full/         (was stage_b1_full/)

archives/
├── stage_a/              (was stage_a1/)
├── stage_b_intermediate/ (was stage_b1_intermediate/)
├── stage_b_full/         (was stage_b1_full/)
└── stage_b_legacy/       (was stage_b1/ - old runs)
```

### 3. Feature Modules Renamed
- `features/stage_b1_config.py` → `features/stage_b_config.py`
- `features/stage_b1_enhanced.py` → `features/stage_b_enhanced.py`
- `features/stage_b1_teamstats.py` → `features/stage_b_teamstats.py`

### 4. Test Files Renamed
- `test_stage_b1.py` → `test_stage_b.py`
- `audit_stage_b1_leakage.py` → `audit_stage_b_leakage.py`

### 5. Documentation Updated
All documentation files updated to reflect new naming:
- README.md
- docs/model_2_plan.md
- docs/stage_a_analysis.md (was stage_a1_analysis.md)
- docs/stage_b_design.md (was stage_b1_design.md)
- docs/stage_b_variants_comparison.md
- docs/stage_b_leakage_audit.md
- All other docs files

## 📊 Naming Convention

### Clear Separation
- **Data Stages:** A, B, C (letters) - describe feature sets
  - **Stage A:** Baseline features (81 features)
  - **Stage B:** Team statistics features (145-221 features)
  - **Stage C:** Future expansion (player stats, etc.)

- **Model Versions:** 1, 2, 3 (numbers) - describe architectures
  - **Model 1:** Original MLP (128-64-32)
  - **Model 2:** MLP + Team Embeddings
  - **Model 3:** Future architectures (RNN, Transformer, etc.)

### Stage B Variants
- **stage_b_intermediate:** 10 metrics, seasonal reset (145 features)
- **stage_b_full:** 17 metrics, continuous rolling (221 features)

## 🎯 Additional Cleanup Recommendations

### Priority 1: High Impact, Low Effort

1. **Rename old model file**
   ```bash
   # Remove old naming artifact
   Remove-Item "models/stage_a1_mlp.keras" -ErrorAction SilentlyContinue
   ```

2. **Update .gitignore patterns**
   - Add `outputs/stage_*/` instead of `outputs/stage_a1/`, etc.
   - Add `models/stage_*/` for all stage directories
   - Add `archives/stage_*/` for consistency

3. **Consolidate test files**
   ```
   tests/
   ├── test_stage_a.py
   ├── test_stage_b.py
   ├── test_models.py       (combine model tests)
   └── test_embeddings.py   (test_team_encoder.py + test_embedding_extraction.py)
   ```

4. **Organize utility scripts**
   ```
   utils/
   ├── analyze_results.py     (from root)
   ├── compare_thresholds.py  (from root)
   └── audit_stage_b_leakage.py (from root)
   ```

### Priority 2: Medium Impact, Medium Effort

5. **Create Stage C placeholder**
   ```python
   # features/stage_c_config.py
   # Future: Player-level features
   # - Individual player statistics
   # - Injury reports
   # - Player matchup history
   ```

6. **Standardize archive naming**
   - Ensure all archives use format: `run_YYYYMMDD_HHMMSS_threshold-{metric}_model{version}`
   - Old archives without model version can stay as-is

7. **Add Model 3 placeholder**
   ```python
   # src/models/model_3.py
   # Future: Advanced architectures
   # - Recurrent models (LSTM/GRU)
   # - Attention mechanisms
   # - Transformer architectures
   ```

8. **Improve pipeline configuration**
   ```python
   # config/pipeline_config.py
   STAGES = {
       'A': {'features': 81, 'description': 'Baseline'},
       'B': {'features': '145-221', 'description': 'Team stats'},
       'C': {'features': 'TBD', 'description': 'Player stats'}
   }
   
   MODELS = {
       1: {'architecture': 'MLP', 'params': '~27K'},
       2: {'architecture': 'MLP+Embeddings', 'params': '28-33K'},
       3: {'architecture': 'TBD', 'params': 'TBD'}
   }
   ```

### Priority 3: Lower Priority, Higher Effort

9. **Create unified run script**
   ```bash
   # run_pipeline.py --stage {a,b_inter,b_full,c} --model {1,2,3} [options]
   python run_pipeline.py --stage b_full --model 2 --threshold_metric f1
   ```

10. **Refactor feature engineering**
    - Create `features/base.py` with common feature logic
    - Each stage becomes a configuration, not separate files
    - DRY principle: remove duplicated code

11. **Add experiment tracking**
    - Integrate MLflow or Weights & Biases
    - Track all hyperparameters automatically
    - Compare experiments visually

12. **Create performance dashboard**
    - Web interface to compare all archived runs
    - Filter by stage, model, threshold metric
    - Visualize performance trends over time

## 🔍 Quality Improvements

### Code Quality
- [ ] Add type hints to all functions
- [ ] Increase docstring coverage to 100%
- [ ] Add unit tests for all utility functions
- [ ] Set up pre-commit hooks (black, flake8, mypy)

### Documentation
- [ ] Add architecture diagrams
- [ ] Create API reference documentation
- [ ] Add troubleshooting guide
- [ ] Document common workflows

### CI/CD
- [ ] Set up GitHub Actions for testing
- [ ] Automated model validation on push
- [ ] Performance regression detection
- [ ] Automated documentation builds

## 📝 Migration Notes

### For Existing Scripts
If you have custom scripts referencing old names:
```python
# OLD
from features.stage_b1_enhanced import engineer_stage_b1_features_configurable

# NEW
from features.stage_b_enhanced import engineer_stage_b1_features_configurable
```

### For Archive References
Old archives remain accessible:
```bash
# Still works
archives/stage_a/run_20251212_020547_threshold-f1/results.json

# Old naming also preserved in archive subdirectories
archives/stage_b_legacy/run_20251211_235817/results.json
```

## ✅ Testing Verification

Confirmed working:
```bash
$ python run_stage_a.py --help
✓ Script loads successfully
✓ All imports resolved
✓ Help text displays correctly

$ python run_stage_b_intermediate.py --model_version 2
✓ Pipeline runs without errors
✓ Feature modules imported correctly
✓ Outputs saved to stage_b_intermediate/

$ python run_stage_b_full.py --model_version 2 --embedding_dim 8
✓ Model 2 with 8D embeddings trains successfully
✓ Results archived correctly
✓ Embeddings visualized
```

## 🎉 Benefits Achieved

1. **Clarity:** Stages (data) vs Models (architecture) clearly separated
2. **Scalability:** Easy to add Stage C, Model 3, Model 4, etc.
3. **Consistency:** Uniform naming across all files and directories
4. **Maintainability:** Easier to understand project structure
5. **Documentation:** All docs updated and aligned with new naming

## 📚 Quick Reference

### Command Patterns
```bash
# Pattern: python run_stage_{STAGE}.py --model_version {MODEL} [options]

# Stage A (Baseline)
python run_stage_a.py --model_version 2 --threshold_metric f1

# Stage B Intermediate (Recommended)
python run_stage_b_intermediate.py --model_version 2 --embedding_dim 16

# Stage B Full (Benchmark)
python run_stage_b_full.py --model_version 1 --threshold_metric accuracy
```

### Directory Lookup
- **Stage outputs:** `outputs/stage_{a|b_intermediate|b_full}/`
- **Trained models:** `models/stage_{a|b_intermediate|b_full}/`
- **Archived runs:** `archives/stage_{a|b_intermediate|b_full}/run_YYYYMMDD_HHMMSS_*/`

---

**Refactoring Status:** ✅ **COMPLETE**  
All core changes implemented, tested, and documented.
