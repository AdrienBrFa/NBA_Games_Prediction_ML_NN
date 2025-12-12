# Model 2 Plan: Enhanced MLP Architecture with Team Embeddings

**Status**: ✅ **IMPLEMENTED**  
**Goal**: Improve model capacity with team embeddings + enhanced MLP architecture  
**Expected Gain**: +2-4% AUC over Model 1

---

## Overview

Model 2 introduces architectural improvements to increase model capacity and learning ability while maintaining compatibility with all existing stages (A1, B1-intermediate, B1-full).

**Key Features**:
- ✅ Learned team embeddings (replaces one-hot encoding)
- ✅ Enhanced MLP architecture with BatchNorm
- ✅ Improved regularization strategy
- ✅ 2D embedding visualization for interpretability
- ✅ Full backward compatibility with Model 1

---

## What Are Team Embeddings?

### The Problem with One-Hot Encoding

In Model 1, teams are represented using **one-hot encoding**:
- 30 NBA teams → 30 binary features
- Each team is a completely independent dimension
- No learned relationships between teams
- Example: `LAL = [1,0,0,...], GSW = [0,1,0,...]`

**Limitations**:
1. Large dimensionality (60 features for home + away teams)
2. No notion of team similarity or strength
3. Cannot generalize across similar teams
4. Each team learned independently

### The Solution: Learned Embeddings

**Team embeddings** are dense, low-dimensional vector representations of teams:
- 30 teams → 16-dimensional vectors (configurable)
- Model learns relationships during training
- Similar teams cluster in embedding space
- Example: Strong defensive teams might have similar embeddings

**How It Works**:
1. Each team gets a unique integer ID (0-29)
2. An embedding layer maps ID → dense vector (e.g., 16 dimensions)
3. For each game: `home_emb = lookup(home_team_id)`, `away_emb = lookup(away_team_id)`
4. Combine: `concat(home_emb, away_emb, home_emb - away_emb)`
5. Concatenate with numeric features → MLP

**Benefits**:
- Dimensionality reduction: 60 → 48 features (for dim=16)
- Learned team strength/style representations
- Better generalization to new matchups
- Interpretable via visualization

---

## Model 1 vs Model 2

### Model 1 (Current - Baseline)

**Architecture**:
```
Input (145-203 features)
  ↓
Dense(80, ReLU) + L2(0.001) + Dropout(0.3)
  ↓
Dense(40, ReLU) + L2(0.001) + Dropout(0.3)
  ↓
Dense(20, ReLU) + L2(0.001) + Dropout(0.2)
  ↓
Dense(1, Sigmoid)
```

**Regularization**:
- L2 penalty (0.001)
- Dropout (0.3, 0.3, 0.2)
- Early stopping (patience=5)
- Learning rate reduction (patience=3, factor=0.5)

**Results** (B1 Intermediate):
- Test AUC: 0.658
- Test Accuracy: 62.1%
- Test F1: 0.718

**Limitations**:
1. Fixed-width hidden layers (80-40-20) - limited capacity
2. No learned representations for categorical features (teams, seasons)
3. No batch normalization - training stability issues with deeper networks
4. Simple dropout schedule - not optimized
5. No residual connections - limits depth

---

## Model 2 Improvements

### 1. Team Embeddings (Critical)

**Current (Model 1)**: One-hot encoding for teams
- 30 teams → 30 binary features
- No learned relationships between teams
- Large input dimensionality

**Model 2**: Learned team embeddings
- 30 teams → embedding dimension (e.g., 8-16)
- Model learns team similarity (e.g., strong teams cluster together)
- Reduced dimensionality
- Better generalization

**Implementation**:
```python
team_input = Input(shape=(1,), name='team_id')
team_embedding = Embedding(
    input_dim=30,  # 30 NBA teams
    output_dim=12,  # embedding dimension
    name='team_embedding'
)(team_input)
team_embedding = Flatten()(team_embedding)
```

**Benefit**: +1-2% AUC (teams have inherent quality differences)

---

### 2. Season Embeddings (Optional)

**Current (Model 1)**: One-hot encoding for seasons
- 35 seasons (1990-2024) → 35 binary features
- No temporal relationships

**Model 2**: Learned season embeddings
- Captures era differences (e.g., pace changes, rule changes)
- Smaller dimension: 35 seasons → 4-8 embedding dimensions

**Implementation**:
```python
season_input = Input(shape=(1,), name='season_year')
season_embedding = Embedding(
    input_dim=35,  # 1990-2024
    output_dim=6,   # embedding dimension
    name='season_embedding'
)(season_input)
season_embedding = Flatten()(season_embedding)
```

**Benefit**: +0.5-1% AUC (eras have distinct playing styles)

---

### 3. Deeper Architecture with Batch Normalization

**Current (Model 1)**: 3 hidden layers (80-40-20)
- Shallow - limited capacity
- No batch norm - training instability

**Model 2**: 5-6 hidden layers with batch norm
```
Input Layer
  ↓
[Home Team Embedding (12) + Away Team Embedding (12) + Season (6) + Numeric Features (~120)]
  ↓
Dense(256, ReLU) + Batch Norm + Dropout(0.4)
  ↓
Dense(128, ReLU) + Batch Norm + Dropout(0.4)
  ↓
Dense(64, ReLU) + Batch Norm + Dropout(0.3)
  ↓
Dense(32, ReLU) + Batch Norm + Dropout(0.3)
  ↓
Dense(16, ReLU) + Batch Norm + Dropout(0.2)
  ↓
Dense(1, Sigmoid)
```

**Batch Normalization Benefits**:
- Stabilizes training (normalizes activations)
- Allows higher learning rates
- Reduces internal covariate shift
- Acts as additional regularization

**Benefit**: +1-2% AUC (more capacity to learn complex patterns)

---

### 4. Advanced Regularization

**Current (Model 1)**:
- L2 only (0.001)
- Fixed dropout rates

**Model 2**:
- **L1 + L2 (ElasticNet)**: Encourages sparsity + weight decay
  - L1: 0.0001 (feature selection)
  - L2: 0.001 (weight smoothing)
- **Tuned dropout schedule**: Higher dropout in earlier layers
  - Early layers: 0.4 (prevent memorization)
  - Middle layers: 0.3 (balance)
  - Late layers: 0.2 (preserve information)
- **Gradient clipping**: Prevents exploding gradients with deeper network
  - `clipnorm=1.0` or `clipvalue=0.5`

**Benefit**: +0.5-1% AUC (better generalization)

---

### 5. Advanced Optimizers (Optional)

**Current (Model 1)**: Adam with default parameters
- Learning rate: 0.001
- No warmup, no scheduling

**Model 2**: Adam with improvements
- **Learning rate warmup**: Start low (1e-5), ramp up to 1e-3 over 5 epochs
- **Cosine annealing**: Gradually reduce LR following cosine curve
- **Gradient centralization**: Normalize gradients (improves convergence)

**Alternative**: AdamW (Adam with decoupled weight decay)
- Better weight decay implementation
- Often outperforms Adam

**Benefit**: +0.5-1% AUC (better convergence)

---

### 6. Attention Mechanisms (Advanced - Optional)

**Concept**: Let model focus on most important features

**Implementation**:
```python
# Self-attention over features
attention_scores = Dense(num_features, activation='softmax')(hidden_layer)
weighted_features = Multiply()([input_features, attention_scores])
```

**Benefit**: +0.5-1% AUC (if features have varying importance)

**Caution**: Adds complexity, requires careful tuning

---

## Model 2 Variants

### Model 2A: Architecture Only (Baseline B1 Data)

**Goal**: Test new architecture with minimal data

**Data**: Same as Stage B Full (original implementation)
- No seasonal reset
- All 17 metrics
- 203 features

**Purpose**: Isolate architecture improvements

**Expected**: Test AUC 0.69-0.71 (+1.6-3.6% over Model 1 Full)

---

### Model 2B-Intermediate: New Architecture + B1 Intermediate Data

**Goal**: Production-ready model with best data and best architecture

**Data**: Stage B Intermediate
- 10 core metrics
- Seasonal reset
- 145 features

**Purpose**: Recommended variant for deployment

**Expected**: Test AUC 0.68-0.70 (+2.2-4.2% over Model 1 Intermediate)

**Target**: F1 score > 0.73 (currently 0.718)

---

### Model 2B-Full: New Architecture + B1 Full Data

**Goal**: Maximum performance benchmark

**Data**: Stage B Full
- All 17 metrics
- Continuous rolling
- 203 features

**Purpose**: Research and upper-bound estimation

**Expected**: Test AUC 0.70-0.72 (+2.6-4.6% over Model 1 Full)

**Risk**: Overfitting with more features + more capacity

---

## Implementation Plan

### Phase 1: Basic Model 2 (Embeddings + Deeper Network)

**Tasks**:
1. Create `src/models/model_2/` directory structure
2. Implement team embeddings (home + away)
3. Implement deeper network (5 layers, batch norm)
4. Train Model 2A (baseline B1 data)
5. Compare with Model 1 baseline

**Files to create**:
- `src/models/model_2/__init__.py`
- `src/models/model_2/embeddings.py` - embedding layer utilities
- `src/models/model_2/architecture.py` - Model 2 architecture definition
- `src/models/model_2/train.py` - training loop with new architecture
- `run_model_2a.py` - pipeline script

**Expected completion**: 1-2 days

**Success criteria**: Model 2A achieves > 0.69 test AUC

---

### Phase 2: Enhanced Regularization

**Tasks**:
1. Add L1+L2 regularization
2. Tune dropout schedule
3. Add gradient clipping
4. Experiment with AdamW optimizer

**Expected completion**: 1 day

**Success criteria**: Model 2A achieves > 0.70 test AUC

---

### Phase 3: Model 2B Variants

**Tasks**:
1. Create `run_model_2b_intermediate.py`
2. Create `run_model_2b_full.py`
3. Train both variants
4. Compare all 5 models (A1, B1-int, B1-full, M2B-int, M2B-full)
5. Document results in `docs/model_2_results.md`

**Expected completion**: 1 day

**Success criteria**: Model 2B-intermediate achieves > 0.68 test AUC with F1 > 0.73

---

### Phase 4: Advanced Features (Optional)

**Tasks**:
1. Implement season embeddings
2. Experiment with attention mechanisms
3. Try cosine annealing schedule
4. Hyperparameter tuning (embedding dimensions, layer sizes)

**Expected completion**: 2-3 days

**Success criteria**: Squeeze out additional +0.5-1% AUC

---

## Expected Results Summary

| Model | Data | Features | Arch Layers | Expected Test AUC | Expected F1 | Gain vs M1 |
|-------|------|----------|-------------|-------------------|-------------|------------|
| M1 A1 | Games | 20 | 3 (80-40-20) | 0.598 | 0.710 | Baseline |
| M1 B1-int | +Team Stats | 145 | 3 (80-40-20) | 0.658 | 0.718 | +6.0% |
| M1 B1-full | +Team Stats | 203 | 3 (80-40-20) | 0.674 | 0.701 | +7.6% |
| **M2A** | +Team Stats | 203 | 5 (256-128-64-32-16) | **0.69-0.71** | **0.71-0.72** | **+9.2-11.2%** |
| **M2B-int** | +Team Stats | 145 | 5 (256-128-64-32-16) | **0.68-0.70** | **0.73-0.74** | **+8.2-10.2%** |
| **M2B-full** | +Team Stats | 203 | 5 (256-128-64-32-16) | **0.70-0.72** | **0.72-0.73** | **+10.2-12.2%** |

---

## Risk Assessment

### Risk 1: Overfitting (High)

**Problem**: Deeper network + more capacity → memorization

**Mitigation**:
- Strong regularization (L1+L2, dropout 0.4)
- Early stopping (patience=5)
- Monitor train-val gap closely
- Use B1 Intermediate (fewer features) as primary variant

---

### Risk 2: Unstable Training (Medium)

**Problem**: Deeper network → gradient issues

**Mitigation**:
- Batch normalization after each layer
- Gradient clipping (clipnorm=1.0)
- Learning rate warmup
- Monitor gradient norms during training

---

### Risk 3: Marginal Gains (Medium)

**Problem**: Architecture improvements may not translate to large gains

**Reality Check**:
- NBA games are inherently unpredictable (~60-70% ceiling)
- Diminishing returns as model improves
- Expected +2-4% AUC is realistic but not guaranteed

**Response**: Document all experiments, keep Model 1 as fallback

---

### Risk 4: Hyperparameter Explosion (Low)

**Problem**: More hyperparameters to tune (embedding dims, layer sizes, etc.)

**Mitigation**:
- Start with reasonable defaults from literature
- Tune systematically (one parameter at a time)
- Use validation set for tuning
- Document all configurations in `configs/model_2_config.json`

---

## Success Criteria

### Minimum Viable Product (MVP)

✅ Model 2A achieves **> 0.69 test AUC** (+1.6% over M1 Full)  
✅ Model 2B-intermediate achieves **> 0.68 test AUC** (+2.2% over M1 Intermediate)  
✅ No significant overfitting (train-val AUC gap < 0.05)  
✅ All pipelines run without errors  
✅ Results properly archived and documented

### Stretch Goals

🎯 Model 2B-intermediate achieves **> 0.70 test AUC** (+4.2% over M1 Intermediate)  
🎯 Model 2B-intermediate achieves **F1 > 0.73** (currently 0.718)  
🎯 Model 2B-full achieves **> 0.72 test AUC** (+4.6% over M1 Full)  
🎯 Implement attention mechanisms successfully

---

## File Structure (After Model 2)

```
NBA_Games_Predictions_ML_NN/
│
├── src/
│   └── models/
│       ├── model_1/              # Current simple MLP (optional refactor)
│       └── model_2/              # NEW: Enhanced MLP
│           ├── __init__.py
│           ├── embeddings.py     # Embedding utilities
│           ├── architecture.py   # Model 2 architecture
│           ├── train.py          # Training loop
│           └── config.py         # Hyperparameters
│
├── configs/                       # NEW: Configuration files
│   ├── model_1_config.json       # Model 1 hyperparameters
│   └── model_2_config.json       # Model 2 hyperparameters
│
├── run_model_2a.py               # NEW: Model 2A pipeline
├── run_model_2b_intermediate.py  # NEW: Model 2B intermediate
├── run_model_2b_full.py          # NEW: Model 2B full
│
├── docs/
│   ├── model_2_plan.md           # This file
│   └── model_2_results.md        # NEW: Experimental results
│
├── archives/
│   ├── model_2a/                 # NEW: M2A archives
│   ├── model_2b_intermediate/    # NEW: M2B-int archives
│   └── model_2b_full/            # NEW: M2B-full archives
```

---

## Implementation Details

### Files Created

```
src/
├── models/
│   └── model_2.py              # ✅ Model 2 architecture with embeddings
├── utils/
    ├── team_encoding.py         # ✅ Team ID encoding utilities
    └── embedding_viz.py         # ✅ Embedding visualization

scripts/
└── train_model.py              # ✅ Updated to support model_version parameter

run_stage_a.py                 # ✅ Updated with --model_version flag
run_stage_b_intermediate.py    # ✅ Updated with --model_version flag
run_stage_b_full.py            # ✅ Updated with --model_version flag
```

### Usage

**Train Model 2A** (Stage A + embeddings):
```bash
python run_stage_a.py --model_version 2 --threshold_metric f1 --embedding_dim 16
```

**Train Model 2B-intermediate**:
```bash
python run_stage_b_intermediate.py --model_version 2
```

**Train Model 2B-full**:
```bash
python run_stage_b_full.py --model_version 2
```

**CLI Arguments**:
- `--model_version`: 1 (original MLP) or 2 (with embeddings), default=1
- `--embedding_dim`: Embedding dimension for Model 2, default=16
- `--threshold_metric`: f1, accuracy, or balanced_accuracy, default=f1

### Backward Compatibility

✅ **Model 1 remains unchanged**:
- Default behavior: `--model_version 1`
- All existing scripts work without modification
- Archived Model 1 runs are preserved
- No breaking changes to existing pipelines

---

## Team Embedding Visualization

### What Gets Visualized

After training Model 2, the pipeline automatically generates:

1. **2D Embedding Plot** (`team_embeddings_2d.png`):
   - Each point = one NBA team
   - Points labeled with team abbreviation
   - Similar teams cluster together
   - Generated using PCA (Principal Component Analysis)

2. **Embedding Data** (`team_embeddings_raw.csv`):
   - Full embedding vectors for each team
   - 2D coordinates for plotting
   - Can be used for further analysis

### How It Works

1. **Training**: Model learns 16-dimensional embedding for each team
2. **Extraction**: After training, embedding weights extracted from model
3. **Dimensionality Reduction**: PCA reduces 16D → 2D for visualization
4. **Plotting**: Scatter plot with team labels
5. **Archiving**: PNG + CSV saved with run results

### Interpreting the Plot

**What to look for**:
- **Clusters**: Teams with similar styles/strength group together
- **Distance**: Closer teams = more similar embeddings
- **Outliers**: Teams with unique characteristics
- **Axes**: Principal components capture major variations

**Example Patterns**:
- Strong teams (e.g., GSW, LAL) might cluster
- Weak teams might cluster separately
- Defensive-focused teams might be near each other
- High-pace teams vs slow-pace teams might separate

**Important Notes**:
- Embeddings are learned from **win patterns**, not stats
- 2D is a simplified view (original is 16D)
- Patterns emerge from training data structure
- Different runs may have different orientations (PCA rotation)

### Customization

To use t-SNE instead of PCA:
```python
# In src/utils/embedding_viz.py
visualize_model_2_embeddings(
    model=model,
    team_to_id=team_to_id,
    reduction_method='tsne',  # Change from 'pca' to 'tsne'
    ...
)
```

---

## Model 2 Variants

### Model 2A: Stage A + Embeddings
- **Data**: Historical game features only (no team stats)
- **Features**: ~50 numeric + team embeddings
- **Expected AUC**: 0.69-0.71
- **Use Case**: Baseline for embedding effectiveness

### Model 2B-intermediate: B1 Intermediate + Embeddings
- **Data**: Historical + selected rolling stats
- **Features**: ~40-60 numeric + team embeddings
- **Expected AUC**: 0.68-0.70
- **Use Case**: Balance between features and embeddings

### Model 2B-full: B1 Full + Embeddings
- **Data**: Historical + full rolling stats
- **Features**: ~177 numeric + team embeddings
- **Expected AUC**: 0.70-0.72
- **Use Case**: Maximum information utilization

---

## Next Steps

1. ✅ **Complete documentation** (this file)
2. ✅ **Create directory structure** (`src/models/`, `src/utils/`)
3. ✅ **Implement team embeddings** (`model_2.py`, `team_encoding.py`)
4. ✅ **Implement embedding visualization** (`embedding_viz.py`)
5. ✅ **Update all run scripts** (A1, B1-intermediate, B1-full)
6. 📋 **Create unit tests** (test_model2_forward.py, etc.)
7. 📋 **Train and evaluate Model 2A**
8. 📋 **Train and evaluate Model 2B variants**
9. 📋 **Compare with Model 1 baseline**
10. 📋 **Document results** (`model_2_results.md`)

---

## References

**Embeddings**:
- Guo & Berkhahn (2016): "Entity Embeddings of Categorical Variables"
- Recommended embedding dimension: `min(50, (num_categories + 1) // 2)`

**Deep Learning Best Practices**:
- Batch normalization: Ioffe & Szegedy (2015)
- Dropout: Srivastava et al. (2014)
- Adam optimizer: Kingma & Ba (2015)

**NBA Prediction Literature**:
- Typical accuracy ceiling: 60-70%
- Best published models: 65-72% test accuracy
- Our current: 63% (competitive with literature)

---

**Document Status**: ✅ Implementation Complete  
**Last Updated**: December 12, 2025  
**Next Review**: After Model 2 evaluation
