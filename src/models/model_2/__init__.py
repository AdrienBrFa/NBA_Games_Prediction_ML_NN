"""
Model 2: Enhanced MLP with embeddings and deeper architecture.

This package contains the implementation of Model 2, which improves upon
Model 1 (simple MLP) by introducing:
- Team embeddings (learned representations for 30 NBA teams)
- Season embeddings (optional - learned representations for different eras)
- Deeper architecture (5-6 layers with batch normalization)
- Advanced regularization (L1+L2, tuned dropout)
- Improved training (gradient clipping, learning rate scheduling)

Modules:
- embeddings.py: Embedding layer utilities for teams and seasons
- architecture.py: Model 2 architecture definition
- train.py: Training loop with enhanced features
- config.py: Hyperparameters and configuration

Variants:
- Model 2A: Baseline B1 data + new architecture
- Model 2B-intermediate: B1 Intermediate data + new architecture
- Model 2B-full: B1 Full data + new architecture

Expected performance: +2-4% AUC over Model 1 variants
"""

__version__ = "2.0.0"
__author__ = "NBA Prediction Project"

# Future imports will go here after implementation
# from .embeddings import create_team_embeddings, create_season_embeddings
# from .architecture import build_model_2
# from .train import train_model_2
# from .config import MODEL_2_CONFIG
