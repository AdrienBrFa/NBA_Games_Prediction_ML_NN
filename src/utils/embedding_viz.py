"""
Team embedding visualization for Model 2.

This module provides utilities to visualize learned team embeddings
in 2D space using dimensionality reduction techniques (PCA, t-SNE).

After training Model 2, embeddings can be extracted and visualized to:
- Understand team relationships learned by the model
- Identify clusters of similar teams
- Validate model behavior and interpretability
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, Optional
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import tensorflow as tf
from tensorflow import keras

from src.models.model_2 import extract_team_embeddings


def reduce_embeddings_2d(
    embeddings: np.ndarray,
    method: str = 'pca',
    random_state: int = 42
) -> np.ndarray:
    """
    Reduce team embeddings to 2D using PCA or t-SNE.
    
    Args:
        embeddings: Array of shape (num_teams, embedding_dim)
        method: 'pca' or 'tsne' (default: 'pca')
        random_state: Random seed for reproducibility
        
    Returns:
        Array of shape (num_teams, 2) with 2D coordinates
        
    Example:
        >>> embeddings = np.random.randn(30, 16)
        >>> coords_2d = reduce_embeddings_2d(embeddings, method='pca')
        >>> print(coords_2d.shape)  # (30, 2)
    """
    if method.lower() == 'pca':
        reducer = PCA(n_components=2, random_state=random_state)
        coords_2d = reducer.fit_transform(embeddings)
        print(f"PCA explained variance: {reducer.explained_variance_ratio_.sum():.2%}")
    elif method.lower() == 'tsne':
        reducer = TSNE(
            n_components=2,
            random_state=random_state,
            perplexity=min(30, embeddings.shape[0] - 1),
            n_iter=1000
        )
        coords_2d = reducer.fit_transform(embeddings)
        print("t-SNE reduction complete")
    else:
        raise ValueError(f"Unknown method: {method}. Use 'pca' or 'tsne'.")
    
    return coords_2d


def plot_team_embeddings(
    coords_2d: np.ndarray,
    team_names: list,
    title: str = "Team Embeddings (2D)",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 10),
    show: bool = False
) -> None:
    """
    Create a 2D scatter plot of team embeddings with labels.
    
    Args:
        coords_2d: Array of shape (num_teams, 2) with 2D coordinates
        team_names: List of team names corresponding to coordinates
        title: Plot title
        save_path: Path to save PNG file (optional)
        figsize: Figure size (width, height)
        show: Whether to display plot interactively
        
    Example:
        >>> coords_2d = np.random.randn(30, 2)
        >>> teams = ['LAL', 'GSW', 'BOS', ...]
        >>> plot_team_embeddings(coords_2d, teams, save_path='outputs/embeddings.png')
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Scatter plot
    scatter = ax.scatter(
        coords_2d[:, 0],
        coords_2d[:, 1],
        c=range(len(team_names)),
        cmap='tab20',
        s=200,
        alpha=0.7,
        edgecolors='black',
        linewidth=1.5
    )
    
    # Add team labels
    for i, team in enumerate(team_names):
        ax.annotate(
            team,
            (coords_2d[i, 0], coords_2d[i, 1]),
            fontsize=9,
            fontweight='bold',
            ha='center',
            va='center'
        )
    
    ax.set_xlabel('Component 1', fontsize=12)
    ax.set_ylabel('Component 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Embedding plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def save_embeddings_csv(
    embeddings: np.ndarray,
    team_names: list,
    coords_2d: np.ndarray,
    save_path: Path
) -> None:
    """
    Save team embeddings and 2D coordinates to CSV.
    
    Args:
        embeddings: Full embedding vectors of shape (num_teams, embedding_dim)
        team_names: List of team names
        coords_2d: 2D coordinates of shape (num_teams, 2)
        save_path: Path to save CSV file
        
    Example:
        >>> embeddings = np.random.randn(30, 16)
        >>> coords_2d = np.random.randn(30, 2)
        >>> teams = ['LAL', 'GSW', ...]
        >>> save_embeddings_csv(embeddings, teams, coords_2d, 'outputs/embeddings.csv')
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create DataFrame with team names, 2D coords, and full embeddings
    data = {
        'team': team_names,
        'x_2d': coords_2d[:, 0],
        'y_2d': coords_2d[:, 1]
    }
    
    # Add embedding dimensions
    embedding_dim = embeddings.shape[1]
    for i in range(embedding_dim):
        data[f'emb_dim_{i}'] = embeddings[:, i]
    
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)
    print(f"Embedding data saved to {save_path}")


def visualize_model_2_embeddings(
    model: keras.Model,
    team_to_id: Dict[str, int],
    stage_name: str = "Model 2",
    embedding_dim: int = 16,
    threshold_metric: str = "f1",
    output_dir: Path = Path("outputs"),
    reduction_method: str = 'pca',
    show: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract, reduce, and visualize team embeddings from trained Model 2.
    
    This is the main function to call after training Model 2.
    
    Args:
        model: Trained Model 2 instance
        team_to_id: Team name to ID mapping (from team encoding)
        stage_name: Name of model variant (e.g., "2A", "2B-inter", "2B-full")
        embedding_dim: Embedding dimension used in training
        threshold_metric: Threshold metric used ("f1", "accuracy", etc.)
        output_dir: Directory to save outputs
        reduction_method: 'pca' or 'tsne' (default: 'pca')
        show: Whether to display plot interactively
        
    Returns:
        Tuple of (full_embeddings, coords_2d)
        
    Example:
        >>> from src.utils.team_encoding import load_team_encoder
        >>> team_to_id = load_team_encoder('outputs/team_encoding.json')
        >>> visualize_model_2_embeddings(
        ...     model=trained_model,
        ...     team_to_id=team_to_id,
        ...     stage_name="2A",
        ...     output_dir=Path("outputs")
        ... )
    """
    print("\n" + "="*60)
    print("Visualizing Team Embeddings")
    print("="*60)
    
    # Extract embeddings from model
    print("Extracting embeddings from model...")
    full_embeddings = extract_team_embeddings(model).numpy()
    
    # Get team names in order of IDs (skip UNK at index 0)
    id_to_team = {v: k for k, v in team_to_id.items()}
    team_names = [id_to_team[i] for i in range(1, len(team_to_id))]
    embeddings_without_unk = full_embeddings[1:, :]  # Skip UNK token
    
    print(f"Embedding shape: {embeddings_without_unk.shape}")
    print(f"Number of teams: {len(team_names)}")
    
    # Reduce to 2D
    print(f"Reducing embeddings to 2D using {reduction_method.upper()}...")
    coords_2d = reduce_embeddings_2d(
        embeddings_without_unk,
        method=reduction_method,
        random_state=42
    )
    
    # Create plot title
    title = (
        f"Team Embeddings - {stage_name}\n"
        f"(dim={embedding_dim}, threshold={threshold_metric}, method={reduction_method.upper()})"
    )
    
    # Plot embeddings
    print("Generating visualization...")
    plot_path = output_dir / "team_embeddings_2d.png"
    plot_team_embeddings(
        coords_2d=coords_2d,
        team_names=team_names,
        title=title,
        save_path=plot_path,
        show=show
    )
    
    # Save raw embeddings and 2D coordinates to CSV
    print("Saving embedding data...")
    csv_path = output_dir / "team_embeddings_raw.csv"
    save_embeddings_csv(
        embeddings=embeddings_without_unk,
        team_names=team_names,
        coords_2d=coords_2d,
        save_path=csv_path
    )
    
    print("="*60)
    print("✓ Embedding visualization complete!")
    print("="*60 + "\n")
    
    return full_embeddings, coords_2d


if __name__ == '__main__':
    # Simple test with synthetic data
    print("Testing embedding visualization with synthetic data...")
    
    # Create synthetic embeddings (30 teams, 16 dimensions)
    np.random.seed(42)
    synthetic_embeddings = np.random.randn(30, 16)
    
    # Create fake team names
    team_names = [f"TEAM{i:02d}" for i in range(1, 31)]
    
    # Test reduction
    print("\nTesting PCA reduction...")
    coords_pca = reduce_embeddings_2d(synthetic_embeddings, method='pca')
    
    # Test plotting
    print("\nTesting plot generation...")
    plot_team_embeddings(
        coords_2d=coords_pca,
        team_names=team_names,
        title="Test Team Embeddings (Synthetic Data)",
        save_path=Path("test_embeddings.png"),
        show=False
    )
    
    # Test CSV export
    print("\nTesting CSV export...")
    save_embeddings_csv(
        embeddings=synthetic_embeddings,
        team_names=team_names,
        coords_2d=coords_pca,
        save_path=Path("test_embeddings.csv")
    )
    
    print("\n✓ All tests passed!")
    print("  - Generated: test_embeddings.png")
    print("  - Generated: test_embeddings.csv")
