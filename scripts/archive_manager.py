"""
Archive manager for NBA game prediction pipeline.

Automatically archives previous results before new training runs.
Enables comparison between different model configurations and runs.
"""

import shutil
from pathlib import Path
from datetime import datetime
import json
import os


def archive_previous_results(
    outputs_dir: str = "outputs",
    archive_base_dir: str = "archives",
    run_suffix: str = ""
) -> str:
    """
    Archive previous results before starting a new training run.
    
    Creates a timestamped archive containing:
    - results.json
    - All plots from outputs/plots/
    - Model metrics and configuration
    
    Args:
        outputs_dir: Directory containing outputs to archive
        archive_base_dir: Base directory for archives
        run_suffix: Optional suffix to append to run name (e.g., "threshold-f1")
        
    Returns:
        Path to the created archive directory, or empty string if nothing to archive
    """
    outputs_path = Path(outputs_dir)
    
    # Check if there are results to archive
    results_file = outputs_path / "results.json"
    if not results_file.exists():
        print("No previous results found to archive.")
        return ""
    
    # Create archive directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}"
    if run_suffix:
        run_name += f"_{run_suffix}"
    archive_path = Path(archive_base_dir) / run_name
    archive_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("ARCHIVING PREVIOUS RESULTS")
    print("="*80)
    
    archived_items = []
    
    # Archive results.json
    if results_file.exists():
        shutil.copy2(results_file, archive_path / "results.json")
        archived_items.append("results.json")
        print(f"✓ Archived: results.json")
    
    # Archive plots directory
    plots_dir = outputs_path / "plots"
    if plots_dir.exists() and plots_dir.is_dir():
        archive_plots_dir = archive_path / "plots"
        shutil.copytree(plots_dir, archive_plots_dir)
        num_plots = len(list(archive_plots_dir.glob("*.png")))
        archived_items.append(f"plots/ ({num_plots} files)")
        print(f"✓ Archived: plots/ ({num_plots} visualizations)")
    
    # Archive model file if it exists
    model_file = Path("models/stage_a1_mlp.keras")
    if model_file.exists():
        archive_models_dir = archive_path / "models"
        archive_models_dir.mkdir(exist_ok=True)
        shutil.copy2(model_file, archive_models_dir / "stage_a1_mlp.keras")
        archived_items.append("models/stage_a1_mlp.keras")
        print(f"✓ Archived: models/stage_a1_mlp.keras")
    
    # Create archive metadata
    metadata = {
        "archive_timestamp": timestamp,
        "archive_date": datetime.now().isoformat(),
        "archived_items": archived_items,
        "original_results": None
    }
    
    # Include summary from results.json
    if results_file.exists():
        with open(results_file, 'r') as f:
            original_results = json.load(f)
            metadata["original_results"] = {
                "timestamp": original_results.get("timestamp"),
                "epochs_trained": original_results.get("epochs_trained"),
                "test_accuracy": original_results.get("test_metrics", {}).get("accuracy"),
                "test_auc": original_results.get("test_metrics", {}).get("auc"),
                "test_f1": original_results.get("test_metrics", {}).get("f1_score"),
                "optimal_threshold": original_results.get("optimal_threshold", 0.5)
            }
    
    # Save metadata
    with open(archive_path / "archive_info.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Archive created: {archive_path}")
    print("="*80)
    
    return str(archive_path)


def list_archives(archive_base_dir: str = "archives") -> list:
    """
    List all available archives with their metadata.
    
    Args:
        archive_base_dir: Base directory containing archives
        
    Returns:
        List of dictionaries containing archive information
    """
    archive_path = Path(archive_base_dir)
    
    if not archive_path.exists():
        return []
    
    archives = []
    for archive_dir in sorted(archive_path.glob("run_*"), reverse=True):
        info_file = archive_dir / "archive_info.json"
        if info_file.exists():
            with open(info_file, 'r') as f:
                metadata = json.load(f)
                metadata['path'] = str(archive_dir)
                archives.append(metadata)
        else:
            # Create basic metadata for old archives without info file
            archives.append({
                'archive_timestamp': archive_dir.name.replace('run_', ''),
                'path': str(archive_dir),
                'archived_items': ['Unknown']
            })
    
    return archives


def compare_archives(archive1_path: str, archive2_path: str = None) -> dict:
    """
    Compare two archived runs or compare archive with current results.
    
    Args:
        archive1_path: Path to first archive or 'latest' for most recent
        archive2_path: Path to second archive or None for current results
        
    Returns:
        Dictionary with comparison metrics
    """
    # Load first archive
    if archive1_path == 'latest':
        archives = list_archives()
        if not archives:
            print("No archives found.")
            return {}
        archive1_path = archives[0]['path']
    
    archive1_results_file = Path(archive1_path) / "results.json"
    if not archive1_results_file.exists():
        print(f"No results.json found in {archive1_path}")
        return {}
    
    with open(archive1_results_file, 'r') as f:
        results1 = json.load(f)
    
    # Load second archive or current results
    if archive2_path is None:
        results2_file = Path("outputs/results.json")
        if not results2_file.exists():
            print("No current results found.")
            return {}
        with open(results2_file, 'r') as f:
            results2 = json.load(f)
        label2 = "Current"
    else:
        archive2_results_file = Path(archive2_path) / "results.json"
        if not archive2_results_file.exists():
            print(f"No results.json found in {archive2_path}")
            return {}
        with open(archive2_results_file, 'r') as f:
            results2 = json.load(f)
        label2 = archive2_path
    
    # Compare metrics
    comparison = {
        'run1': {
            'label': archive1_path,
            'timestamp': results1.get('timestamp'),
            'epochs': results1.get('epochs_trained'),
            'test_accuracy': results1.get('test_metrics', {}).get('accuracy'),
            'test_auc': results1.get('test_metrics', {}).get('auc'),
            'test_f1': results1.get('test_metrics', {}).get('f1_score'),
            'threshold': results1.get('optimal_threshold', 0.5)
        },
        'run2': {
            'label': label2,
            'timestamp': results2.get('timestamp'),
            'epochs': results2.get('epochs_trained'),
            'test_accuracy': results2.get('test_metrics', {}).get('accuracy'),
            'test_auc': results2.get('test_metrics', {}).get('auc'),
            'test_f1': results2.get('test_metrics', {}).get('f1_score'),
            'threshold': results2.get('optimal_threshold', 0.5)
        }
    }
    
    # Calculate differences
    comparison['differences'] = {
        'accuracy_diff': comparison['run2']['test_accuracy'] - comparison['run1']['test_accuracy'],
        'auc_diff': comparison['run2']['test_auc'] - comparison['run1']['test_auc'],
        'f1_diff': comparison['run2']['test_f1'] - comparison['run1']['test_f1'] if comparison['run2']['test_f1'] and comparison['run1']['test_f1'] else None,
        'epochs_diff': comparison['run2']['epochs'] - comparison['run1']['epochs']
    }
    
    return comparison


def print_comparison(comparison: dict):
    """Pretty print comparison between two runs."""
    if not comparison:
        return
    
    print("\n" + "="*80)
    print("COMPARISON ENTRE DEUX RUNS")
    print("="*80)
    
    run1 = comparison['run1']
    run2 = comparison['run2']
    diff = comparison['differences']
    
    print(f"\nRun 1: {run1['timestamp']}")
    print(f"Run 2: {run2['timestamp']}")
    
    print("\n" + "-"*80)
    print(f"{'Métrique':<20} {'Run 1':>15} {'Run 2':>15} {'Différence':>15}")
    print("-"*80)
    
    # Test Accuracy
    print(f"{'Test Accuracy':<20} {run1['test_accuracy']:>15.4f} {run2['test_accuracy']:>15.4f} {diff['accuracy_diff']:>+15.4f}")
    
    # Test AUC
    print(f"{'Test AUC':<20} {run1['test_auc']:>15.4f} {run2['test_auc']:>15.4f} {diff['auc_diff']:>+15.4f}")
    
    # Test F1
    if run1['test_f1'] and run2['test_f1']:
        print(f"{'Test F1':<20} {run1['test_f1']:>15.4f} {run2['test_f1']:>15.4f} {diff['f1_diff']:>+15.4f}")
    
    # Epochs
    print(f"{'Epochs':<20} {run1['epochs']:>15} {run2['epochs']:>15} {diff['epochs_diff']:>+15}")
    
    # Threshold
    print(f"{'Threshold':<20} {run1['threshold']:>15.3f} {run2['threshold']:>15.3f} {run2['threshold']-run1['threshold']:>+15.3f}")
    
    print("-"*80)
    
    # Summary
    print("\n💡 RÉSUMÉ:")
    if diff['accuracy_diff'] > 0.01:
        print(f"   ✅ Amélioration de l'accuracy: +{diff['accuracy_diff']*100:.2f}%")
    elif diff['accuracy_diff'] < -0.01:
        print(f"   ❌ Dégradation de l'accuracy: {diff['accuracy_diff']*100:.2f}%")
    else:
        print(f"   ➖ Accuracy similaire (diff: {diff['accuracy_diff']*100:.2f}%)")
    
    if diff['auc_diff'] > 0.01:
        print(f"   ✅ Amélioration de l'AUC: +{diff['auc_diff']:.4f}")
    elif diff['auc_diff'] < -0.01:
        print(f"   ❌ Dégradation de l'AUC: {diff['auc_diff']:.4f}")
    
    print("="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Archive manager for NBA predictions")
    parser.add_argument('--list', action='store_true', help='List all archives')
    parser.add_argument('--compare', nargs='*', help='Compare archives (latest, path1, path1 path2)')
    parser.add_argument('--archive', action='store_true', help='Archive current results')
    
    args = parser.parse_args()
    
    if args.list:
        archives = list_archives()
        print("\n" + "="*80)
        print(f"ARCHIVES DISPONIBLES ({len(archives)} runs)")
        print("="*80)
        for i, archive in enumerate(archives, 1):
            print(f"\n{i}. {archive['archive_timestamp']}")
            print(f"   Path: {archive['path']}")
            if archive.get('original_results'):
                res = archive['original_results']
                acc = res.get('test_accuracy')
                auc = res.get('test_auc')
                epochs = res.get('epochs_trained')
                if acc is not None:
                    print(f"   Accuracy: {acc:.4f}")
                if auc is not None:
                    print(f"   AUC: {auc:.4f}")
                if epochs is not None:
                    print(f"   Epochs: {epochs}")
    
    elif args.compare is not None:
        if len(args.compare) == 0:
            # Compare latest with current
            comparison = compare_archives('latest')
            print_comparison(comparison)
        elif len(args.compare) == 1:
            # Compare specified archive with current
            comparison = compare_archives(args.compare[0])
            print_comparison(comparison)
        elif len(args.compare) == 2:
            # Compare two archives
            comparison = compare_archives(args.compare[0], args.compare[1])
            print_comparison(comparison)
    
    elif args.archive:
        archive_previous_results()
    
    else:
        parser.print_help()
