"""
Threshold Comparison Script for NBA Game Predictions

This script analyzes and compares results from different threshold optimization strategies
across all pipeline stages (A1, B1-intermediate, B1-full).

It reads archived results and generates a comprehensive comparison document showing:
- Optimal thresholds selected
- Test set performance metrics (accuracy, F1, AUC, log loss)
- Confusion matrices
- Differences between threshold strategies

Usage:
    python compare_thresholds.py
    
Output:
    docs/threshold_comparison.md - Comprehensive comparison report
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np


def find_latest_archives_by_threshold(stage_name: str) -> Dict[str, Path]:
    """
    Find the latest archive for each threshold metric for a given stage.
    
    Args:
        stage_name: Name of the stage (e.g., 'stage_a1', 'stage_b1_intermediate')
        
    Returns:
        Dictionary mapping threshold_metric to archive path
    """
    archives_dir = Path("archives") / stage_name
    if not archives_dir.exists():
        return {}
    
    # Group archives by threshold metric
    archives_by_metric = {}
    
    for archive_dir in sorted(archives_dir.iterdir(), reverse=True):
        if not archive_dir.is_dir():
            continue
            
        results_file = archive_dir / "results.json"
        if not results_file.exists():
            continue
            
        # Read results to get threshold metric
        with open(results_file, 'r') as f:
            results = json.load(f)
            threshold_metric = results.get('threshold_metric', 'f1')
            
        # Keep the latest (most recent) archive for each threshold metric
        if threshold_metric not in archives_by_metric:
            archives_by_metric[threshold_metric] = archive_dir
    
    return archives_by_metric


def load_results_from_archive(archive_path: Path) -> Optional[Dict]:
    """Load results.json from an archive directory."""
    results_file = archive_path / "results.json"
    if not results_file.exists():
        return None
        
    with open(results_file, 'r') as f:
        return json.load(f)


def format_confusion_matrix(metrics: Dict) -> str:
    """Format confusion matrix for markdown display."""
    cm = metrics.get('confusion_matrix', [[0, 0], [0, 0]])
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    
    lines = [
        "```",
        "             Predicted",
        "             0      1",
        "Actual  0  | {:4d}  {:4d}".format(tn, fp),
        "        1  | {:4d}  {:4d}".format(fn, tp),
        "```"
    ]
    return "\n".join(lines)


def generate_stage_comparison(stage_name: str, stage_display: str) -> List[str]:
    """
    Generate comparison section for a specific stage.
    
    Args:
        stage_name: Internal name (e.g., 'stage_a1')
        stage_display: Display name (e.g., 'Stage A1')
        
    Returns:
        List of markdown lines
    """
    lines = [
        f"## {stage_display}",
        ""
    ]
    
    # Find latest archives for each threshold metric
    archives = find_latest_archives_by_threshold(stage_name)
    
    if not archives:
        lines.extend([
            "*No archived results found for this stage.*",
            ""
        ])
        return lines
    
    # Load results for each threshold metric
    results_by_metric = {}
    for threshold_metric, archive_path in archives.items():
        results = load_results_from_archive(archive_path)
        if results:
            results_by_metric[threshold_metric] = {
                'results': results,
                'archive_path': archive_path
            }
    
    if not results_by_metric:
        lines.extend([
            "*Unable to load results for this stage.*",
            ""
        ])
        return lines
    
    # Generate comparison table
    lines.extend([
        "### Results Comparison",
        "",
        "| Threshold Metric | Optimal Threshold | Test Accuracy | Test F1 | Test AUC | Test Log Loss |",
        "|-----------------|-------------------|---------------|---------|----------|---------------|"
    ])
    
    for threshold_metric in sorted(results_by_metric.keys()):
        data = results_by_metric[threshold_metric]
        results = data['results']
        test_metrics = results.get('test_metrics', {})
        optimal_threshold = results.get('optimal_threshold', 0.5)
        
        lines.append(
            f"| **{threshold_metric}** | "
            f"{optimal_threshold:.3f} | "
            f"{test_metrics.get('accuracy', 0):.4f} | "
            f"{test_metrics.get('f1_score', 0):.4f} | "
            f"{test_metrics.get('auc', 0):.4f} | "
            f"{test_metrics.get('log_loss', 0):.4f} |"
        )
    
    lines.extend(["", ""])
    
    # Add confusion matrices for each threshold metric
    lines.extend([
        "### Confusion Matrices (Test Set)",
        ""
    ])
    
    for threshold_metric in sorted(results_by_metric.keys()):
        data = results_by_metric[threshold_metric]
        results = data['results']
        test_metrics = results.get('test_metrics', {})
        
        lines.extend([
            f"#### {threshold_metric.upper()}",
            "",
            format_confusion_matrix(test_metrics),
            ""
        ])
    
    # Add archive information
    lines.extend([
        "### Archive Information",
        ""
    ])
    
    for threshold_metric in sorted(results_by_metric.keys()):
        data = results_by_metric[threshold_metric]
        archive_path = data['archive_path']
        results = data['results']
        timestamp = results.get('timestamp', 'Unknown')
        
        lines.append(f"- **{threshold_metric}**: `{archive_path.name}` (trained: {timestamp})")
    
    lines.extend(["", "---", ""])
    
    return lines


def calculate_metric_differences(results_by_stage: Dict) -> List[str]:
    """
    Calculate and format differences between F1 and accuracy optimization.
    
    Args:
        results_by_stage: Dictionary mapping stage names to their results
        
    Returns:
        List of markdown lines
    """
    lines = [
        "## Key Findings",
        "",
        "### Metric Differences (Accuracy-optimized vs F1-optimized)",
        "",
        "Positive values indicate that accuracy-optimized threshold performed better.",
        "",
        "| Stage | Δ Accuracy | Δ F1 Score | Δ AUC | Δ Log Loss | Threshold Shift |",
        "|-------|-----------|-----------|-------|-----------|----------------|"
    ]
    
    stage_names = {
        'stage_a1': 'Stage A1',
        'stage_b1_intermediate': 'Stage B1-Int',
        'stage_b1_full': 'Stage B1-Full'
    }
    
    for stage_key, stage_display in stage_names.items():
        stage_data = results_by_stage.get(stage_key, {})
        
        f1_results = stage_data.get('f1', {}).get('results', {}).get('test_metrics', {})
        acc_results = stage_data.get('accuracy', {}).get('results', {}).get('test_metrics', {})
        
        f1_threshold = stage_data.get('f1', {}).get('results', {}).get('optimal_threshold', 0.5)
        acc_threshold = stage_data.get('accuracy', {}).get('results', {}).get('optimal_threshold', 0.5)
        
        if not f1_results or not acc_results:
            lines.append(f"| {stage_display} | N/A | N/A | N/A | N/A | N/A |")
            continue
        
        delta_accuracy = acc_results.get('accuracy', 0) - f1_results.get('accuracy', 0)
        delta_f1 = acc_results.get('f1_score', 0) - f1_results.get('f1_score', 0)
        delta_auc = acc_results.get('auc', 0) - f1_results.get('auc', 0)
        delta_logloss = acc_results.get('log_loss', 0) - f1_results.get('log_loss', 0)
        threshold_shift = acc_threshold - f1_threshold
        
        lines.append(
            f"| {stage_display} | "
            f"{delta_accuracy:+.4f} | "
            f"{delta_f1:+.4f} | "
            f"{delta_auc:+.4f} | "
            f"{delta_logloss:+.4f} | "
            f"{threshold_shift:+.3f} |"
        )
    
    lines.extend(["", ""])
    
    return lines


def generate_comparison_report():
    """Generate comprehensive threshold comparison report."""
    
    print("="*80)
    print("THRESHOLD COMPARISON ANALYSIS")
    print("="*80)
    print()
    
    # Header
    lines = [
        "# Threshold Optimization Strategy Comparison",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "This report compares model performance across different threshold optimization strategies:",
        "- **F1-optimized**: Threshold selected to maximize F1 score on validation set (default)",
        "- **Accuracy-optimized**: Threshold selected to maximize accuracy on validation set",
        "",
        "**Note:** All models use the same trained weights. Only the decision threshold differs.",
        "",
        "---",
        ""
    ]
    
    # Process each stage
    stages = [
        ('stage_a1', 'Stage A1 - Historical Features Only'),
        ('stage_b1_intermediate', 'Stage B1 Intermediate - Selected Rolling Metrics'),
        ('stage_b1_full', 'Stage B1 Full - Complete Feature Set')
    ]
    
    # Collect all results for cross-stage analysis
    results_by_stage = {}
    
    for stage_name, stage_display in stages:
        print(f"Processing {stage_display}...")
        stage_lines = generate_stage_comparison(stage_name, stage_display)
        lines.extend(stage_lines)
        
        # Collect results for key findings
        archives = find_latest_archives_by_threshold(stage_name)
        stage_results = {}
        for threshold_metric, archive_path in archives.items():
            results = load_results_from_archive(archive_path)
            if results:
                stage_results[threshold_metric] = {
                    'results': results,
                    'archive_path': archive_path
                }
        results_by_stage[stage_name] = stage_results
    
    # Add key findings section
    if any(results_by_stage.values()):
        findings_lines = calculate_metric_differences(results_by_stage)
        lines.extend(findings_lines)
    
    # Add interpretation guidance
    lines.extend([
        "## Interpretation Guide",
        "",
        "### Threshold Selection",
        "- **Lower thresholds** (< 0.5): Model predicts class 1 more frequently",
        "- **Higher thresholds** (> 0.5): Model is more conservative, requires higher confidence",
        "- **F1-optimized**: Balances precision and recall",
        "- **Accuracy-optimized**: Maximizes overall correct predictions",
        "",
        "### When to Use Each Strategy",
        "- **F1 optimization**: When false positives and false negatives have similar costs",
        "- **Accuracy optimization**: When overall correctness matters most",
        "- **Custom threshold**: Set manually based on domain requirements and cost analysis",
        "",
        "### AUC vs Threshold",
        "- **AUC** is threshold-independent (same for all strategies with same model weights)",
        "- Small AUC differences indicate numerical/random variation, not meaningful differences",
        "",
        "---",
        "",
        "*This report was automatically generated by `compare_thresholds.py`*"
    ])
    
    # Write report
    output_path = Path("docs") / "threshold_comparison.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print()
    print(f"✓ Comparison report saved to: {output_path}")
    print()
    print("="*80)
    print("COMPARISON COMPLETE")
    print("="*80)


if __name__ == "__main__":
    generate_comparison_report()
