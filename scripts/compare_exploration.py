#!/usr/bin/env python3
"""
Compare exploration experiments (Exp 1-10) to find best direction

Usage:
    python scripts/compare_exploration.py
    python scripts/compare_exploration.py --plot
"""

import json
import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


def load_experiments():
    """Load experiments from JSON"""
    json_path = Path("experiments/experiments.json")
    if not json_path.exists():
        print("❌ No experiments found!")
        print("   Run some experiments first:")
        print("   python train.py -c 1 -d dummy -o --max 50 -e 50 --num 1")
        return []

    with open(json_path, 'r') as f:
        experiments = json.load(f)

    # Filter exploration experiments (num 1-10)
    explorations = [exp for exp in experiments if 1 <= exp.get('exp_num', 0) <= 10]
    return sorted(explorations, key=lambda x: x['exp_num'])


def print_exploration_summary(experiments):
    """Print summary table of explorations"""
    if not experiments:
        print("No exploration experiments found (exp_num 1-10)")
        return

    print("\n" + "="*100)
    print("🔬 EXPLORATION RESULTS (Exp 1-10)")
    print("="*100)

    # Header
    print(f"{'Exp':<5} {'Challenge':<10} {'Group':<15} {'Subjects':<10} {'Epochs':<8} "
          f"{'Dropout':<10} {'LR':<10} {'Batch':<8} {'Val NRMSE':<12} {'Best Epoch':<12}")
    print("-"*100)

    # Group names
    groups = {
        (1, 2): "Baseline",
        (3, 4): "More Data",
        (5, 6): "High Dropout",
        (7, 8): "Lower LR",
        (9, 10): "Large Batch"
    }

    # Print rows
    for exp in experiments:
        exp_num = exp['exp_num']
        challenge = f"C{exp['challenge']}"
        subjects = exp['config'].get('max_subjects', 'All')
        epochs = exp['config']['epochs']
        dropout = exp['config']['dropout']
        lr = exp['config']['lr']
        batch = exp['config'].get('batch_size', 32)
        nrmse = exp['results'].get('best_val_nrmse', 0)
        best_epoch = exp['results']['best_epoch']

        # Find group
        group = ""
        for (start, end), name in groups.items():
            if start <= exp_num <= end:
                group = name
                break

        print(f"{exp_num:<5} {challenge:<10} {group:<15} {str(subjects):<10} {epochs:<8} "
              f"{dropout:<10.2f} {lr:<10.1e} {batch:<8} {nrmse:<12.4f} {best_epoch:<12}")

    print("="*100)


def analyze_best_direction(experiments):
    """Analyze which direction worked best"""
    if len(experiments) < 4:
        print("\n⚠️  Need at least 4 experiments to analyze directions")
        return

    print("\n" + "="*100)
    print("📊 DIRECTION ANALYSIS")
    print("="*100)

    # Group experiments
    groups = {
        "Baseline (50 subj)": [exp for exp in experiments if exp['exp_num'] in [1, 2]],
        "More Data (200 subj)": [exp for exp in experiments if exp['exp_num'] in [3, 4]],
        "High Dropout (0.4)": [exp for exp in experiments if exp['exp_num'] in [5, 6]],
        "Lower LR (5e-4)": [exp for exp in experiments if exp['exp_num'] in [7, 8]],
        "Large Batch (64)": [exp for exp in experiments if exp['exp_num'] in [9, 10]],
    }

    baseline_avg = None
    best_direction = None
    best_improvement = 0

    for group_name, group_exps in groups.items():
        if not group_exps:
            continue

        # Average NRMSE for this group
        avg_nrmse = sum(exp['results'].get('best_val_nrmse', 999) for exp in group_exps) / len(group_exps)

        # Calculate combined score (0.3 * C1 + 0.7 * C2)
        c1_nrmse = next((exp['results'].get('best_val_nrmse', 999) for exp in group_exps if exp['challenge'] == 1), None)
        c2_nrmse = next((exp['results'].get('best_val_nrmse', 999) for exp in group_exps if exp['challenge'] == 2), None)

        combined = None
        if c1_nrmse and c2_nrmse:
            combined = 0.3 * c1_nrmse + 0.7 * c2_nrmse

        print(f"\n{group_name}:")
        print(f"  Average Val NRMSE: {avg_nrmse:.4f}")
        if combined:
            print(f"  Combined Score: {combined:.4f} (0.3×C1 + 0.7×C2)")

        # Track baseline
        if "Baseline" in group_name:
            baseline_avg = avg_nrmse

        # Track best improvement
        if baseline_avg and "Baseline" not in group_name:
            improvement = ((baseline_avg - avg_nrmse) / baseline_avg) * 100
            print(f"  vs Baseline: {improvement:+.1f}%")

            if improvement > best_improvement:
                best_improvement = improvement
                best_direction = group_name

    # Recommendations
    print("\n" + "="*100)
    print("💡 RECOMMENDATIONS")
    print("="*100)

    if baseline_avg is None:
        print("\n⚠️  Run baseline experiments (Exp 1-2) first!")
        return

    print(f"\nBaseline performance: {baseline_avg:.4f}")

    if best_direction:
        print(f"🏆 Best direction: {best_direction} ({best_improvement:+.1f}% improvement)")

        if "More Data" in best_direction:
            print("\n✅ Direction: DATA QUANTITY")
            print("Next steps:")
            print("  - Use 300-500 subjects")
            print("  - Try full dataset")
            print("  - Longer training (150-200 epochs)")

        elif "High Dropout" in best_direction:
            print("\n✅ Direction: REGULARIZATION")
            print("Next steps:")
            print("  - Test dropout 0.3, 0.5")
            print("  - Add data augmentation")
            print("  - Try label smoothing")

        elif "Lower LR" in best_direction:
            print("\n✅ Direction: OPTIMIZATION")
            print("Next steps:")
            print("  - Test LR 1e-4, 2e-4")
            print("  - Try cosine annealing")
            print("  - Longer training (200+ epochs)")

        elif "Large Batch" in best_direction:
            print("\n✅ Direction: TRAINING DYNAMICS")
            print("Next steps:")
            print("  - Test batch 96, 128")
            print("  - Gradient accumulation")
            print("  - Learning rate warmup")

    else:
        print("\n⚠️  No clear winner. Consider:")
        print("  - All perform similarly → Try different architecture")
        print("  - All perform poorly → Check data loading/preprocessing")
        print("  - Need more experiments to see pattern")

    print("\n" + "="*100)


def plot_results(experiments):
    """Plot exploration results"""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("❌ matplotlib not installed. Install with: pip install matplotlib")
        return

    if not experiments:
        print("No experiments to plot")
        return

    # Group by challenge
    c1_exps = [exp for exp in experiments if exp['challenge'] == 1]
    c2_exps = [exp for exp in experiments if exp['challenge'] == 2]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Challenge 1
    if c1_exps:
        exp_nums = [exp['exp_num'] for exp in c1_exps]
        nrmses = [exp['results'].get('best_val_nrmse', 0) for exp in c1_exps]

        axes[0].bar(exp_nums, nrmses, color='steelblue', alpha=0.7)
        axes[0].axhline(y=1.45, color='red', linestyle='--', label='Current Best (1.45)')
        axes[0].axhline(y=0.978, color='green', linestyle='--', label='SOTA (0.978)')
        axes[0].set_xlabel('Experiment Number')
        axes[0].set_ylabel('Validation NRMSE')
        axes[0].set_title('Challenge 1: Exploration Results')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

    # Challenge 2
    if c2_exps:
        exp_nums = [exp['exp_num'] for exp in c2_exps]
        nrmses = [exp['results'].get('best_val_nrmse', 0) for exp in c2_exps]

        axes[1].bar(exp_nums, nrmses, color='coral', alpha=0.7)
        axes[1].axhline(y=1.01, color='red', linestyle='--', label='Current Best (1.01)')
        axes[1].axhline(y=0.978, color='green', linestyle='--', label='SOTA (0.978)')
        axes[1].set_xlabel('Experiment Number')
        axes[1].set_ylabel('Validation NRMSE')
        axes[1].set_title('Challenge 2: Exploration Results')
        axes[1].legend()
        axes[1].grid(alpha=0.3)

    plt.tight_layout()
    output_path = Path("experiments/exploration_results.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Plot saved to: {output_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Compare exploration experiments")
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    args = parser.parse_args()

    # Load experiments
    experiments = load_experiments()

    if not experiments:
        return

    # Print summary
    print_exploration_summary(experiments)

    # Analyze best direction
    analyze_best_direction(experiments)

    # Plot if requested
    if args.plot:
        plot_results(experiments)


if __name__ == "__main__":
    main()
