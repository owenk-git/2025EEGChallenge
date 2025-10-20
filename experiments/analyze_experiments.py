#!/usr/bin/env python3
"""
Analyze experiment results and suggest improvements.

Usage:
    python experiments/analyze_experiments.py
    python experiments/analyze_experiments.py --challenge 1
    python experiments/analyze_experiments.py --best 5
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict


def load_experiments() -> List[Dict]:
    """Load all experiments from JSON log"""
    json_path = Path(__file__).parent / "experiments.json"
    if not json_path.exists():
        print("No experiments found. Run training with --num X to log experiments.")
        return []

    with open(json_path, 'r') as f:
        return json.load(f)


def filter_experiments(experiments: List[Dict], challenge: int = None) -> List[Dict]:
    """Filter experiments by challenge"""
    if challenge is None:
        return experiments
    return [exp for exp in experiments if exp['challenge'] == challenge]


def print_summary(experiments: List[Dict]):
    """Print experiment summary table"""
    if not experiments:
        print("No experiments to display.")
        return

    print("\n" + "="*100)
    print("EXPERIMENT SUMMARY")
    print("="*100)

    # Header
    print(f"{'Exp':<5} {'Challenge':<10} {'Subjects':<10} {'Epochs':<8} {'LR':<10} {'Dropout':<10} {'Val Loss':<12} {'Best Epoch':<12}")
    print("-"*100)

    # Rows
    for exp in experiments:
        exp_num = exp['exp_num']
        challenge = f"C{exp['challenge']}"
        subjects = exp['config'].get('max_subjects', 'All')
        epochs = exp['config']['epochs']
        lr = exp['config']['lr']
        dropout = exp['config']['dropout']
        val_loss = exp['results']['final_val_loss']
        best_epoch = exp['results']['best_epoch']

        print(f"{exp_num:<5} {challenge:<10} {str(subjects):<10} {epochs:<8} {lr:<10.1e} {dropout:<10.2f} {val_loss:<12.4f} {best_epoch:<12}")

    print("="*100)


def find_best_experiments(experiments: List[Dict], n: int = 5, challenge: int = None) -> List[Dict]:
    """Find top N best experiments"""
    filtered = filter_experiments(experiments, challenge)
    sorted_exps = sorted(filtered, key=lambda x: x['results']['final_val_loss'])
    return sorted_exps[:n]


def analyze_trends(experiments: List[Dict]):
    """Analyze trends in experiments"""
    if len(experiments) < 2:
        print("\nNeed at least 2 experiments for trend analysis.")
        return

    print("\n" + "="*100)
    print("TREND ANALYSIS")
    print("="*100)

    # Group by challenge
    c1_exps = filter_experiments(experiments, challenge=1)
    c2_exps = filter_experiments(experiments, challenge=2)

    if c1_exps:
        print("\n📊 Challenge 1 Trends:")
        analyze_challenge_trends(c1_exps)

    if c2_exps:
        print("\n📊 Challenge 2 Trends:")
        analyze_challenge_trends(c2_exps)


def analyze_challenge_trends(experiments: List[Dict]):
    """Analyze trends for a specific challenge"""
    # Best performer
    best = min(experiments, key=lambda x: x['results']['final_val_loss'])
    print(f"   Best: Exp #{best['exp_num']} - Loss: {best['results']['final_val_loss']:.4f}")

    # Average performance
    avg_loss = sum(exp['results']['final_val_loss'] for exp in experiments) / len(experiments)
    print(f"   Average Loss: {avg_loss:.4f}")

    # Correlation with hyperparameters
    analyze_hyperparameter_correlation(experiments, 'lr', 'Learning Rate')
    analyze_hyperparameter_correlation(experiments, 'dropout', 'Dropout')
    analyze_hyperparameter_correlation(experiments, 'max_subjects', 'Max Subjects')
    analyze_hyperparameter_correlation(experiments, 'epochs', 'Epochs')


def analyze_hyperparameter_correlation(experiments: List[Dict], param: str, param_name: str):
    """Analyze correlation between hyperparameter and performance"""
    # Group by parameter value
    param_groups = {}
    for exp in experiments:
        value = exp['config'].get(param)
        if value is None:
            continue

        if value not in param_groups:
            param_groups[value] = []
        param_groups[value].append(exp['results']['final_val_loss'])

    if len(param_groups) > 1:
        print(f"\n   {param_name} Impact:")
        for value in sorted(param_groups.keys()):
            avg_loss = sum(param_groups[value]) / len(param_groups[value])
            count = len(param_groups[value])
            print(f"      {value}: {avg_loss:.4f} (n={count})")


def suggest_next_experiments(experiments: List[Dict]):
    """Suggest next experiments based on trends"""
    if not experiments:
        print("\n💡 Suggestion: Start with baseline experiment (Exp #1)")
        return

    print("\n" + "="*100)
    print("SUGGESTIONS FOR NEXT EXPERIMENTS")
    print("="*100)

    # Find best configurations
    c1_exps = filter_experiments(experiments, challenge=1)
    c2_exps = filter_experiments(experiments, challenge=2)

    if c1_exps:
        best_c1 = min(c1_exps, key=lambda x: x['results']['final_val_loss'])
        print(f"\n✅ Best C1: Exp #{best_c1['exp_num']} - Loss: {best_c1['results']['final_val_loss']:.4f}")
        print("   Config:", json.dumps(best_c1['config'], indent=6))

    if c2_exps:
        best_c2 = min(c2_exps, key=lambda x: x['results']['final_val_loss'])
        print(f"\n✅ Best C2: Exp #{best_c2['exp_num']} - Loss: {best_c2['results']['final_val_loss']:.4f}")
        print("   Config:", json.dumps(best_c2['config'], indent=6))

    print("\n💡 Recommendations:")
    print("   1. Try higher max_subjects (200+) if performance is improving with more data")
    print("   2. Experiment with dropout (0.1-0.3) if overfitting is observed")
    print("   3. Adjust learning rate (1e-4 to 5e-3) based on loss curves")
    print("   4. Consider longer training (150+ epochs) if still improving")
    print("   5. Test data augmentation strategies (see docs/strategies/)")
    print("\n" + "="*100)


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument('-c', '--challenge', type=int, choices=[1, 2],
                        help='Filter by challenge')
    parser.add_argument('--best', type=int, default=5,
                        help='Show top N best experiments')
    args = parser.parse_args()

    # Load experiments
    experiments = load_experiments()

    if not experiments:
        return

    # Print summary
    print_summary(experiments)

    # Show best experiments
    best_exps = find_best_experiments(experiments, n=args.best, challenge=args.challenge)
    if best_exps:
        print(f"\n🏆 Top {len(best_exps)} Best Experiments:")
        for i, exp in enumerate(best_exps, 1):
            print(f"   {i}. Exp #{exp['exp_num']} (C{exp['challenge']}) - Loss: {exp['results']['final_val_loss']:.4f}")

    # Analyze trends
    analyze_trends(experiments)

    # Suggestions
    suggest_next_experiments(experiments)


if __name__ == "__main__":
    main()
