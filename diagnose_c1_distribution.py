#!/usr/bin/env python3
"""
Diagnose C1 Training Data Distribution

Check if narrow predictions are correct or if model collapsed
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

def diagnose_training_distribution():
    """Check training RT distribution"""

    print("="*80)
    print("C1 TRAINING DATA DISTRIBUTION ANALYSIS")
    print("="*80)

    # Load training data
    print("\n📊 Loading training data...")
    from data.trial_level_loader import TrialLevelDataset

    try:
        dataset = TrialLevelDataset(challenge='c1', mini=False)
        print(f"✅ Loaded {len(dataset)} trials\n")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return

    # Collect RTs
    print("Extracting RT values...")
    rts = []
    for i in range(len(dataset)):
        try:
            _, rt, _ = dataset[i]
            rts.append(rt)
        except:
            continue

    rts = np.array(rts)

    # Statistics
    print(f"\n{'='*80}")
    print("TRAINING RT STATISTICS (Normalized [0,1])")
    print(f"{'='*80}")
    print(f"Total trials: {len(rts)}")
    print(f"Mean: {rts.mean():.4f}")
    print(f"Std: {rts.std():.4f}")
    print(f"Min: {rts.min():.4f}")
    print(f"Max: {rts.max():.4f}")
    print(f"\nPercentiles:")
    print(f"  5th:  {np.percentile(rts, 5):.4f}")
    print(f"  25th: {np.percentile(rts, 25):.4f}")
    print(f"  50th: {np.percentile(rts, 50):.4f}")
    print(f"  75th: {np.percentile(rts, 75):.4f}")
    print(f"  95th: {np.percentile(rts, 95):.4f}")

    # Distribution
    bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    print(f"\nDistribution:")
    for low, high in bins:
        count = np.sum((rts >= low) & (rts < high))
        pct = 100 * count / len(rts)
        print(f"  [{low:.1f}, {high:.1f}): {count:>6} ({pct:>5.1f}%)")

    # Test predictions comparison
    print(f"\n{'='*80}")
    print("COMPARISON WITH TEST PREDICTIONS")
    print(f"{'='*80}")
    print(f"Training data range: [{rts.min():.4f}, {rts.max():.4f}] (span: {rts.max()-rts.min():.4f})")
    print(f"Test predictions:     [0.7000, 0.8200] (span: 0.1200)")
    print(f"\nTest predictions use {100*0.12/(rts.max()-rts.min()):.1f}% of training range")

    # Diagnosis
    print(f"\n{'='*80}")
    print("DIAGNOSIS")
    print(f"{'='*80}")

    if rts.std() < 0.15:
        print("🟡 Training data has LOW variance")
        print("   → Model predictions are reasonable")
        print("   → Test distribution might actually be narrow")
    elif rts.min() < 0.3 and rts.max() > 0.7:
        print("🔴 Training data has GOOD variance")
        print("   → Model predictions COLLAPSED")
        print("   → Need to expand predictions!")
        print("\n✅ RECOMMENDED: Use temperature scaling T=1.5-2.0")
    else:
        print("🟡 Training data has MODERATE variance")
        print("   → Try temperature scaling T=1.3-1.5")

    # Create histogram
    print(f"\n📊 Creating histogram...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Training distribution
    axes[0].hist(rts, bins=50, alpha=0.7, edgecolor='black')
    axes[0].axvline(rts.mean(), color='r', linestyle='--', label=f'Mean: {rts.mean():.3f}')
    axes[0].axvline(rts.mean() - rts.std(), color='orange', linestyle=':', label=f'±1 std')
    axes[0].axvline(rts.mean() + rts.std(), color='orange', linestyle=':')
    axes[0].set_xlabel('Normalized RT')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'Training RT Distribution (n={len(rts)})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Test predictions (simulated)
    test_preds = np.random.normal(0.75, 0.04, 19000)  # Simulate test predictions
    test_preds = np.clip(test_preds, 0.70, 0.82)

    axes[1].hist(test_preds, bins=50, alpha=0.7, color='orange', edgecolor='black')
    axes[1].axvline(0.75, color='r', linestyle='--', label='Mean: 0.750')
    axes[1].set_xlabel('Normalized RT')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Test Predictions (Collapsed)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = Path('analysis')
    output_path.mkdir(exist_ok=True)
    plot_file = output_path / 'c1_rt_distribution.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"✅ Saved plot: {plot_file}")

    print(f"\n{'='*80}\n")

    return rts


if __name__ == "__main__":
    diagnose_training_distribution()
