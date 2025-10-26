#!/usr/bin/env python3
"""Quick diagnosis of C1 training data"""

import torch
import numpy as np

print("="*80)
print("QUICK C1 TRAINING DATA DIAGNOSIS")
print("="*80)

# Load checkpoint and check what's stored
print("\n1. Checking trained model checkpoint...")
checkpoint = torch.load('checkpoints/trial_level_c1_best.pt', weights_only=False, map_location='cpu')
print(f"   Checkpoint keys: {list(checkpoint.keys())}")
print(f"   Best val NRMSE: {checkpoint.get('best_nrmse', 'N/A')}")

# Load a small sample of training data
print("\n2. Loading sample of training data...")
from data.trial_level_loader import TrialLevelDataset

dataset = TrialLevelDataset(challenge='c1', mini=False)
print(f"   Total trials in dataset: {len(dataset)}")

# Sample 1000 trials
print("\n3. Sampling RT values from first 1000 trials...")
rts = []
for i in range(min(1000, len(dataset))):
    try:
        data, rt_norm, info = dataset[i]
        rts.append(float(rt_norm))
    except Exception as e:
        print(f"   Error at trial {i}: {e}")
        continue

rts = np.array(rts)

print(f"\n{'='*80}")
print("TRAINING RT STATISTICS (Normalized [0,1])")
print(f"{'='*80}")
print(f"Sample size: {len(rts)} trials")
print(f"Mean: {rts.mean():.4f}")
print(f"Std: {rts.std():.4f}")
print(f"Min: {rts.min():.4f}")
print(f"Max: {rts.max():.4f}")

print(f"\nPercentiles:")
for p in [5, 25, 50, 75, 95]:
    print(f"  {p:>2}th: {np.percentile(rts, p):.4f}")

print(f"\nDistribution:")
bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
for low, high in bins:
    count = np.sum((rts >= low) & (rts < high))
    pct = 100 * count / len(rts)
    print(f"  [{low:.1f}, {high:.1f}): {count:>4} ({pct:>5.1f}%)")

print(f"\n{'='*80}")
print("COMPARISON")
print(f"{'='*80}")
print(f"Training range: [{rts.min():.4f}, {rts.max():.4f}] (span: {rts.max()-rts.min():.4f})")
print(f"Training std: {rts.std():.4f}")
print(f"\nTest predictions: [0.70, 0.82] (span: 0.12)")
print(f"Test std: 0.0116")

print(f"\n{'='*80}")
print("DIAGNOSIS")
print(f"{'='*80}")

if rts.std() > 0.15 and (rts.max() - rts.min()) > 0.4:
    print("🔴 PREDICTION COLLAPSE CONFIRMED!")
    print(f"   Training has WIDE range ({rts.max()-rts.min():.2f})")
    print(f"   But test predictions are NARROW (0.12)")
    print(f"   Model is not using its full capacity")
    print(f"\n✅ SOLUTION: Use temperature scaling")
    print(f"   Recommended: T=1.5 or T=2.0")
elif rts.std() < 0.10:
    print("🟡 Training data is NARROW")
    print(f"   Predictions might be correct")
    print(f"   Test RTs might actually be concentrated around 0.75")
else:
    print("🟡 Moderate variance in training")
    print(f"   Try temperature scaling T=1.3-1.5")

print(f"\n{'='*80}\n")
