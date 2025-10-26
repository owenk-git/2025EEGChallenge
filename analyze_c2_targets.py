#!/usr/bin/env python3
"""
Analyze C2 target values (externalizing) to understand the correct output range
"""

from eegdash.dataset import EEGChallengeDataset
import numpy as np

print("Loading C2 dataset...")
dataset = EEGChallengeDataset(
    task="rest",
    release="R11",
    cache_dir="./data_cache/eeg_challenge",
    mini=False,
    train=True
)

print(f"Loaded {len(dataset.datasets)} recordings\n")

# Extract all externalizing values
externalizing_values = []
for idx in range(len(dataset.description)):
    ext_val = dataset.description.iloc[idx].get('externalizing', np.nan)
    if not np.isnan(ext_val):
        externalizing_values.append(ext_val)

externalizing_values = np.array(externalizing_values)

print("="*60)
print("EXTERNALIZING TARGET ANALYSIS")
print("="*60)
print(f"Total values: {len(externalizing_values)}")
print(f"Range: [{externalizing_values.min():.4f}, {externalizing_values.max():.4f}]")
print(f"Mean: {externalizing_values.mean():.4f}")
print(f"Std: {externalizing_values.std():.4f}")
print(f"Median: {np.median(externalizing_values):.4f}")
print(f"25th percentile: {np.percentile(externalizing_values, 25):.4f}")
print(f"75th percentile: {np.percentile(externalizing_values, 75):.4f}")

# Check if values are standardized (mean~0, std~1) or raw
if abs(externalizing_values.mean()) < 0.5 and abs(externalizing_values.std() - 1.0) < 0.5:
    print("\n✓ Values appear to be STANDARDIZED (mean~0, std~1)")
    print(f"  Model output_range=(-3, 3) is appropriate (covers ±3 std)")
else:
    print("\n⚠️ Values do NOT appear to be standardized")
    print(f"  Model output_range=(-3, 3) may need adjustment")

# Distribution
print("\nDistribution:")
bins = np.linspace(externalizing_values.min(), externalizing_values.max(), 10)
hist, _ = np.histogram(externalizing_values, bins=bins)
for i in range(len(hist)):
    bar = '█' * int(hist[i] / hist.max() * 50)
    print(f"  [{bins[i]:6.2f}, {bins[i+1]:6.2f}): {bar} ({hist[i]})")

print("\n" + "="*60)
