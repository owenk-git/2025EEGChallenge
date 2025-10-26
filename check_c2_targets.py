#!/usr/bin/env python3
"""
Simple C2 Target Checker - Uses existing C1 data to check externalizing values
"""

from eegdash.dataset import EEGChallengeDataset
import numpy as np

print("="*60)
print("ANALYZING C2 TARGET VALUES (EXTERNALIZING)")
print("="*60)
print("\nNote: Externalizing is a subject-level trait,")
print("so we can use any task data (using C1's contrastChangeDetection)")
print()

# Load dataset - use same as C1 but look at externalizing column
dataset = EEGChallengeDataset(
    task="contrastChangeDetection",
    release="R11",
    cache_dir="./data_cache/eeg_challenge",
    mini=False
)

print(f"✅ Loaded {len(dataset.datasets)} recordings\n")

# Check what columns are available
print("Available columns in dataset.description:")
print(dataset.description.columns.tolist())
print()

# Extract externalizing values
if 'externalizing' not in dataset.description.columns:
    print("❌ 'externalizing' column not found!")
    print("\nThis means C2 data might not be available yet,")
    print("OR it's in a different format.")
    print("\nFor now, you can:")
    print("1. Train C1 (trial-level) - this is working!")
    print("2. Submit C1 and see the breakthrough results")
    print("3. Check with competition organizers about C2 data access")
    exit(1)

externalizing_values = []
valid_count = 0
nan_count = 0

for idx in range(len(dataset.description)):
    ext_val = dataset.description.iloc[idx].get('externalizing', np.nan)
    if not np.isnan(ext_val):
        externalizing_values.append(ext_val)
        valid_count += 1
    else:
        nan_count += 1

if len(externalizing_values) == 0:
    print("❌ No valid externalizing values found (all NaN)")
    print(f"   Valid: {valid_count}, NaN: {nan_count}")
    print("\nThis suggests C2 target data is not available yet.")
    print("\nRecommendation: Focus on C1 for now!")
    exit(1)

externalizing_values = np.array(externalizing_values)

print(f"✅ Found externalizing values!")
print(f"   Valid: {valid_count}")
print(f"   NaN: {nan_count}")

print("\n" + "="*60)
print("EXTERNALIZING TARGET STATISTICS")
print("="*60)
print(f"Total values: {len(externalizing_values)}")
print(f"Range: [{externalizing_values.min():.4f}, {externalizing_values.max():.4f}]")
print(f"Mean: {externalizing_values.mean():.4f}")
print(f"Std: {externalizing_values.std():.4f}")
print(f"Median: {np.median(externalizing_values):.4f}")

# Check if standardized
mean_val = externalizing_values.mean()
std_val = externalizing_values.std()

print("\n" + "="*60)
print("STANDARDIZATION CHECK")
print("="*60)

is_standardized = abs(mean_val) < 0.5 and abs(std_val - 1.0) < 0.5

if is_standardized:
    print(f"✅ Values appear STANDARDIZED (mean~0, std~1)")
    print(f"   Mean: {mean_val:.4f}")
    print(f"   Std: {std_val:.4f}")
    print(f"\n✅ Model output_range=(-3, 3) is CORRECT")
    print(f"\n✅ SAFE TO TRAIN C2!")
else:
    print(f"⚠️ Values may NOT be standardized")
    print(f"   Mean: {mean_val:.4f}")
    print(f"   Std: {std_val:.4f}")
    print(f"\n⚠️ Model output_range=(-3, 3) might need adjustment")

# Distribution
print("\n" + "="*60)
print("DISTRIBUTION")
print("="*60)

bins = np.linspace(externalizing_values.min(), externalizing_values.max(), 11)
hist, _ = np.histogram(externalizing_values, bins=bins)

for i in range(len(hist)):
    bar = '█' * int(hist[i] / hist.max() * 40) if hist.max() > 0 else ''
    print(f"  [{bins[i]:6.2f}, {bins[i+1]:6.2f}): {bar} ({hist[i]})")

print("\n" + "="*60)
print("CONCLUSION")
print("="*60)

if is_standardized:
    print("✅ C2 targets are standardized - safe to train!")
    print("\nNext steps:")
    print("  1. python3 DEBUG_C2_TRAINING.py --challenge c2 --epochs 100 --batch_size 64")
    print("  2. python3 FINAL_C2_SUBMISSION.py --device cuda")
else:
    print("⚠️ C2 targets may need adjustment - verify before training")
    print("\nRecommend: Focus on C1 first (trial-level breakthrough!)")

print("="*60)
