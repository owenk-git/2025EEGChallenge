#!/usr/bin/env python3
"""
Analyze C2 target values (externalizing) - FIXED VERSION

Checks multiple tasks to find where externalizing data is stored
"""

from eegdash.dataset import EEGChallengeDataset
import numpy as np

# C2 might use different tasks - try to find which one has data
possible_tasks = ["rest", "contrastChangeDetection", "oddball", "all"]

print("="*60)
print("ANALYZING C2 TARGET VALUES (EXTERNALIZING)")
print("="*60)

dataset = None
working_task = None

for task in possible_tasks:
    print(f"\nTrying task: '{task}'...")
    try:
        dataset = EEGChallengeDataset(
            task=task,
            release="R11",
            cache_dir="./data_cache/eeg_challenge",
            mini=False,
            train=True
        )

        if len(dataset.datasets) > 0:
            print(f"✅ Found {len(dataset.datasets)} recordings with task='{task}'")
            working_task = task
            break
        else:
            print(f"⚠️ No recordings found for task='{task}'")
    except Exception as e:
        print(f"❌ Error with task='{task}': {e}")

if dataset is None or len(dataset.datasets) == 0:
    print("\n❌ Could not load any C2 data!")
    print("\nPossible solutions:")
    print("1. Check if data is downloaded")
    print("2. Try different task names")
    print("3. Check eegdash documentation")
    exit(1)

print(f"\n✅ Using task: '{working_task}'")
print(f"✅ Loaded {len(dataset.datasets)} recordings\n")

# Extract all externalizing values
print("Extracting externalizing values...")
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
    print("\n❌ No externalizing values found!")
    print("\nAvailable columns in description:")
    print(dataset.description.columns.tolist())
    print("\nFirst few rows:")
    print(dataset.description.head())
    exit(1)

externalizing_values = np.array(externalizing_values)

print(f"Valid values: {valid_count}")
print(f"NaN values: {nan_count}")

print("\n" + "="*60)
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
print("\n" + "="*60)
print("STANDARDIZATION CHECK")
print("="*60)

mean_val = externalizing_values.mean()
std_val = externalizing_values.std()

is_standardized = abs(mean_val) < 0.5 and abs(std_val - 1.0) < 0.5

if is_standardized:
    print(f"✅ Values appear to be STANDARDIZED (mean~0, std~1)")
    print(f"   Mean: {mean_val:.4f} (close to 0)")
    print(f"   Std: {std_val:.4f} (close to 1)")
    print(f"\n✅ Model output_range=(-3, 3) is APPROPRIATE")
    print(f"   This covers ~99.7% of data (±3 std)")
    print(f"\n✅ NO CHANGES NEEDED - Proceed with training!")
else:
    print(f"⚠️ Values do NOT appear to be standardized!")
    print(f"   Mean: {mean_val:.4f} (far from 0)")
    print(f"   Std: {std_val:.4f} (far from 1)")
    print(f"\n⚠️ Model output_range=(-3, 3) may be INAPPROPRIATE")
    print(f"   Current range: [{externalizing_values.min():.2f}, {externalizing_values.max():.2f}]")
    print(f"   Model range: [-3, 3]")
    print(f"\n⚠️ CRITICAL: Need to adjust model or standardize targets!")
    print(f"   Contact me for a fix before training!")

# Distribution visualization
print("\n" + "="*60)
print("DISTRIBUTION")
print("="*60)

bins = np.linspace(externalizing_values.min(), externalizing_values.max(), 11)
hist, _ = np.histogram(externalizing_values, bins=bins)

for i in range(len(hist)):
    bar = '█' * int(hist[i] / hist.max() * 50)
    print(f"  [{bins[i]:6.2f}, {bins[i+1]:6.2f}): {bar} ({hist[i]})")

# Check coverage of model output range
if is_standardized:
    within_range = ((externalizing_values >= -3) & (externalizing_values <= 3)).sum()
    coverage = within_range / len(externalizing_values) * 100
    print(f"\nCoverage by model output_range [-3, 3]: {coverage:.1f}%")
    if coverage < 95:
        print(f"⚠️ Only {coverage:.1f}% of values within model range!")
    else:
        print(f"✅ Good coverage!")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)

if is_standardized:
    print("✅ READY FOR C2 TRAINING!")
    print("\nNext steps:")
    print("  1. Train C2 model:")
    print("     python3 DEBUG_C2_TRAINING.py --challenge c2 --epochs 100 --batch_size 64")
    print("  2. Create submission:")
    print("     python3 FINAL_C2_SUBMISSION.py --device cuda")
else:
    print("⚠️ NOT READY - NEED FIX!")
    print("\nNext steps:")
    print("  1. Ask me to create a fix for non-standardized targets")
    print("  2. DO NOT train yet!")

print("="*60)
