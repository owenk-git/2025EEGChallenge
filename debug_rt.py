"""
Debug RT extraction - check if it's working correctly
"""

from data.official_dataset_example import OfficialEEGDataset
from data.rt_extractor import extract_response_time, normalize_rt
import numpy as np

print("="*70)
print("Debugging RT Extraction for Challenge 1")
print("="*70)

# Create dataset
dataset = OfficialEEGDataset(
    task="contrastChangeDetection",
    challenge='c1',
    release="R11",
    mini=False,
    max_subjects=10  # Just check first 10 subjects
)

print(f"\nChecking {len(dataset)} recordings...\n")

rt_values = []
failed_count = 0

for i in range(min(20, len(dataset))):
    # Get raw data
    actual_idx = dataset.valid_indices[i]
    raw = dataset.eeg_dataset.datasets[actual_idx].raw

    # Extract RT
    rt = extract_response_time(raw, method='mean', verbose=False)
    rt_norm = normalize_rt(rt)

    if rt is None:
        failed_count += 1
        print(f"Recording {i}: RT extraction FAILED ❌")
    else:
        rt_values.append(rt)
        print(f"Recording {i}: RT = {rt:.3f}s, normalized = {rt_norm:.3f}")

        # Show annotations for first few
        if i < 3:
            annotations = raw.annotations
            unique_desc = set(annotations.description)
            print(f"  Annotations: {len(annotations)} events")
            print(f"  Unique types: {unique_desc}")
            print()

print("\n" + "="*70)
print("Summary")
print("="*70)
print(f"Total checked: {min(20, len(dataset))}")
print(f"Successful: {len(rt_values)}")
print(f"Failed: {failed_count}")

if len(rt_values) > 0:
    print(f"\nRT Statistics:")
    print(f"  Mean: {np.mean(rt_values):.3f}s")
    print(f"  Std:  {np.std(rt_values):.3f}s")
    print(f"  Min:  {np.min(rt_values):.3f}s")
    print(f"  Max:  {np.max(rt_values):.3f}s")

    # Check normalized values
    rt_norm_values = [normalize_rt(rt) for rt in rt_values]
    print(f"\nNormalized RT Statistics:")
    print(f"  Mean: {np.mean(rt_norm_values):.3f}")
    print(f"  Std:  {np.std(rt_norm_values):.3f}")
    print(f"  Min:  {np.min(rt_norm_values):.3f}")
    print(f"  Max:  {np.max(rt_norm_values):.3f}")
else:
    print("\n❌ RT extraction is completely failing!")
    print("   Falling back to age proxy (which explains poor performance)")
