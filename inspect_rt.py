"""
Inspect how to extract response time from EEG data
"""

from eegdash.dataset import EEGChallengeDataset
import numpy as np

# Load dataset
dataset = EEGChallengeDataset(
    task="contrastChangeDetection",
    release="R11",
    cache_dir="./data_cache/eeg_challenge",
    mini=False
)

print(f"Loaded {len(dataset.datasets)} recordings\n")

# Inspect first few recordings
for i in range(min(3, len(dataset.datasets))):
    print(f"{'='*70}")
    print(f"Recording {i}")
    print(f"{'='*70}")

    raw = dataset.datasets[i].raw

    # Get events
    events = raw.annotations
    print(f"\nAnnotations: {len(events)} total")
    print(f"Description unique: {set(events.description)}")

    # Print first few annotations
    print(f"\nFirst 10 annotations:")
    for j in range(min(10, len(events))):
        print(f"  {events.onset[j]:.3f}s: {events.description[j]} (duration: {events.duration[j]:.3f}s)")

    # Check if there's event information
    if hasattr(raw, 'events'):
        print(f"\nEvents: {raw.events}")

    # Check metadata
    if hasattr(dataset.datasets[i], 'metadata'):
        print(f"\nMetadata: {dataset.datasets[i].metadata}")

    print()
