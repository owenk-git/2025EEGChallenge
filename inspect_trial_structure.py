#!/usr/bin/env python3
"""
Investigate if we can extract trial-level RT from EEG recordings
"""

from eegdash.dataset import EEGChallengeDataset
import numpy as np

# Load dataset
dataset = EEGChallengeDataset(
    task="contrastChangeDetection",
    release="R11",
    cache_dir="./data_cache/eeg_challenge",
    mini=True  # Use mini for fast testing
)

print(f"Loaded {len(dataset.datasets)} recordings\n")
print("="*80)

# Inspect first recording in detail
for rec_idx in range(min(3, len(dataset.datasets))):
    print(f"\n{'='*80}")
    print(f"RECORDING {rec_idx}")
    print(f"{'='*80}\n")

    raw = dataset.datasets[rec_idx].raw

    # Basic info
    print(f"Duration: {raw.times[-1]:.1f}s")
    print(f"Sampling rate: {raw.info['sfreq']} Hz")
    print(f"Channels: {len(raw.ch_names)}")
    print(f"Shape: {raw.get_data().shape}")

    # Check annotations (events)
    annotations = raw.annotations
    print(f"\nAnnotations: {len(annotations)} events")

    # Group by description
    from collections import Counter
    event_types = Counter(annotations.description)
    print(f"\nEvent types:")
    for event_type, count in event_types.most_common():
        print(f"  {event_type}: {count}")

    # Look for stimulus and response events
    print(f"\nFirst 15 events:")
    for i in range(min(15, len(annotations))):
        onset = annotations.onset[i]
        duration = annotations.duration[i]
        desc = annotations.description[i]
        print(f"  {onset:7.2f}s [{duration:.3f}s]: {desc}")

    # Try to pair stimulus and response
    print(f"\nTrying to extract trial structure...")

    stimulus_events = []
    response_events = []

    for i in range(len(annotations)):
        desc = annotations.description[i].lower()
        onset = annotations.onset[i]

        if 'stimulus' in desc or 'stim' in desc:
            stimulus_events.append((onset, desc))
        elif 'response' in desc or 'resp' in desc or 'button' in desc:
            response_events.append((onset, desc))

    print(f"Found {len(stimulus_events)} stimulus events")
    print(f"Found {len(response_events)} response events")

    # Try to match stimulus-response pairs
    if len(stimulus_events) > 0 and len(response_events) > 0:
        print(f"\nAttempting to compute RT from paired events:")
        for i in range(min(5, len(stimulus_events))):
            stim_time, stim_desc = stimulus_events[i]

            # Find next response after this stimulus
            for resp_time, resp_desc in response_events:
                if resp_time > stim_time:
                    rt = resp_time - stim_time
                    print(f"  Trial {i}: Stimulus @{stim_time:.2f}s → Response @{resp_time:.2f}s → RT = {rt:.3f}s ({rt*1000:.0f}ms)")
                    break

    # Check if RT is embedded in annotations
    print(f"\nSearching for RT in annotation descriptions...")
    rt_found = []
    for i, desc in enumerate(annotations.description):
        if 'rt' in desc.lower() and '=' in desc:
            print(f"  {annotations.onset[i]:.2f}s: {desc}")
            # Try to extract RT value
            try:
                rt_str = desc.split('rt')[1].split('=')[1].split()[0]
                rt_val = float(rt_str)
                rt_found.append(rt_val)
            except:
                pass

    if len(rt_found) > 0:
        print(f"\nExtracted {len(rt_found)} RT values:")
        print(f"  Mean: {np.mean(rt_found):.3f}s")
        print(f"  Std: {np.std(rt_found):.3f}s")
        print(f"  Range: [{np.min(rt_found):.3f}, {np.max(rt_found):.3f}]s")

print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}\n")
print("Next steps:")
print("1. If we found trial-level RT → Build trial-level predictor")
print("2. If RT is embedded in annotations → Extract and use it")
print("3. If only stimulus/response events → Compute RT = response_time - stimulus_time")
print("4. If nothing found → Check eegdash documentation for trial extraction")
