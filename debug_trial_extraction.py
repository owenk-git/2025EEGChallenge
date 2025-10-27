#!/usr/bin/env python3
"""
Debug Trial Extraction

Check if trial-level dataset is extracting real trials with real RTs
or using sliding windows with fake RTs
"""

import numpy as np
from data.trial_level_loader import TrialLevelDataset
import matplotlib.pyplot as plt


def analyze_trial_extraction():
    print("="*70)
    print("TRIAL EXTRACTION ANALYSIS")
    print("="*70)

    # Load dataset
    print("\n📦 Loading dataset...")
    dataset = TrialLevelDataset(challenge='c1', mini=False)

    print(f"\n✅ Loaded {len(dataset.trials)} trials from {len(dataset.eeg_dataset.datasets)} recordings")

    # Analyze trial sources
    real_trials = 0
    fake_trials = 0
    rts_real = []
    rts_fake = []
    recordings_with_real = set()
    recordings_with_fake = set()

    for trial_data, rt, info in dataset.trials:
        if 'is_sliding_window' in info and info['is_sliding_window']:
            fake_trials += 1
            rts_fake.append(rt)
            recordings_with_fake.add(info['recording_idx'])
        else:
            real_trials += 1
            rts_real.append(rt)
            recordings_with_real.add(info['recording_idx'])

    print(f"\n{'='*70}")
    print(f"TRIAL SOURCE BREAKDOWN")
    print(f"{'='*70}")
    print(f"Real trials (from annotations):  {real_trials:>6} ({real_trials/len(dataset.trials)*100:.1f}%)")
    print(f"Fake trials (sliding windows):   {fake_trials:>6} ({fake_trials/len(dataset.trials)*100:.1f}%)")
    print(f"\nRecordings with real trials:     {len(recordings_with_real)}")
    print(f"Recordings with fake trials:     {len(recordings_with_fake)}")

    # Analyze RT distributions
    if len(rts_real) > 0:
        print(f"\n{'='*70}")
        print(f"REAL TRIAL RT STATISTICS")
        print(f"{'='*70}")
        rts_real = np.array(rts_real)
        print(f"Count:    {len(rts_real)}")
        print(f"Mean:     {rts_real.mean():.3f} s")
        print(f"Std:      {rts_real.std():.3f} s")
        print(f"Min:      {rts_real.min():.3f} s")
        print(f"Max:      {rts_real.max():.3f} s")
        print(f"Median:   {np.median(rts_real):.3f} s")

    if len(rts_fake) > 0:
        print(f"\n{'='*70}")
        print(f"FAKE TRIAL RT STATISTICS (should be suspicious)")
        print(f"{'='*70}")
        rts_fake = np.array(rts_fake)
        print(f"Count:    {len(rts_fake)}")
        print(f"Mean:     {rts_fake.mean():.3f} s")
        print(f"Std:      {rts_fake.std():.3f} s")
        print(f"Min:      {rts_fake.min():.3f} s")
        print(f"Max:      {rts_fake.max():.3f} s")
        print(f"Unique values: {len(np.unique(rts_fake))}")

        # Check if fake RTs are just placeholders
        unique_rts = np.unique(rts_fake)
        print(f"\nUnique RT values in fake trials: {unique_rts[:10]}...")
        if len(unique_rts) < 20:
            print("⚠️  WARNING: Very few unique RT values - these are PLACEHOLDERS!")

    # Sample trials
    print(f"\n{'='*70}")
    print(f"SAMPLE TRIALS (first 5)")
    print(f"{'='*70}")
    for i in range(min(5, len(dataset.trials))):
        trial_data, rt, info = dataset.trials[i]
        trial_type = "FAKE" if info.get('is_sliding_window', False) else "REAL"
        print(f"\nTrial {i}:")
        print(f"  Type: {trial_type}")
        print(f"  RT: {rt:.3f} s")
        print(f"  Recording: {info['recording_idx']}")
        print(f"  Shape: {trial_data.shape}")
        print(f"  Info keys: {list(info.keys())}")

    # Final verdict
    print(f"\n{'='*70}")
    print(f"VERDICT")
    print(f"{'='*70}")

    if fake_trials == 0:
        print("✅ All trials extracted from real annotations - GOOD!")
    elif fake_trials > real_trials:
        print("❌ MORE FAKE TRIALS THAN REAL - THIS IS THE PROBLEM!")
        print("   Models are learning from placeholder RTs, not real RTs!")
        print("   This explains low correlation and poor generalization!")
    elif fake_trials > 0:
        print("⚠️  Some fake trials present - might cause issues")
        print(f"   Recommend removing recordings with no real trials")

    return {
        'total_trials': len(dataset.trials),
        'real_trials': real_trials,
        'fake_trials': fake_trials,
        'percent_fake': fake_trials / len(dataset.trials) * 100
    }


def check_recording_level_targets():
    """Check what targets recording-level models actually use"""
    print(f"\n{'='*70}")
    print(f"RECORDING-LEVEL TARGET ANALYSIS")
    print(f"{'='*70}")

    try:
        from eegdash.dataset import EEGChallengeDataset

        dataset = EEGChallengeDataset(
            task="contrastChangeDetection",
            release="R11",
            cache_dir='./data_cache/eeg_challenge',
            mini=True
        )

        print(f"\n✅ Loaded {len(dataset.datasets)} recordings")

        # Check targets
        targets = []
        for i in range(min(10, len(dataset.datasets))):
            raw = dataset.datasets[i].raw
            metadata = dataset.datasets[i].metadata

            print(f"\nRecording {i}:")
            print(f"  Duration: {raw.times[-1]:.1f} s")
            print(f"  Metadata keys: {list(metadata.keys())}")

            # Try to find RT target
            if 'rt' in metadata:
                print(f"  RT target: {metadata['rt']}")
                targets.append(metadata['rt'])
            elif 'mean_rt' in metadata:
                print(f"  Mean RT: {metadata['mean_rt']}")
                targets.append(metadata['mean_rt'])
            else:
                print(f"  No RT found in metadata!")

        if len(targets) > 0:
            targets = np.array(targets)
            print(f"\nRecording-level target statistics:")
            print(f"  Mean: {targets.mean():.3f}")
            print(f"  Std:  {targets.std():.3f}")
            print(f"  Range: [{targets.min():.3f}, {targets.max():.3f}]")

    except Exception as e:
        print(f"Could not load recording-level dataset: {e}")


if __name__ == '__main__':
    # Analyze trial extraction
    stats = analyze_trial_extraction()

    # Check recording-level
    check_recording_level_targets()

    # Summary recommendation
    print(f"\n{'='*70}")
    print(f"RECOMMENDATION")
    print(f"{'='*70}")

    if stats['percent_fake'] > 50:
        print("\n❌ STOP USING TRIAL-LEVEL APPROACH!")
        print("   > 50% of trials are fake (sliding windows)")
        print("   Models cannot learn meaningful RT predictions")
        print("   Switch to recording-level models")
    elif stats['percent_fake'] > 20:
        print("\n⚠️  Trial extraction has issues")
        print(f"   {stats['percent_fake']:.1f}% of trials are fake")
        print("   Consider filtering or fixing trial extraction")
    else:
        print("\n✅ Trial extraction looks reasonable")
        print("   Low percentage of fake trials")
        print("   Issue might be elsewhere")

    print(f"\n{'='*70}\n")
