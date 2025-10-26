#!/usr/bin/env python3
"""
FINAL CHALLENGE 1 SUBMISSION - TRIAL-LEVEL APPROACH

Based on breakthrough insight:
- C1 requires TRIAL-LEVEL prediction (not recording-level)
- Extract individual trials, predict RT per trial, aggregate to recording
- Achieved Val NRMSE: 0.9693 (11% better than previous best 1.09)

Expected test NRMSE: 0.96-1.00 (beating current best 1.09)
"""

import torch
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import zipfile
import datetime

from models.trial_level_rt_predictor import TrialLevelRTPredictor

try:
    from eegdash.dataset import EEGChallengeDataset
    EEGDASH_AVAILABLE = True
except ImportError:
    EEGDASH_AVAILABLE = False
    print("⚠️ eegdash not available")


def extract_trials_from_recording(raw, pre_stim=0.5, post_stim=1.5, sfreq=100, n_channels=129):
    """
    Extract individual trials from recording

    Returns: List of trial data arrays
    """
    annotations = raw.annotations
    trials = []
    trial_length = int((pre_stim + post_stim) * sfreq)  # 200 points

    # Method 1: Find stimulus-response pairs (preferred)
    stim_events = []
    resp_events = []

    for i in range(len(annotations)):
        desc = annotations.description[i].lower()
        onset = annotations.onset[i]

        if any(kw in desc for kw in ['stimulus', 'stim', 'change', 'target', 'cue']):
            stim_events.append({'onset': onset})
        elif any(kw in desc for kw in ['response', 'resp', 'button', 'press', 'key']):
            resp_events.append({'onset': onset})

    # Extract trials around each stimulus
    for stim in stim_events:
        stim_time = stim['onset']

        # Extract [-pre_stim, +post_stim] around stimulus
        start_sample = int((stim_time - pre_stim) * sfreq)
        stop_sample = int((stim_time + post_stim) * sfreq)

        # Check bounds
        if start_sample < 0 or stop_sample > raw.n_times:
            continue

        # Extract trial EEG
        trial_data = raw.get_data(start=start_sample, stop=stop_sample)

        # Ensure correct shape
        if trial_data.shape[1] != trial_length:
            if trial_data.shape[1] < trial_length:
                pad_width = trial_length - trial_data.shape[1]
                trial_data = np.pad(trial_data, ((0, 0), (0, pad_width)), mode='edge')
            else:
                trial_data = trial_data[:, :trial_length]

        if trial_data.shape[0] > n_channels:
            trial_data = trial_data[:n_channels, :]
        elif trial_data.shape[0] < n_channels:
            pad_width = n_channels - trial_data.shape[0]
            trial_data = np.pad(trial_data, ((0, pad_width), (0, 0)), mode='constant')

        trials.append(trial_data)

    # Method 2: Fallback to sliding windows if no events found
    if len(trials) == 0:
        print(f"⚠️ No events found, using sliding window fallback")
        data = raw.get_data()
        window_samples = trial_length

        # Create non-overlapping windows
        num_windows = max(1, data.shape[1] // window_samples)

        for win_idx in range(num_windows):
            start = win_idx * window_samples
            stop = min(start + window_samples, data.shape[1])

            trial_data = data[:, start:stop]

            if trial_data.shape[1] < trial_length:
                pad_width = trial_length - trial_data.shape[1]
                trial_data = np.pad(trial_data, ((0, 0), (0, pad_width)), mode='edge')

            if trial_data.shape[0] > n_channels:
                trial_data = trial_data[:n_channels, :]
            elif trial_data.shape[0] < n_channels:
                pad_width = n_channels - trial_data.shape[0]
                trial_data = np.pad(trial_data, ((0, pad_width), (0, 0)), mode='constant')

            trials.append(trial_data)

    return trials


def create_c1_submission(
    model_path='checkpoints/trial_level_c1_best.pt',
    output_dir='submissions',
    device='cuda'
):
    """
    Create Challenge 1 submission using trial-level predictions

    Process:
    1. Load trained trial-level model
    2. Load test dataset
    3. For each test recording:
       a. Extract individual trials
       b. Predict RT for each trial
       c. Aggregate (median) to recording-level prediction
    4. Save predictions and create zip
    """
    print("="*80)
    print("CHALLENGE 1 SUBMISSION - TRIAL-LEVEL APPROACH")
    print("="*80)
    print("\nBreakthrough: Predicting per-trial RT, not per-recording")
    print(f"Validation NRMSE: 0.9693 (11% better than previous best 1.09)\n")

    if not EEGDASH_AVAILABLE:
        raise ImportError("eegdash required for submission")

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load model
    print(f"Loading model from {model_path}...")
    model = TrialLevelRTPredictor(
        n_channels=129,
        trial_length=200,
        pre_stim_points=50
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    best_nrmse = checkpoint.get('best_nrmse', 'N/A')
    print(f"✅ Model loaded (Val NRMSE: {best_nrmse})\n")

    # Load test dataset
    print("Loading test dataset...")
    test_dataset = EEGChallengeDataset(
        task="contrastChangeDetection",
        release="R11",
        cache_dir="./data_cache/eeg_challenge",
        mini=False,
        train=False  # Test set
    )

    print(f"✅ Loaded {len(test_dataset.datasets)} test recordings\n")

    # Make predictions
    print("Making trial-level predictions...")
    predictions = []
    recording_info = []

    with torch.no_grad():
        for rec_idx in tqdm(range(len(test_dataset.datasets)), desc="Processing"):
            raw = test_dataset.datasets[rec_idx].raw

            # Extract trials from this recording
            trials = extract_trials_from_recording(raw)

            if len(trials) == 0:
                # Ultimate fallback
                predictions.append(1.0)
                recording_info.append({
                    'idx': rec_idx,
                    'num_trials': 0,
                    'prediction': 1.0,
                    'method': 'fallback'
                })
                continue

            # Predict RT for each trial
            trial_predictions = []
            for trial_data in trials:
                trial_tensor = torch.tensor(trial_data, dtype=torch.float32).unsqueeze(0).to(device)
                rt_pred = model(trial_tensor).item()
                trial_predictions.append(rt_pred)

            # Aggregate trial predictions (median is robust to outliers)
            recording_prediction = np.median(trial_predictions)

            # Denormalize from [0, 1] to actual RT
            # Model trained on: (rt - 0.2) / (2.0 - 0.2)
            rt_actual = recording_prediction * (2.0 - 0.2) + 0.2

            # Map to output range [0.5, 1.5] centered at 1.0
            # Based on observation: typical RT ~0.5-0.8s
            # Map [0.4, 1.0]s → [0.5, 1.5]
            output_value = 0.5 + (rt_actual - 0.4) / 0.6
            output_value = np.clip(output_value, 0.5, 1.5)

            predictions.append(output_value)
            recording_info.append({
                'idx': rec_idx,
                'num_trials': len(trials),
                'prediction': output_value,
                'method': 'trial_level'
            })

    predictions = np.array(predictions)

    # Statistics
    print(f"\n{'='*80}")
    print("PREDICTION STATISTICS")
    print(f"{'='*80}")
    print(f"Total recordings: {len(predictions)}")
    print(f"Predictions range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"Mean: {predictions.mean():.4f}")
    print(f"Std: {predictions.std():.4f}")
    print(f"Median: {np.median(predictions):.4f}")

    # Trial extraction success rate
    trial_counts = [info['num_trials'] for info in recording_info]
    print(f"\nTrial extraction:")
    print(f"  Mean trials per recording: {np.mean(trial_counts):.1f}")
    print(f"  Total trials extracted: {sum(trial_counts)}")
    print(f"  Success rate: {sum(1 for c in trial_counts if c > 0)/len(trial_counts)*100:.1f}%")

    # Save predictions
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pred_file = output_path / 'y_pred.csv'
    np.savetxt(pred_file, predictions, delimiter=',', fmt='%.6f')
    print(f"\n✅ Saved predictions to {pred_file}")

    # Create submission zip
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    zip_name = f'c1_trial_level_{timestamp}.zip'
    zip_path = output_path / zip_name

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(pred_file, 'y_pred.csv')

    print(f"✅ Created submission: {zip_path}")

    # Expected performance
    print(f"\n{'='*80}")
    print("EXPECTED PERFORMANCE")
    print(f"{'='*80}")
    print(f"Validation NRMSE: {best_nrmse}")
    print(f"Expected test NRMSE: {float(best_nrmse)*1.0:.4f} - {float(best_nrmse)*1.05:.4f}")
    print(f"\nComparison:")
    print(f"  Current best: 1.09")
    print(f"  Expected: 0.96-1.00")
    print(f"  Improvement: ~10%")
    print(f"  Target (top team): 0.976")
    print(f"  Gap to target: ~1%")
    print(f"\n{'='*80}")

    return zip_path


def main():
    parser = argparse.ArgumentParser(description='Create C1 Trial-Level Submission')
    parser.add_argument('--model_path', type=str,
                        default='checkpoints/trial_level_c1_best.pt',
                        help='Path to trained model')
    parser.add_argument('--output_dir', type=str, default='submissions',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')

    args = parser.parse_args()

    create_c1_submission(
        model_path=args.model_path,
        output_dir=args.output_dir,
        device=args.device
    )


if __name__ == '__main__':
    main()
