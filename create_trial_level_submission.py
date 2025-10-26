#!/usr/bin/env python3
"""
Create submission for trial-level RT predictor

Key difference from recording-level:
- Load test recordings
- Extract trials from each recording
- Predict RT for each trial
- AGGREGATE trial predictions to get recording-level prediction
"""

import torch
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import zipfile

from models.trial_level_rt_predictor import TrialLevelRTPredictor
from data.trial_level_loader import TrialLevelDataset

try:
    from eegdash.dataset import EEGChallengeDataset
    EEGDASH_AVAILABLE = True
except ImportError:
    EEGDASH_AVAILABLE = False


def extract_trials_from_recording(raw, pre_stim=0.5, post_stim=1.5, sfreq=100, n_channels=129):
    """
    Extract individual trials from a recording

    Returns: List of trial data arrays (each is n_channels x trial_length)
    """
    annotations = raw.annotations

    trials = []

    # Find stimulus-response pairs
    stim_events = []
    resp_events = []

    for i in range(len(annotations)):
        desc = annotations.description[i].lower()
        onset = annotations.onset[i]

        if any(keyword in desc for keyword in ['stimulus', 'stim', 'change', 'target']):
            stim_events.append({'onset': onset})
        elif any(keyword in desc for keyword in ['response', 'resp', 'button', 'press']):
            resp_events.append({'onset': onset})

    # Extract trials around each stimulus
    trial_length = int((pre_stim + post_stim) * sfreq)

    for stim in stim_events:
        stim_time = stim['onset']

        # Extract [-pre_stim, +post_stim] around stimulus
        start_sample = int((stim_time - pre_stim) * sfreq)
        stop_sample = int((stim_time + post_stim) * sfreq)

        # Check bounds
        if start_sample < 0 or stop_sample > raw.n_times:
            continue

        # Extract trial data
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

    # If no trials extracted, use sliding windows as fallback
    if len(trials) == 0:
        data = raw.get_data()
        window_samples = trial_length
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


def create_trial_level_submission(
    challenge='c1',
    model_path='checkpoints/trial_level_c1_best.pt',
    output_dir='submissions',
    device='cuda'
):
    """
    Create submission using trial-level predictions

    Process:
    1. Load test dataset
    2. For each recording:
       a. Extract individual trials
       b. Predict RT for each trial
       c. Aggregate (median) to get recording-level prediction
    3. Save predictions
    """
    print("="*60)
    print(f"CREATING TRIAL-LEVEL SUBMISSION FOR C{challenge[-1]}")
    print("="*60)

    if not EEGDASH_AVAILABLE:
        raise ImportError("eegdash required")

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
    print(f"✅ Model loaded (best NRMSE: {checkpoint['best_nrmse']:.4f})\n")

    # Load test dataset
    print("Loading test dataset...")
    task = "contrastChangeDetection" if challenge == 'c1' else "rest"

    test_dataset = EEGChallengeDataset(
        task=task,
        release="R11",
        cache_dir="./data_cache/eeg_challenge",
        mini=False,
        train=False  # Test set
    )

    print(f"✅ Loaded {len(test_dataset.datasets)} test recordings\n")

    # Make predictions
    print("Making predictions...")
    predictions = []

    with torch.no_grad():
        for rec_idx in tqdm(range(len(test_dataset.datasets)), desc="Processing recordings"):
            raw = test_dataset.datasets[rec_idx].raw

            # Extract trials from this recording
            trials = extract_trials_from_recording(raw)

            if len(trials) == 0:
                # Fallback: predict 1.0 (middle of range)
                predictions.append(1.0)
                continue

            # Predict RT for each trial
            trial_predictions = []
            for trial_data in trials:
                trial_tensor = torch.tensor(trial_data, dtype=torch.float32).unsqueeze(0).to(device)
                rt_pred = model(trial_tensor).item()
                trial_predictions.append(rt_pred)

            # Aggregate trial predictions (median is robust)
            recording_prediction = np.median(trial_predictions)

            # Denormalize from [0, 1] to actual RT range
            # Model was trained on normalized RT: (rt - 0.2) / (2.0 - 0.2)
            rt_actual = recording_prediction * (2.0 - 0.2) + 0.2

            # For C1, output should be in range [0.5, 1.5] (normalized to mean=1.0)
            # Convert actual RT [0.2-2.0]s to output range [0.5-1.5]
            # Typical RT is ~0.5-0.8s, so normalize assuming mean ~0.65s
            output_value = 0.5 + (rt_actual - 0.4) / 0.6  # Map [0.4, 1.0]s -> [0.5, 1.5]
            output_value = np.clip(output_value, 0.5, 1.5)

            predictions.append(output_value)

    predictions = np.array(predictions)

    print(f"\n✅ Generated {len(predictions)} predictions")
    print(f"   Range: [{predictions.min():.3f}, {predictions.max():.3f}]")
    print(f"   Mean: {predictions.mean():.3f}")
    print(f"   Std: {predictions.std():.3f}")

    # Save predictions
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pred_file = output_path / 'y_pred.csv'
    np.savetxt(pred_file, predictions, delimiter=',', fmt='%.6f')
    print(f"\n✅ Saved predictions to {pred_file}")

    # Create submission zip
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    zip_name = f'trial_level_{challenge}_{timestamp}.zip'
    zip_path = output_path / zip_name

    with zipfile.ZipFile(zip_path, 'w') as zipf:
        zipf.write(pred_file, 'y_pred.csv')

    print(f"✅ Created submission: {zip_path}")
    print(f"\n{'='*60}")
    print("SUBMISSION READY!")
    print(f"{'='*60}")
    print(f"\nUpload this file to competition:")
    print(f"  {zip_path}")
    print(f"\nExpected performance:")
    print(f"  Validation NRMSE: {checkpoint['best_nrmse']:.4f}")
    print(f"  Test NRMSE (estimated): {checkpoint['best_nrmse'] * 1.05:.4f} - {checkpoint['best_nrmse'] * 1.15:.4f}")

    return zip_path


def main():
    parser = argparse.ArgumentParser(description='Create Trial-Level Submission')
    parser.add_argument('--challenge', type=str, default='c1', choices=['c1', 'c2'],
                        help='Challenge (c1 or c2)')
    parser.add_argument('--model_path', type=str, default='checkpoints/trial_level_c1_best.pt',
                        help='Path to trained model')
    parser.add_argument('--output_dir', type=str, default='submissions',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')

    args = parser.parse_args()

    create_trial_level_submission(
        challenge=args.challenge,
        model_path=args.model_path,
        output_dir=args.output_dir,
        device=args.device
    )


if __name__ == '__main__':
    main()
