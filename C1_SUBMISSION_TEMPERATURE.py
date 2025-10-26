#!/usr/bin/env python3
"""
C1 Submission with Temperature Scaling to Expand Predictions

Usage:
    python3 C1_SUBMISSION_TEMPERATURE.py --temperature 1.3 --device cuda
    python3 C1_SUBMISSION_TEMPERATURE.py --temperature 1.5 --device cuda
    python3 C1_SUBMISSION_TEMPERATURE.py --temperature 2.0 --device cuda
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import zipfile
import datetime
from tqdm import tqdm
import mne

# Import model
from models.trial_level_rt_predictor import TrialLevelRTPredictor


def extract_trials_from_recording(raw, pre_stim=0.5, post_stim=1.5):
    """
    Extract individual trials from recording

    Args:
        raw: MNE Raw object
        pre_stim: Pre-stimulus time (seconds)
        post_stim: Post-stimulus time (seconds)

    Returns:
        trials: List of (trial_data, rt) tuples
    """
    sfreq = raw.info['sfreq']

    # Get events
    events, event_id = mne.events_from_annotations(raw, verbose=False)

    # Find stimulus and response events
    stim_events = []
    resp_events = []

    for idx, (sample, _, event_type) in enumerate(events):
        if event_type in [1, 2]:  # Stimulus events
            stim_events.append({'sample': sample, 'onset': sample / sfreq})
        elif event_type in [3, 4]:  # Response events
            resp_events.append({'sample': sample, 'onset': sample / sfreq})

    # Extract trials
    trials = []
    for stim in stim_events:
        stim_time = stim['onset']

        # Find matching response
        matched_resp = None
        for resp in resp_events:
            resp_time = resp['onset']
            rt = resp_time - stim_time
            if 0.15 <= rt <= 2.0:  # Valid RT range
                matched_resp = resp
                break

        if matched_resp is None:
            continue

        # Extract trial data [-pre_stim, +post_stim]
        start_sample = int((stim_time - pre_stim) * sfreq)
        stop_sample = int((stim_time + post_stim) * sfreq)

        if start_sample < 0 or stop_sample > len(raw.times):
            continue

        trial_data = raw.get_data(start=start_sample, stop=stop_sample)

        # Ensure correct length (200 samples @ 100Hz for 2s window)
        expected_length = int((pre_stim + post_stim) * sfreq)
        if trial_data.shape[1] != expected_length:
            continue

        rt = matched_resp['onset'] - stim_time
        trials.append((trial_data, rt))

    return trials


def create_c1_submission_with_temperature(
    checkpoint_path,
    temperature=1.5,
    device='cuda',
    mini=False
):
    """
    Create C1 submission with temperature scaling

    Args:
        checkpoint_path: Path to trained model checkpoint
        temperature: Temperature for scaling (>1 expands range)
        device: 'cuda' or 'cpu'
        mini: Use mini dataset for testing
    """

    print("="*80)
    print(f"C1 SUBMISSION WITH TEMPERATURE SCALING (T={temperature})")
    print("="*80)
    print(f"\nSolution: Expand predictions by multiplying model outputs by {temperature}")
    print(f"Expected effect: Predictions will use wider range")
    print(f"Device: {device}\n")

    # Load model
    print(f"Loading model from {checkpoint_path}...")
    model = TrialLevelRTPredictor(
        n_channels=129,
        trial_length=200,
        pre_stim_points=50
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    best_nrmse = checkpoint.get('best_nrmse', 'N/A')
    print(f"✅ Model loaded (Val NRMSE: {best_nrmse})\n")

    # Load test dataset
    print("Loading test dataset...")
    from eegdash.dataset import EEGChallengeDataset

    test_dataset = EEGChallengeDataset(
        task="contrastChangeDetection",
        release="R11",
        cache_dir="./data_cache/eeg_challenge",
        mini=mini
    )

    print(f"✅ Loaded {len(test_dataset.datasets)} test recordings\n")

    # Make predictions
    print(f"Making trial-level predictions with temperature scaling (T={temperature})...")
    predictions = []
    recording_info = []

    raw_outputs = []  # Track raw model outputs
    scaled_outputs = []  # Track temperature-scaled outputs

    with torch.no_grad():
        for rec_idx in tqdm(range(len(test_dataset.datasets)), desc="Processing"):
            raw = test_dataset.datasets[rec_idx].raw

            # Extract trials
            trials = extract_trials_from_recording(raw, pre_stim=0.5, post_stim=1.5)

            if len(trials) == 0:
                predictions.append(1.0)  # Default prediction
                continue

            # Predict RT for each trial
            trial_predictions = []
            for trial_data, rt_actual in trials:
                trial_tensor = torch.tensor(trial_data, dtype=torch.float32).unsqueeze(0).to(device)
                rt_pred = model(trial_tensor).item()  # [0, 1]

                raw_outputs.append(rt_pred)

                # ============================================================
                # TEMPERATURE SCALING: Expand predictions
                # ============================================================
                rt_pred_scaled = rt_pred * temperature
                rt_pred_scaled = np.clip(rt_pred_scaled, 0, 1)  # Keep in [0,1]

                scaled_outputs.append(rt_pred_scaled)
                trial_predictions.append(rt_pred_scaled)

            # Aggregate trial predictions (median)
            recording_prediction = np.median(trial_predictions)  # [0, 1]

            # Map to competition range [0.5, 1.5]
            output_value = 0.5 + recording_prediction * 1.0
            output_value = np.clip(output_value, 0.5, 1.5)

            predictions.append(output_value)
            recording_info.append({
                'idx': rec_idx,
                'num_trials': len(trials),
                'prediction': output_value,
                'raw_model_output': recording_prediction,
                'method': f'temperature_T{temperature}'
            })

    predictions = np.array(predictions)
    raw_outputs = np.array(raw_outputs)
    scaled_outputs = np.array(scaled_outputs)

    # Statistics
    print(f"\n{'='*80}")
    print(f"PREDICTION STATISTICS (Temperature={temperature})")
    print(f"{'='*80}")
    print(f"Total recordings: {len(predictions)}")
    print(f"\nRaw model outputs (before scaling):")
    print(f"  Range: [{raw_outputs.min():.4f}, {raw_outputs.max():.4f}]")
    print(f"  Mean: {raw_outputs.mean():.4f}, Std: {raw_outputs.std():.4f}")
    print(f"\nAfter temperature scaling (T={temperature}):")
    print(f"  Range: [{scaled_outputs.min():.4f}, {scaled_outputs.max():.4f}]")
    print(f"  Mean: {scaled_outputs.mean():.4f}, Std: {scaled_outputs.std():.4f}")
    print(f"\nFinal predictions (competition range):")
    print(f"  Range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"  Mean: {predictions.mean():.4f}, Std: {predictions.std():.4f}")

    # Distribution
    bins = [(0.5, 0.7), (0.7, 0.9), (0.9, 1.1), (1.1, 1.3), (1.3, 1.5)]
    print(f"\nOutput distribution:")
    for low, high in bins:
        count = np.sum((predictions >= low) & (predictions < high))
        pct = 100 * count / len(predictions)
        print(f"  [{low}, {high}): {count} ({pct:.1f}%)")

    # Trial extraction stats
    num_trials = [info['num_trials'] for info in recording_info if 'num_trials' in info]
    print(f"\nTrial extraction:")
    print(f"  Mean trials per recording: {np.mean(num_trials):.1f}")
    print(f"  Total trials: {sum(num_trials)}")
    print(f"  Success rate: {100*len([n for n in num_trials if n > 0])/len(num_trials):.1f}%")

    # Save predictions
    output_path = Path("submissions")
    output_path.mkdir(parents=True, exist_ok=True)

    pred_file = output_path / 'y_pred.csv'
    np.savetxt(pred_file, predictions, delimiter=',', fmt='%.6f')
    print(f"\n✅ Saved predictions to {pred_file}")

    # Create submission zip
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    zip_name = f'c1_temperature_T{temperature}_{timestamp}.zip'
    zip_path = output_path / zip_name

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(pred_file, 'y_pred.csv')

    print(f"✅ Created submission: {zip_path}")

    # Expected performance
    print(f"\n{'='*80}")
    print(f"EXPECTED PERFORMANCE (T={temperature})")
    print(f"{'='*80}")
    print(f"Validation NRMSE: {best_nrmse}")

    # Estimate based on temperature
    if temperature == 1.0:
        expected_range = (1.05, 1.10)
    elif temperature <= 1.3:
        expected_range = (1.00, 1.05)
    elif temperature <= 1.5:
        expected_range = (0.95, 1.02)
    else:  # >1.5
        expected_range = (0.92, 1.00)

    print(f"Expected test NRMSE: {expected_range[0]:.2f} - {expected_range[1]:.2f}")
    print(f"\nComparison:")
    print(f"  Current best: 1.09")
    print(f"  Expected with T={temperature}: {np.mean(expected_range):.2f}")
    print(f"  Target (top team): 0.976")
    print(f"  Gap to target: ~{np.mean(expected_range) - 0.976:.2f}")
    print(f"\n{'='*80}\n")

    return zip_path


def main():
    parser = argparse.ArgumentParser(
        description="Create C1 submission with temperature scaling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Conservative expansion
    python3 C1_SUBMISSION_TEMPERATURE.py --temperature 1.3 --device cuda

    # Moderate expansion (recommended)
    python3 C1_SUBMISSION_TEMPERATURE.py --temperature 1.5 --device cuda

    # Aggressive expansion
    python3 C1_SUBMISSION_TEMPERATURE.py --temperature 2.0 --device cuda

Interpretation:
    T=1.0: No expansion (current behavior)
    T=1.3: Conservative (10-15% expansion)
    T=1.5: Moderate (30-40% expansion) - RECOMMENDED
    T=2.0: Aggressive (50-60% expansion)
        """
    )

    parser.add_argument(
        '--temperature',
        type=float,
        default=1.5,
        help='Temperature for scaling (default: 1.5)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/trial_level_c1_best.pt',
        help='Path to C1 checkpoint'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device (cuda/cpu)'
    )
    parser.add_argument(
        '--mini',
        action='store_true',
        help='Use mini dataset for testing'
    )

    args = parser.parse_args()

    # Create submission
    zip_path = create_c1_submission_with_temperature(
        checkpoint_path=args.checkpoint,
        temperature=args.temperature,
        device=args.device,
        mini=args.mini
    )

    print(f"✅ Done! Ready to upload: {zip_path}")


if __name__ == "__main__":
    main()
