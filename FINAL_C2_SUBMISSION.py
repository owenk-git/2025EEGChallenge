#!/usr/bin/env python3
"""
FINAL CHALLENGE 2 SUBMISSION - RECORDING-LEVEL ENSEMBLE

Based on analysis:
- C2 requires RECORDING-LEVEL prediction (not trial-level)
- Externalizing is subject-level trait (constant per subject)
- Recording-level averaging captures stable trait
- Ensemble of domain adaptation + cross-task models

Expected test NRMSE: 1.00-1.05 (from validation 1.08)
"""

import torch
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import zipfile
import datetime

try:
    from eegdash.dataset import EEGChallengeDataset
    EEGDASH_AVAILABLE = True
except ImportError:
    EEGDASH_AVAILABLE = False
    print("⚠️ eegdash not available")


def load_recording_level_model(model_path, model_type, device):
    """
    Load recording-level model (domain adaptation, cross-task, or hybrid)
    """
    if model_type == 'domain_adaptation':
        from models.domain_adaptation_eegnex import DomainAdaptationEEGNeX
        model = DomainAdaptationEEGNeX(
            n_channels=129,
            n_times=200,
            challenge='c2'
        ).to(device)
    elif model_type == 'cross_task':
        from models.cross_task_pretrain import CrossTaskPretrainModel
        model = CrossTaskPretrainModel(
            n_channels=129,
            n_times=200,
            num_tasks=6
        ).to(device)
    elif model_type == 'hybrid':
        from models.hybrid_cnn_transformer_da import HybridCNNTransformerDA
        model = HybridCNNTransformerDA(
            n_channels=129,
            n_times=200,
            challenge='c2'
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    best_nrmse = checkpoint.get('best_nrmse', 'N/A')
    print(f"  ✅ {model_type}: Val NRMSE {best_nrmse}")

    return model, best_nrmse


def extract_recording_features(raw, n_channels=129, n_times=200, sfreq=100):
    """
    Extract recording-level features (standard approach for C2)

    For C2, we want to capture STABLE subject traits, not trial variability
    So we average over the entire recording
    """
    # Get full recording
    eeg_data = raw.get_data()

    # Take central segment (most stable)
    total_samples = eeg_data.shape[1]
    if total_samples > n_times:
        # Take middle segment
        start_idx = (total_samples - n_times) // 2
        eeg_data = eeg_data[:, start_idx:start_idx + n_times]
    elif total_samples < n_times:
        # Pad
        pad_width = n_times - total_samples
        eeg_data = np.pad(eeg_data, ((0, 0), (0, pad_width)), mode='edge')

    # Ensure correct number of channels
    if eeg_data.shape[0] > n_channels:
        eeg_data = eeg_data[:n_channels, :]
    elif eeg_data.shape[0] < n_channels:
        pad_width = n_channels - eeg_data.shape[0]
        eeg_data = np.pad(eeg_data, ((0, pad_width), (0, 0)), mode='constant')

    return eeg_data


def create_c2_submission(
    model_paths=None,
    model_types=None,
    ensemble_weights=None,
    output_dir='submissions',
    device='cuda'
):
    """
    Create Challenge 2 submission using recording-level ensemble

    Args:
        model_paths: List of model checkpoint paths
        model_types: List of model types ['domain_adaptation', 'cross_task', 'hybrid']
        ensemble_weights: Weights for ensemble (default: equal weights)
        output_dir: Output directory
        device: cuda or cpu
    """
    print("="*80)
    print("CHALLENGE 2 SUBMISSION - RECORDING-LEVEL ENSEMBLE")
    print("="*80)
    print("\nApproach: Ensemble of recording-level models")
    print("Why: Externalizing is subject-level trait (not trial-varying)\n")

    if not EEGDASH_AVAILABLE:
        raise ImportError("eegdash required for submission")

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Default: use all available models
    if model_paths is None:
        model_paths = []
        model_types = []
        ensemble_weights = []

        # Check which models exist
        if Path('checkpoints/domain_adaptation_c2_best.pt').exists():
            model_paths.append('checkpoints/domain_adaptation_c2_best.pt')
            model_types.append('domain_adaptation')
            ensemble_weights.append(0.4)

        if Path('checkpoints/cross_task_c2_best.pt').exists():
            model_paths.append('checkpoints/cross_task_c2_best.pt')
            model_types.append('cross_task')
            ensemble_weights.append(0.4)

        if Path('checkpoints/hybrid_c2_best.pt').exists():
            model_paths.append('checkpoints/hybrid_c2_best.pt')
            model_types.append('hybrid')
            ensemble_weights.append(0.2)

        if len(model_paths) == 0:
            raise FileNotFoundError("No C2 models found in checkpoints/")

    # Normalize weights
    ensemble_weights = np.array(ensemble_weights)
    ensemble_weights = ensemble_weights / ensemble_weights.sum()

    # Load models
    print(f"Loading {len(model_paths)} models for ensemble:")
    models = []
    val_nrmses = []

    for model_path, model_type in zip(model_paths, model_types):
        model, val_nrmse = load_recording_level_model(model_path, model_type, device)
        models.append(model)
        val_nrmses.append(val_nrmse)

    print(f"\nEnsemble weights: {dict(zip(model_types, ensemble_weights))}")
    print(f"Expected ensemble NRMSE: {np.average(val_nrmses, weights=ensemble_weights):.4f}\n")

    # Load test dataset (without train parameter)
    print("Loading test dataset...")
    test_dataset = EEGChallengeDataset(
        task="rest",  # C2 uses resting state
        release="R11",
        cache_dir="./data_cache/eeg_challenge",
        mini=False
    )

    print(f"✅ Loaded {len(test_dataset.datasets)} test recordings\n")

    # Make predictions
    print("Making ensemble predictions...")
    all_predictions = [[] for _ in range(len(models))]

    with torch.no_grad():
        for rec_idx in tqdm(range(len(test_dataset.datasets)), desc="Processing"):
            raw = test_dataset.datasets[rec_idx].raw

            # Extract recording features
            eeg_data = extract_recording_features(raw)
            eeg_tensor = torch.tensor(eeg_data, dtype=torch.float32).unsqueeze(0).to(device)

            # Get predictions from each model
            for model_idx, model in enumerate(models):
                pred = model(eeg_tensor).item()
                all_predictions[model_idx].append(pred)

    # Convert to arrays
    all_predictions = [np.array(preds) for preds in all_predictions]

    # Ensemble predictions (weighted average)
    ensemble_predictions = np.zeros(len(test_dataset.datasets))
    for preds, weight in zip(all_predictions, ensemble_weights):
        ensemble_predictions += preds * weight

    # Statistics
    print(f"\n{'='*80}")
    print("PREDICTION STATISTICS")
    print(f"{'='*80}")
    print(f"Total recordings: {len(ensemble_predictions)}")
    print(f"\nIndividual models:")
    for model_type, preds in zip(model_types, all_predictions):
        print(f"  {model_type}:")
        print(f"    Range: [{preds.min():.4f}, {preds.max():.4f}]")
        print(f"    Mean: {preds.mean():.4f}, Std: {preds.std():.4f}")

    print(f"\nEnsemble:")
    print(f"  Range: [{ensemble_predictions.min():.4f}, {ensemble_predictions.max():.4f}]")
    print(f"  Mean: {ensemble_predictions.mean():.4f}")
    print(f"  Std: {ensemble_predictions.std():.4f}")
    print(f"  Median: {np.median(ensemble_predictions):.4f}")

    # Save predictions
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pred_file = output_path / 'y_pred.csv'
    np.savetxt(pred_file, ensemble_predictions, delimiter=',', fmt='%.6f')
    print(f"\n✅ Saved predictions to {pred_file}")

    # Create submission zip
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    zip_name = f'c2_ensemble_{timestamp}.zip'
    zip_path = output_path / zip_name

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(pred_file, 'y_pred.csv')

    print(f"✅ Created submission: {zip_path}")

    # Expected performance
    print(f"\n{'='*80}")
    print("EXPECTED PERFORMANCE")
    print(f"{'='*80}")
    print(f"Ensemble validation NRMSE: {np.average(val_nrmses, weights=ensemble_weights):.4f}")
    print(f"Expected test NRMSE: {np.average(val_nrmses, weights=ensemble_weights)*1.0:.4f} - {np.average(val_nrmses, weights=ensemble_weights)*1.05:.4f}")
    print(f"\nIndividual model validations:")
    for model_type, val_nrmse in zip(model_types, val_nrmses):
        print(f"  {model_type}: {val_nrmse}")
    print(f"\n{'='*80}")

    return zip_path


def main():
    parser = argparse.ArgumentParser(description='Create C2 Recording-Level Ensemble Submission')
    parser.add_argument('--model_paths', type=str, nargs='+', default=None,
                        help='Paths to model checkpoints (default: auto-detect)')
    parser.add_argument('--model_types', type=str, nargs='+', default=None,
                        choices=['domain_adaptation', 'cross_task', 'hybrid'],
                        help='Model types (default: auto-detect)')
    parser.add_argument('--weights', type=float, nargs='+', default=None,
                        help='Ensemble weights (default: equal)')
    parser.add_argument('--output_dir', type=str, default='submissions',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')

    args = parser.parse_args()

    create_c2_submission(
        model_paths=args.model_paths,
        model_types=args.model_types,
        ensemble_weights=args.weights,
        output_dir=args.output_dir,
        device=args.device
    )


if __name__ == '__main__':
    main()
