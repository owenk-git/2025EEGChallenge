"""
Create submission with N-Fold Ensemble

Loads ALL K fold checkpoints (best_k1, best_k2, ..., best_kN) and
averages their predictions at test time.

For K=5 folds:
- Load fold_0_best.pth, fold_1_best.pth, ..., fold_4_best.pth
- Each was the best model on its respective validation set
- Ensemble all 5 models at inference time
- Average predictions → more robust than any single fold

This typically improves performance by 2-5% over single best fold.

Usage:
    python create_kfold_ensemble_submission.py \
      --model erp_mlp \
      --c1_dir checkpoints_kfold/erp_mlp_c1 \
      --c2_dir checkpoints_kfold/erp_mlp_c2 \
      --name erp_kfold_ensemble
"""

import argparse
import zipfile
import shutil
from pathlib import Path
from datetime import datetime
import json


def get_model_code(model_name):
    """Read the model code from file"""
    model_files = {
        'erp_mlp': 'models/erp_features_mlp.py',
        'cnn_ensemble': 'models/cnn_ensemble.py',
        'eegnex_improved': 'models/eegnex_augmented.py'
    }

    if model_name not in model_files:
        raise ValueError(f"Unknown model: {model_name}")

    with open(model_files[model_name], 'r') as f:
        code = f.read()

    # Extract only the model classes (remove if __name__ == "__main__" section)
    lines = code.split('\n')
    model_lines = []

    for line in lines:
        if line.startswith('if __name__'):
            break
        model_lines.append(line)

    # Find first import and remove everything before it
    code = '\n'.join(model_lines)
    import_idx = code.find('import torch')
    if import_idx > 0:
        code = code[import_idx:]

    return code


def create_submission_code(model_name, n_folds_c1, n_folds_c2):
    """Generate submission.py with N-fold ensemble"""

    # Get model code
    model_code = get_model_code(model_name)

    # Model class names
    if model_name == 'erp_mlp':
        model_class = 'ERPMLP'
    elif model_name == 'cnn_ensemble':
        model_class = 'CNNEnsemble'
    elif model_name == 'eegnex_improved':
        model_class = 'EEGNeXImproved'
    else:
        raise ValueError(f"Unknown model: {model_name}")

    submission_code = f'''"""
EEG Challenge 2025 - {n_folds_c1}-Fold Ensemble Submission
Generated: {datetime.now().strftime("%Y%m%d_%H%M")}

Model: {model_name}
C1 Folds: {n_folds_c1}
C2 Folds: {n_folds_c2}

Strategy: Load all fold checkpoints and average predictions at test time.
This improves robustness and typically gains 2-5% over single best fold.
"""

from pathlib import Path
import torch
import torch.nn as nn
import numpy as np

# ============================================================================
# MODEL CODE
# ============================================================================

{model_code}

# ============================================================================
# SUBMISSION CLASS
# ============================================================================

class Submission:
    """
    N-Fold Ensemble Submission

    Loads {n_folds_c1} models for C1 and {n_folds_c2} models for C2.
    Averages predictions across all folds for robust inference.
    """

    def __init__(self, SFREQ, DEVICE):
        self.sfreq = SFREQ
        self.device = DEVICE
        self.model_path = self.load_model_path()

        # Number of folds
        self.n_folds_c1 = {n_folds_c1}
        self.n_folds_c2 = {n_folds_c2}

        print(f"🔄 Loading {{self.n_folds_c1}}-fold ensemble for C1")
        print(f"🔄 Loading {{self.n_folds_c2}}-fold ensemble for C2")

    def load_model_path(self):
        """Find model directory"""
        # Competition server structure
        model_dir = Path('/app/program')
        if not model_dir.exists():
            model_dir = Path('.')
        return model_dir

    def get_model_challenge_1(self):
        """
        Load C1 ensemble: {n_folds_c1} models

        Returns a list of models that will be ensembled.
        """
        models = []

        for fold_idx in range(self.n_folds_c1):
            # Create model
            model = {model_class}(
                n_channels=129,
                sfreq=100,
                challenge_name='c1',
                output_range=(0.5, 1.5)
            )

            # Load checkpoint
            checkpoint_name = f'c1_fold_{{fold_idx}}.pth'
            checkpoint_path = self.model_path / checkpoint_name

            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                model.to(self.device)
                models.append(model)
                print(f"✅ Loaded C1 fold {{fold_idx}}")
            except Exception as e:
                print(f"⚠️  Failed to load C1 fold {{fold_idx}}: {{e}}")
                # Continue even if one fold fails
                pass

        if len(models) == 0:
            raise RuntimeError("Failed to load any C1 models!")

        print(f"✅ Loaded {{len(models)}}/{n_folds_c1} C1 models")

        # Return ensemble wrapper
        return EnsembleModel(models, self.device)

    def get_model_challenge_2(self):
        """
        Load C2 ensemble: {n_folds_c2} models

        Returns a list of models that will be ensembled.
        """
        models = []

        for fold_idx in range(self.n_folds_c2):
            # Create model
            model = {model_class}(
                n_channels=129,
                sfreq=100,
                challenge_name='c2',
                output_range=(-3, 3)
            )

            # Load checkpoint
            checkpoint_name = f'c2_fold_{{fold_idx}}.pth'
            checkpoint_path = self.model_path / checkpoint_name

            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                model.to(self.device)
                models.append(model)
                print(f"✅ Loaded C2 fold {{fold_idx}}")
            except Exception as e:
                print(f"⚠️  Failed to load C2 fold {{fold_idx}}: {{e}}")
                # Continue even if one fold fails
                pass

        if len(models) == 0:
            raise RuntimeError("Failed to load any C2 models!")

        print(f"✅ Loaded {{len(models)}}/{n_folds_c2} C2 models")

        # Return ensemble wrapper
        return EnsembleModel(models, self.device)


class EnsembleModel(nn.Module):
    """
    Ensemble wrapper that averages predictions from multiple models
    """

    def __init__(self, models, device):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.device = device

    def forward(self, x):
        """
        Average predictions from all models

        Args:
            x: (batch, n_channels, n_times)

        Returns:
            (batch, 1) averaged predictions
        """
        # Collect predictions from all models
        predictions = []

        for model in self.models:
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)

        # Average predictions
        avg_pred = torch.stack(predictions).mean(dim=0)

        return avg_pred
'''

    return submission_code


def main():
    parser = argparse.ArgumentParser(description='Create N-Fold Ensemble Submission')
    parser.add_argument('--model', type=str, required=True,
                       choices=['erp_mlp', 'cnn_ensemble', 'eegnex_improved'],
                       help='Model architecture')
    parser.add_argument('--c1_dir', type=str, required=True,
                       help='Directory with C1 fold checkpoints')
    parser.add_argument('--c2_dir', type=str, required=True,
                       help='Directory with C2 fold checkpoints')
    parser.add_argument('--name', type=str, default=None,
                       help='Submission name (default: auto-generated)')

    args = parser.parse_args()

    # Convert to Path objects
    c1_dir = Path(args.c1_dir)
    c2_dir = Path(args.c2_dir)

    if not c1_dir.exists():
        raise FileNotFoundError(f"C1 directory not found: {c1_dir}")
    if not c2_dir.exists():
        raise FileNotFoundError(f"C2 directory not found: {c2_dir}")

    # Find fold checkpoints
    c1_checkpoints = sorted(c1_dir.glob('fold_*_best.pth'))
    c2_checkpoints = sorted(c2_dir.glob('fold_*_best.pth'))

    if len(c1_checkpoints) == 0:
        raise FileNotFoundError(f"No C1 fold checkpoints found in {c1_dir}")
    if len(c2_checkpoints) == 0:
        raise FileNotFoundError(f"No C2 fold checkpoints found in {c2_dir}")

    n_folds_c1 = len(c1_checkpoints)
    n_folds_c2 = len(c2_checkpoints)

    print(f"Found {n_folds_c1} C1 folds: {[f.name for f in c1_checkpoints]}")
    print(f"Found {n_folds_c2} C2 folds: {[f.name for f in c2_checkpoints]}")

    # Generate submission name
    if args.name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        submission_name = f"{timestamp}_{args.model}_kfold{n_folds_c1}"
    else:
        submission_name = args.name

    # Create temp directory
    temp_dir = Path(f'temp_submission_{submission_name}')
    temp_dir.mkdir(exist_ok=True)

    try:
        # Generate submission.py
        print("\n📝 Generating submission.py...")
        submission_code = create_submission_code(args.model, n_folds_c1, n_folds_c2)

        submission_file = temp_dir / 'submission.py'
        with open(submission_file, 'w') as f:
            f.write(submission_code)

        print(f"✅ Created {submission_file}")

        # Copy C1 checkpoints with renamed structure
        print(f"\n📦 Copying {n_folds_c1} C1 checkpoints...")
        for idx, checkpoint in enumerate(c1_checkpoints):
            dest = temp_dir / f'c1_fold_{idx}.pth'
            shutil.copy(checkpoint, dest)
            print(f"   ✅ {checkpoint.name} -> {dest.name}")

        # Copy C2 checkpoints with renamed structure
        print(f"\n📦 Copying {n_folds_c2} C2 checkpoints...")
        for idx, checkpoint in enumerate(c2_checkpoints):
            dest = temp_dir / f'c2_fold_{idx}.pth'
            shutil.copy(checkpoint, dest)
            print(f"   ✅ {checkpoint.name} -> {dest.name}")

        # Create metadata file
        metadata = {
            'model': args.model,
            'n_folds_c1': n_folds_c1,
            'n_folds_c2': n_folds_c2,
            'c1_checkpoints': [str(f) for f in c1_checkpoints],
            'c2_checkpoints': [str(f) for f in c2_checkpoints],
            'created': datetime.now().isoformat()
        }

        metadata_file = temp_dir / 'metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n📋 Created {metadata_file}")

        # Create ZIP
        zip_name = f"{submission_name}.zip"
        print(f"\n🗜️  Creating {zip_name}...")

        with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in temp_dir.glob('*'):
                if file_path.is_file():
                    zipf.write(file_path, file_path.name)
                    print(f"   ✅ Added {file_path.name}")

        print(f"\n✅ Created {zip_name}")
        print(f"\n📊 Submission Stats:")
        print(f"   Model: {args.model}")
        print(f"   C1 Folds: {n_folds_c1}")
        print(f"   C2 Folds: {n_folds_c2}")
        print(f"   Total Checkpoints: {n_folds_c1 + n_folds_c2}")

        # Check file size
        zip_size = Path(zip_name).stat().st_size / (1024 * 1024)
        print(f"   ZIP Size: {zip_size:.2f} MB")

        if zip_size > 100:
            print(f"\n⚠️  WARNING: ZIP is large ({zip_size:.2f} MB)")
            print(f"   Competition may have size limits. Consider reducing number of folds.")

        print(f"\n✅ Ready to submit: {zip_name}")

    finally:
        # Clean up temp directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print(f"\n🧹 Cleaned up {temp_dir}")


if __name__ == '__main__':
    main()
