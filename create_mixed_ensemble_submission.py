"""
Create Mixed Ensemble Submission

Flexible ensemble that allows:
- Multiple C1 models from different sources (different training runs, architectures, etc.)
- Single or multiple C2 models
- Mix K-fold checkpoints with standalone checkpoints

Usage:
    # Ensemble 5 different C1 models + 1 C2 model
    python3 create_mixed_ensemble_submission.py \
      --c1 checkpoints_kfold/erp_mlp_c1/fold_0_best.pth \
      --c1 checkpoints_kfold/erp_mlp_c1/fold_1_best.pth \
      --c1 checkpoints_kfold/erp_mlp_c1/fold_2_best.pth \
      --c1 checkpoints_kfold/erp_mlp_c1/fold_3_best.pth \
      --c1 checkpoints_kfold/erp_mlp_c1/fold_4_best.pth \
      --c2 checkpoints/domain_adaptation_c2_best.pt \
      --name mixed_ensemble

    # Can also mix different model types for C1
    python3 create_mixed_ensemble_submission.py \
      --c1 checkpoints/erp_mlp_c1.pth \
      --c1 checkpoints/cnn_ensemble_c1.pth \
      --c1 checkpoints/transformer_c1.pth \
      --c2 checkpoints/domain_adaptation_c2_best.pt \
      --name heterogeneous_ensemble
"""

import argparse
import zipfile
import shutil
import torch
from pathlib import Path
from datetime import datetime
import json


def detect_model_architecture(checkpoint_path):
    """
    Detect model architecture from checkpoint

    Returns model code and class name
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # First check if model_name is stored in checkpoint (BEST METHOD)
    if 'model_name' in checkpoint:
        model_name = checkpoint['model_name']
        print(f"  Found model_name in checkpoint: {model_name}")
        if model_name == 'erp_mlp':
            return 'erp_mlp', 'ERPMLP'
        elif model_name == 'cnn_ensemble':
            return 'cnn_ensemble', 'CNNEnsemble'
        elif model_name == 'eegnex_improved':
            return 'eegnex_improved', 'EEGNeXImproved'
        else:
            print(f"  WARNING: Unknown model_name in checkpoint: {model_name}")
            # Still try to detect from state_dict
    else:
        print(f"  No model_name in checkpoint, trying state_dict detection...")

    state_dict = checkpoint.get('model_state_dict', checkpoint)

    # Check for characteristic keys
    keys = list(state_dict.keys())

    print(f"  DEBUG: Detecting architecture from {len(keys)} keys")
    print(f"  DEBUG: Sample keys: {keys[:5] if len(keys) > 5 else keys}")

    # Domain Adaptation EEGNeX: has 'subject_discriminator'
    if any('subject_discriminator' in k for k in keys):
        return 'domain_adaptation_eegnex', 'DomainAdaptationEEGNeX'

    # CNN Ensemble: has branch patterns (spatial_branch, temporal_branch, hybrid_branch)
    if any('spatial_branch' in k or 'temporal_branch' in k or 'hybrid_branch' in k for k in keys):
        return 'cnn_ensemble', 'CNNEnsemble'

    # ERP MLP: has 'mlp' and small number of params
    if any('mlp' in k for k in keys) and len(keys) < 30:
        return 'erp_mlp', 'ERPMLP'

    # EEGNeX: has 'depthwise' or 'pointwise' or 'block'
    if any('depthwise' in k or 'pointwise' in k for k in keys):
        return 'eegnex_improved', 'EEGNeXImproved'

    # Check for block structure (EEGNeX pattern)
    if any('block' in k for k in keys):
        return 'eegnex_improved', 'EEGNeXImproved'

    # Trial Level RT Predictor
    if any('pre_encoder' in k and 'post_encoder' in k for k in keys):
        return 'trial_level_rt', 'TrialLevelRTPredictor'

    # Print keys for debugging
    print(f"  WARNING: Could not detect model type. Keys: {keys[:10]}")

    # Default fallback - assume EEGNeX if it has many layers
    if len(keys) > 50:
        return 'eegnex_improved', 'EEGNeXImproved'
    else:
        return 'cnn_ensemble', 'CNNEnsemble'


def get_model_code(model_name):
    """Read model code from file"""
    model_files = {
        'erp_mlp': 'models/erp_features_mlp.py',
        'cnn_ensemble': 'models/cnn_ensemble.py',
        'eegnex_improved': 'models/eegnex_augmented.py',
        'domain_adaptation_eegnex': 'models/domain_adaptation_eegnex.py',
        'trial_level_rt': 'models/trial_level_rt_predictor.py'
    }

    if model_name not in model_files:
        return None

    model_file = Path(model_files[model_name])
    if not model_file.exists():
        return None

    with open(model_file, 'r') as f:
        code = f.read()

    # Extract only model classes
    lines = code.split('\n')
    model_lines = []

    for line in lines:
        if line.startswith('if __name__'):
            break
        model_lines.append(line)

    code = '\n'.join(model_lines)
    import_idx = code.find('import torch')
    if import_idx > 0:
        code = code[import_idx:]

    return code


def create_submission_code(c1_models_info, c2_models_info):
    """
    Generate submission.py with mixed ensemble

    Args:
        c1_models_info: List of (checkpoint_path, model_name, class_name)
        c2_models_info: List of (checkpoint_path, model_name, class_name)
    """

    # Collect unique model codes
    # c1_models_info and c2_models_info are (model_name, class_name) tuples
    all_model_names = set([info[0] for info in c1_models_info + c2_models_info])
    model_codes = {}

    print(f"\nLoading model code for: {all_model_names}")

    for model_name in all_model_names:
        if model_name == 'unknown':
            raise ValueError(f"Cannot create submission with unknown model types! Please check model detection.")

        code = get_model_code(model_name)
        if code:
            model_codes[model_name] = code
            print(f"  ✅ Loaded code for {model_name}")
        else:
            raise ValueError(f"Failed to load model code for {model_name}. Check if model file exists.")

    if len(model_codes) == 0:
        raise ValueError("No model code loaded! Cannot create submission.")

    # Combine all model codes
    combined_model_code = '\n\n'.join([
        f'# ============================================================================\n'
        f'# {name.upper()} MODEL\n'
        f'# ============================================================================\n\n'
        f'{code}'
        for name, code in model_codes.items()
    ])

    # Build submission code using string concatenation to avoid f-string nesting issues
    # c1_models_info and c2_models_info are already (model_name, class_name) tuples
    n_c1 = len(c1_models_info)
    n_c2 = len(c2_models_info)
    c1_types = ", ".join(set(info[0] for info in c1_models_info))
    c2_types = ", ".join(set(info[0] for info in c2_models_info))
    c1_configs_str = str(c1_models_info)
    c2_configs_str = str(c2_models_info)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    submission_code = f'''"""
EEG Challenge 2025 - Mixed Ensemble Submission
Generated: {timestamp}

C1 Models: {n_c1} ({c1_types})
C2 Models: {n_c2} ({c2_types})

Strategy: Ensemble multiple models at test time by averaging predictions.
"""

from pathlib import Path
import torch
import torch.nn as nn
import numpy as np

# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

{combined_model_code}

# ============================================================================
# SUBMISSION CLASS
# ============================================================================

class Submission:
    """
    Mixed Ensemble Submission

    C1: {n_c1} models
    C2: {n_c2} models
    """

    def __init__(self, SFREQ, DEVICE):
        self.sfreq = SFREQ
        self.device = DEVICE
        self.model_path = self.load_model_path()

        print(f"🔄 Loading {n_c1} C1 models")
        print(f"🔄 Loading {n_c2} C2 models")

    def load_model_path(self):
        """Find model directory - checkpoints are in same dir as submission.py"""
        # On competition server, files are in /app/input/res/
        # Locally, they're in current directory
        # Use __file__ to find where this script is located
        script_dir = Path(__file__).parent
        return script_dir

    def get_model_challenge_1(self):
        """Load C1 ensemble"""
        models = []

        # Model configurations
        c1_configs = {c1_configs_str}

        for idx, (model_name, class_name) in enumerate(c1_configs):
            # Create model based on class name
            if class_name == 'ERPMLP':
                model = ERPMLP(n_channels=129, sfreq=100, challenge_name='c1', output_range=(0.5, 1.5))
            elif class_name == 'CNNEnsemble':
                model = CNNEnsemble(n_channels=129, n_times=200, challenge_name='c1')
            elif class_name == 'EEGNeXImproved':
                model = EEGNeXImproved(n_channels=129, n_times=200, challenge_name='c1')
            elif class_name == 'DomainAdaptationEEGNeX':
                model = DomainAdaptationEEGNeX(n_channels=129, n_times=200, challenge='c1', num_subjects=100, output_range=(0.5, 1.5))
            elif class_name == 'TrialLevelRTPredictor':
                model = TrialLevelRTPredictor(n_channels=129, trial_length=200, pre_stim_points=50)
            else:
                print("⚠️  Unknown model class: " + str(class_name))
                continue

            # Load checkpoint
            checkpoint_name = 'c1_model_' + str(idx) + '.pth'
            checkpoint_path = self.model_path / checkpoint_name

            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
                model.eval()
                model.to(self.device)
                models.append(model)
                print("✅ Loaded C1 model " + str(idx) + " (" + str(model_name) + ")")
            except Exception as e:
                print("⚠️  Failed to load C1 model " + str(idx) + ": " + str(e))
                pass

        if len(models) == 0:
            raise RuntimeError("Failed to load any C1 models!")

        print("✅ Loaded " + str(len(models)) + " C1 models")
        return EnsembleModel(models, self.device)

    def get_model_challenge_2(self):
        """Load C2 ensemble"""
        models = []

        # Model configurations
        c2_configs = {c2_configs_str}

        for idx, (model_name, class_name) in enumerate(c2_configs):
            # Create model based on class name
            if class_name == 'ERPMLP':
                model = ERPMLP(n_channels=129, sfreq=100, challenge_name='c2', output_range=(-3, 3))
            elif class_name == 'CNNEnsemble':
                model = CNNEnsemble(n_channels=129, n_times=200, challenge_name='c2')
            elif class_name == 'EEGNeXImproved':
                model = EEGNeXImproved(n_channels=129, n_times=200, challenge_name='c2')
            elif class_name == 'DomainAdaptationEEGNeX':
                model = DomainAdaptationEEGNeX(n_channels=129, n_times=200, challenge='c2', num_subjects=100, output_range=(-3, 3))
            elif class_name == 'TrialLevelRTPredictor':
                print("⚠️  TrialLevelRTPredictor not suitable for C2")
                continue
            else:
                print("⚠️  Unknown model class: " + str(class_name))
                continue

            # Load checkpoint
            checkpoint_name = 'c2_model_' + str(idx) + '.pth'
            checkpoint_path = self.model_path / checkpoint_name

            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
                model.eval()
                model.to(self.device)
                models.append(model)
                print("✅ Loaded C2 model " + str(idx) + " (" + str(model_name) + ")")
            except Exception as e:
                print("⚠️  Failed to load C2 model " + str(idx) + ": " + str(e))
                pass

        if len(models) == 0:
            raise RuntimeError("Failed to load any C2 models!")

        print("✅ Loaded " + str(len(models)) + " C2 models")
        return EnsembleModel(models, self.device)


class EnsembleModel(nn.Module):
    """Ensemble wrapper that averages predictions"""

    def __init__(self, models, device):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.device = device

    def forward(self, x):
        """Average predictions from all models"""
        predictions = []

        for model in self.models:
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)

        avg_pred = torch.stack(predictions).mean(dim=0)
        return avg_pred
'''

    return submission_code


def main():
    parser = argparse.ArgumentParser(description='Create Mixed Ensemble Submission')
    parser.add_argument('--c1', action='append', required=True,
                       help='C1 checkpoint path (can specify multiple times)')
    parser.add_argument('--c2', action='append', required=True,
                       help='C2 checkpoint path (can specify multiple times)')
    parser.add_argument('--name', type=str, required=True,
                       help='Submission name')

    args = parser.parse_args()

    print("="*60)
    print("MIXED ENSEMBLE SUBMISSION CREATOR")
    print("="*60)

    # Process C1 checkpoints
    c1_models_info = []
    print(f"\n📦 Processing {len(args.c1)} C1 checkpoints...")
    for idx, ckpt_path in enumerate(args.c1):
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            print(f"⚠️  C1 checkpoint not found: {ckpt_path}")
            continue

        model_name, class_name = detect_model_architecture(ckpt_path)
        c1_models_info.append((str(ckpt_path), model_name, class_name))
        print(f"   {idx+1}. {ckpt_path.name} → {model_name} ({class_name})")

    # Process C2 checkpoints
    c2_models_info = []
    print(f"\n📦 Processing {len(args.c2)} C2 checkpoints...")
    for idx, ckpt_path in enumerate(args.c2):
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            print(f"⚠️  C2 checkpoint not found: {ckpt_path}")
            continue

        model_name, class_name = detect_model_architecture(ckpt_path)
        c2_models_info.append((str(ckpt_path), model_name, class_name))
        print(f"   {idx+1}. {ckpt_path.name} → {model_name} ({class_name})")

    if len(c1_models_info) == 0:
        raise ValueError("No valid C1 checkpoints found!")
    if len(c2_models_info) == 0:
        raise ValueError("No valid C2 checkpoints found!")

    print(f"\n✅ Total: {len(c1_models_info)} C1 models + {len(c2_models_info)} C2 models")

    # Create temp directory
    temp_dir = Path(f'temp_submission_{args.name}')
    temp_dir.mkdir(exist_ok=True)

    try:
        # Generate submission.py
        print(f"\n📝 Generating submission.py...")

        # For submission code, only need (model_name, class_name)
        c1_configs = [(info[1], info[2]) for info in c1_models_info]
        c2_configs = [(info[1], info[2]) for info in c2_models_info]

        submission_code = create_submission_code(c1_configs, c2_configs)

        submission_file = temp_dir / 'submission.py'
        with open(submission_file, 'w') as f:
            f.write(submission_code)

        print(f"✅ Created {submission_file}")

        # Copy C1 checkpoints
        print(f"\n📦 Copying {len(c1_models_info)} C1 checkpoints...")
        for idx, (ckpt_path, model_name, class_name) in enumerate(c1_models_info):
            dest = temp_dir / f'c1_model_{idx}.pth'
            shutil.copy(ckpt_path, dest)
            print(f"   ✅ {Path(ckpt_path).name} → {dest.name}")

        # Copy C2 checkpoints
        print(f"\n📦 Copying {len(c2_models_info)} C2 checkpoints...")
        for idx, (ckpt_path, model_name, class_name) in enumerate(c2_models_info):
            dest = temp_dir / f'c2_model_{idx}.pth'
            shutil.copy(ckpt_path, dest)
            print(f"   ✅ {Path(ckpt_path).name} → {dest.name}")

        # Create metadata
        metadata = {
            'name': args.name,
            'c1_models': [{'path': info[0], 'type': info[1], 'class': info[2]} for info in c1_models_info],
            'c2_models': [{'path': info[0], 'type': info[1], 'class': info[2]} for info in c2_models_info],
            'created': datetime.now().isoformat()
        }

        metadata_file = temp_dir / 'metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Create ZIP
        zip_name = f"{args.name}.zip"
        print(f"\n🗜️  Creating {zip_name}...")

        with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in temp_dir.glob('*'):
                if file_path.is_file():
                    zipf.write(file_path, file_path.name)
                    print(f"   ✅ Added {file_path.name}")

        # Check size
        zip_size = Path(zip_name).stat().st_size / (1024 * 1024)

        print(f"\n✅ Created {zip_name}")
        print(f"\n📊 Submission Stats:")
        print(f"   C1 Models: {len(c1_models_info)}")
        print(f"   C2 Models: {len(c2_models_info)}")
        print(f"   ZIP Size: {zip_size:.2f} MB")

        if zip_size > 100:
            print(f"\n⚠️  WARNING: Large ZIP ({zip_size:.2f} MB)")

        print(f"\n✅ Ready to submit: {zip_name}")

    finally:
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print(f"\n🧹 Cleaned up {temp_dir}")


if __name__ == '__main__':
    main()
