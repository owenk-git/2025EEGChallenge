"""
Create submission for universal models (FeatureMLP, CNNEnsemble, EEGNeXImproved)

This embeds the actual model code into the submission to avoid architecture mismatches.

Usage:
    python create_universal_submission.py --model feature_mlp --c1 checkpoints_feature_mlp/c1_best.pth --c2 checkpoints_feature_mlp/c2_best.pth
    python create_universal_submission.py --model cnn_ensemble --c1 checkpoints_cnn_ensemble/c1_best.pth --c2 checkpoints_cnn_ensemble/c2_best.pth
"""

import argparse
import zipfile
import shutil
from pathlib import Path
from datetime import datetime


def get_model_code(model_name):
    """Read the model code from file"""
    model_files = {
        'feature_mlp': 'models/feature_mlp.py',
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


SUBMISSION_TEMPLATE = '''"""
EEG Challenge 2025 - {model_display_name} Submission
Generated: {timestamp}

Model: {model_name}
"""

from pathlib import Path
{model_code}


def load_model_path():
    """Return directory containing this submission.py"""
    return Path(__file__).parent


class Submission:
    """Submission class for {model_display_name}"""

    def __init__(self, SFREQ, DEVICE):
        self.sfreq = SFREQ
        self.device = DEVICE
        self.model_path = load_model_path()

        print(f"🤖 {model_display_name} Submission")
        print(f"   Device: {{DEVICE}}")
        print(f"   Sample rate: {{SFREQ}} Hz")

    def get_model_challenge_1(self):
        """Load C1 model"""
        print("📦 Loading C1 {model_display_name}...")

        model = {create_model_c1}

        checkpoint_path = self.model_path / "c1_model.pth"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"C1 checkpoint not found: {{checkpoint_path}}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

        model = model.to(self.device)
        model.eval()

        print(f"✅ C1 {model_display_name} loaded")
        return model

    def get_model_challenge_2(self):
        """Load C2 model"""
        print("📦 Loading C2 {model_display_name}...")

        model = {create_model_c2}

        checkpoint_path = self.model_path / "c2_model.pth"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"C2 checkpoint not found: {{checkpoint_path}}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

        model = model.to(self.device)
        model.eval()

        print(f"✅ C2 {model_display_name} loaded")
        return model
'''


def get_model_creation_code(model_name, challenge):
    """Get the code to create model for specific challenge"""

    if model_name == 'feature_mlp':
        if challenge == 'c1':
            return '''FeatureMLP(
            n_channels=129,
            sfreq=100,
            challenge_name='c1',
            output_range=(0.5, 1.5)
        )'''
        else:
            return '''FeatureMLP(
            n_channels=129,
            sfreq=100,
            challenge_name='c2',
            output_range=(-3, 3)
        )'''

    elif model_name == 'cnn_ensemble':
        if challenge == 'c1':
            return '''CNNEnsemble(
            n_channels=129,
            n_times=200,
            n_classes=1,
            challenge_name='c1',
            output_range=(0.5, 1.5)
        )'''
        else:
            return '''CNNEnsemble(
            n_channels=129,
            n_times=200,
            n_classes=1,
            challenge_name='c2',
            output_range=(-3, 3)
        )'''

    elif model_name == 'eegnex_improved':
        if challenge == 'c1':
            return '''EEGNeXImproved(
            n_channels=129,
            n_times=200,
            n_classes=1,
            challenge_name='c1',
            output_range=(0.5, 1.5),
            use_augmentation=False
        )'''
        else:
            return '''EEGNeXImproved(
            n_channels=129,
            n_times=200,
            n_classes=1,
            challenge_name='c2',
            output_range=(-3, 3),
            use_augmentation=False
        )'''

    else:
        raise ValueError(f"Unknown model: {model_name}")


def create_universal_submission(model_name, c1_checkpoint, c2_checkpoint, output_name=None):
    """Create submission with embedded model code"""

    model_display_names = {
        'feature_mlp': 'Feature MLP',
        'cnn_ensemble': 'CNN Ensemble',
        'eegnex_improved': 'Improved EEGNeX'
    }

    print("="*70)
    print(f"📦 Creating {model_display_names[model_name]} Submission")
    print("="*70)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    if output_name is None:
        output_name = f"{timestamp}_{model_name}_submission.zip"

    # Get model code
    print(f"📄 Reading {model_name} architecture...")
    model_code = get_model_code(model_name)

    # Get model creation code
    create_c1 = get_model_creation_code(model_name, 'c1')
    create_c2 = get_model_creation_code(model_name, 'c2')

    temp_dir = Path(f"temp_{model_name}_{timestamp}")
    temp_dir.mkdir(exist_ok=True)

    try:
        # Write submission.py with embedded model code
        submission_content = SUBMISSION_TEMPLATE.format(
            timestamp=timestamp,
            model_name=model_name,
            model_display_name=model_display_names[model_name],
            model_code=model_code,
            create_model_c1=create_c1,
            create_model_c2=create_c2
        )
        submission_path = temp_dir / "submission.py"
        with open(submission_path, 'w') as f:
            f.write(submission_content)
        print(f"✅ Created submission.py with {model_name} architecture")

        # Copy checkpoints
        if Path(c1_checkpoint).exists():
            shutil.copy(c1_checkpoint, temp_dir / "c1_model.pth")
            print(f"✅ Copied C1 checkpoint: {c1_checkpoint}")
        else:
            print(f"⚠️  C1 checkpoint not found: {c1_checkpoint}")
            return None

        if Path(c2_checkpoint).exists():
            shutil.copy(c2_checkpoint, temp_dir / "c2_model.pth")
            print(f"✅ Copied C2 checkpoint: {c2_checkpoint}")
        else:
            print(f"⚠️  C2 checkpoint not found: {c2_checkpoint}")
            return None

        # Create ZIP
        with zipfile.ZipFile(output_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in temp_dir.rglob('*'):
                if file.is_file():
                    arcname = file.relative_to(temp_dir)
                    zipf.write(file, arcname)

        print(f"\n✅ {model_display_names[model_name]} submission created: {output_name}")

        zip_size = Path(output_name).stat().st_size / (1024 * 1024)
        print(f"   Size: {zip_size:.2f} MB")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("="*70)
    print("🚀 Ready to submit!")
    print(f"   Upload {output_name} to Codabench")
    print("="*70)

    return output_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create universal model submission")

    parser.add_argument('--model', type=str, required=True,
                       choices=['feature_mlp', 'cnn_ensemble', 'eegnex_improved'],
                       help='Model type')
    parser.add_argument('--c1', type=str, required=True,
                       help='Path to C1 checkpoint')
    parser.add_argument('--c2', type=str, required=True,
                       help='Path to C2 checkpoint')
    parser.add_argument('--output', type=str, default=None,
                       help='Output ZIP filename')

    args = parser.parse_args()

    create_universal_submission(
        model_name=args.model,
        c1_checkpoint=args.c1,
        c2_checkpoint=args.c2,
        output_name=args.output
    )
