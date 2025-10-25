"""
Create ensemble submission by averaging Transformer + CNN Ensemble predictions

This combines two different architectures for variance reduction.

Usage:
    python create_ensemble_average_submission.py
"""

import argparse
import zipfile
import shutil
from pathlib import Path
from datetime import datetime


# Read transformer model code
def get_transformer_code():
    with open('models/eeg_transformer.py', 'r') as f:
        code = f.read()
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


# Read CNN ensemble code
def get_cnn_ensemble_code():
    with open('models/cnn_ensemble.py', 'r') as f:
        code = f.read()
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


ENSEMBLE_SUBMISSION = '''"""
EEG Challenge 2025 - Ensemble Averaging Submission
Generated: {timestamp}

Strategy: Average predictions from Transformer + CNN Ensemble
Expected: Variance reduction → better generalization
"""

from pathlib import Path
{transformer_code}

# Separator between models
# ============================================================================

{cnn_code}


def load_model_path():
    """Return directory containing this submission.py"""
    return Path(__file__).parent


class Submission:
    """Ensemble averaging submission"""

    def __init__(self, SFREQ, DEVICE):
        self.sfreq = SFREQ
        self.device = DEVICE
        self.model_path = load_model_path()

        print(f"🎯 Ensemble Averaging Submission")
        print(f"   Models: Transformer + CNN Ensemble")
        print(f"   Device: {{DEVICE}}")

    def get_model_challenge_1(self):
        """Load C1 ensemble (Transformer + CNN Ensemble)"""
        print("📦 Loading C1 Ensemble...")

        # Load Transformer
        transformer = EEGTransformer(
            n_channels=129,
            n_times=200,
            patch_size=10,
            embed_dim=128,
            n_layers=6,
            n_heads=8,
            n_classes=1,
            challenge_name='c1',
            output_range=(0.5, 1.5)
        )
        t_checkpoint = torch.load(
            self.model_path / "c1_transformer.pth",
            map_location=self.device,
            weights_only=False
        )
        transformer.load_state_dict(t_checkpoint['model_state_dict'])
        transformer = transformer.to(self.device)
        transformer.eval()

        # Load CNN Ensemble
        cnn = CNNEnsemble(
            n_channels=129,
            n_times=200,
            n_classes=1,
            challenge_name='c1',
            output_range=(0.5, 1.5)
        )
        cnn_checkpoint = torch.load(
            self.model_path / "c1_cnn.pth",
            map_location=self.device,
            weights_only=False
        )
        cnn.load_state_dict(cnn_checkpoint['model_state_dict'])
        cnn = cnn.to(self.device)
        cnn.eval()

        # Create ensemble wrapper
        class EnsembleModel:
            def __init__(self, model1, model2):
                self.model1 = model1
                self.model2 = model2

            def eval(self):
                self.model1.eval()
                self.model2.eval()
                return self

            def to(self, device):
                self.model1.to(device)
                self.model2.to(device)
                return self

            def __call__(self, x):
                with torch.no_grad():
                    pred1 = self.model1(x)
                    pred2 = self.model2(x)
                    # Average predictions
                    return (pred1 + pred2) / 2

        ensemble = EnsembleModel(transformer, cnn)
        print(f"✅ C1 Ensemble loaded (Transformer + CNN)")
        return ensemble

    def get_model_challenge_2(self):
        """Load C2 ensemble (Transformer + CNN Ensemble)"""
        print("📦 Loading C2 Ensemble...")

        # Load Transformer
        transformer = EEGTransformer(
            n_channels=129,
            n_times=200,
            patch_size=10,
            embed_dim=128,
            n_layers=6,
            n_heads=8,
            n_classes=1,
            challenge_name='c2',
            output_range=(-3, 3)
        )
        t_checkpoint = torch.load(
            self.model_path / "c2_transformer.pth",
            map_location=self.device,
            weights_only=False
        )
        transformer.load_state_dict(t_checkpoint['model_state_dict'])
        transformer = transformer.to(self.device)
        transformer.eval()

        # Load CNN Ensemble
        cnn = CNNEnsemble(
            n_channels=129,
            n_times=200,
            n_classes=1,
            challenge_name='c2',
            output_range=(-3, 3)
        )
        cnn_checkpoint = torch.load(
            self.model_path / "c2_cnn.pth",
            map_location=self.device,
            weights_only=False
        )
        cnn.load_state_dict(cnn_checkpoint['model_state_dict'])
        cnn = cnn.to(self.device)
        cnn.eval()

        # Create ensemble wrapper
        class EnsembleModel:
            def __init__(self, model1, model2):
                self.model1 = model1
                self.model2 = model2

            def eval(self):
                self.model1.eval()
                self.model2.eval()
                return self

            def to(self, device):
                self.model1.to(device)
                self.model2.to(device)
                return self

            def __call__(self, x):
                with torch.no_grad():
                    pred1 = self.model1(x)
                    pred2 = self.model2(x)
                    # Average predictions
                    return (pred1 + pred2) / 2

        ensemble = EnsembleModel(transformer, cnn)
        print(f"✅ C2 Ensemble loaded (Transformer + CNN)")
        return ensemble
'''


def create_ensemble_submission(output_name=None):
    """Create ensemble averaging submission"""

    print("="*70)
    print("📦 Creating Ensemble Averaging Submission")
    print("="*70)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    if output_name is None:
        output_name = f"{timestamp}_ensemble_average.zip"

    # Get model codes
    print("📄 Reading Transformer architecture...")
    transformer_code = get_transformer_code()

    print("📄 Reading CNN Ensemble architecture...")
    cnn_code = get_cnn_ensemble_code()

    temp_dir = Path(f"temp_ensemble_{timestamp}")
    temp_dir.mkdir(exist_ok=True)

    try:
        # Write submission.py
        submission_content = ENSEMBLE_SUBMISSION.format(
            timestamp=timestamp,
            transformer_code=transformer_code,
            cnn_code=cnn_code
        )
        submission_path = temp_dir / "submission.py"
        with open(submission_path, 'w') as f:
            f.write(submission_content)
        print(f"✅ Created submission.py with ensemble code")

        # Copy checkpoints
        checkpoints = [
            ('checkpoints_transformer/c1_transformer_best.pth', 'c1_transformer.pth'),
            ('checkpoints_transformer/c2_transformer_best.pth', 'c2_transformer.pth'),
            ('checkpoints_cnn_ensemble/c1_best.pth', 'c1_cnn.pth'),
            ('checkpoints_cnn_ensemble/c2_best.pth', 'c2_cnn.pth'),
        ]

        for src, dst in checkpoints:
            if Path(src).exists():
                shutil.copy(src, temp_dir / dst)
                print(f"✅ Copied {src}")
            else:
                print(f"⚠️  Checkpoint not found: {src}")
                return None

        # Create ZIP
        with zipfile.ZipFile(output_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in temp_dir.rglob('*'):
                if file.is_file():
                    arcname = file.relative_to(temp_dir)
                    zipf.write(file, arcname)

        print(f"\n✅ Ensemble submission created: {output_name}")

        zip_size = Path(output_name).stat().st_size / (1024 * 1024)
        print(f"   Size: {zip_size:.2f} MB")

        print(f"\n🎯 Strategy:")
        print(f"   Average Transformer + CNN Ensemble predictions")
        print(f"   Expected: 1.08-1.09 (variance reduction)")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("="*70)
    print("🚀 Ready to submit!")
    print(f"   Upload {output_name} to Codabench")
    print("="*70)

    return output_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create ensemble averaging submission")
    parser.add_argument('--output', type=str, default=None, help='Output ZIP filename')
    args = parser.parse_args()

    create_ensemble_submission(output_name=args.output)
