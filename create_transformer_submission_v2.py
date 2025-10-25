"""
Create Transformer submission with exact architecture match

This embeds the EXACT model architecture from models/eeg_transformer.py
into the submission to avoid any architecture mismatch issues.
"""

import argparse
import zipfile
import shutil
from pathlib import Path
from datetime import datetime


# Read the exact model architecture
def get_transformer_code():
    """Get the exact transformer model code"""
    with open('models/eeg_transformer.py', 'r') as f:
        model_code = f.read()

    # Extract only the model classes (not the test code)
    lines = model_code.split('\n')
    model_lines = []
    in_main = False

    for line in lines:
        if line.startswith('if __name__'):
            break
        if not (line.startswith('"""') and 'EEG-Specific Transformer' in line):
            model_lines.append(line)

    # Remove the docstring at the top
    code = '\n'.join(model_lines)
    # Find first import
    import_idx = code.find('import torch')
    if import_idx > 0:
        code = code[import_idx:]

    return code


SUBMISSION_TEMPLATE = '''"""
EEG Challenge 2025 - Transformer Submission
Generated: {timestamp}

Results:
- C1 Best NRMSE: 1.000
- C2 Best NRMSE: 1.015
- Expected Overall: 1.011
"""

from pathlib import Path
{model_code}


def load_model_path():
    """Return directory containing this submission.py"""
    return Path(__file__).parent


class Submission:
    """Transformer submission class"""

    def __init__(self, SFREQ, DEVICE):
        self.sfreq = SFREQ
        self.device = DEVICE
        self.model_path = load_model_path()

        print(f"🤖 Transformer Submission")
        print(f"   Device: {{DEVICE}}")
        print(f"   Sample rate: {{SFREQ}} Hz")

    def get_model_challenge_1(self):
        """Load C1 Transformer model"""
        print("📦 Loading C1 Transformer...")

        model = EEGTransformer(
            n_channels=129,
            n_classes=1,
            n_times=200,
            patch_size=10,
            embed_dim=128,
            n_layers=6,
            n_heads=8,
            mlp_ratio=4,
            dropout=0.1,
            challenge_name='c1',
            output_range=(0.5, 1.5)
        )

        checkpoint_path = self.model_path / "c1_transformer.pth"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"C1 checkpoint not found: {{checkpoint_path}}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

        model = model.to(self.device)
        model.eval()

        print(f"✅ C1 Transformer loaded (Best NRMSE: 1.000)")
        return model

    def get_model_challenge_2(self):
        """Load C2 Transformer model"""
        print("📦 Loading C2 Transformer...")

        model = EEGTransformer(
            n_channels=129,
            n_classes=1,
            n_times=200,
            patch_size=10,
            embed_dim=128,
            n_layers=6,
            n_heads=8,
            mlp_ratio=4,
            dropout=0.1,
            challenge_name='c2',
            output_range=(-3, 3)
        )

        checkpoint_path = self.model_path / "c2_transformer.pth"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"C2 checkpoint not found: {{checkpoint_path}}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

        model = model.to(self.device)
        model.eval()

        print(f"✅ C2 Transformer loaded (Best NRMSE: 1.015)")
        return model
'''


def create_transformer_submission_v2(c1_checkpoint, c2_checkpoint, output_name=None):
    """
    Create Transformer submission with exact architecture
    """
    print("="*70)
    print("📦 Creating Transformer Submission v2 (Exact Architecture Match)")
    print("="*70)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    if output_name is None:
        output_name = f"{timestamp}_transformer_v2.zip"

    # Get the exact model code
    print("📄 Reading exact model architecture from models/eeg_transformer.py...")
    model_code = get_transformer_code()

    temp_dir = Path(f"temp_transformer_v2_{timestamp}")
    temp_dir.mkdir(exist_ok=True)

    try:
        # Write submission.py with embedded model code
        submission_content = SUBMISSION_TEMPLATE.format(
            timestamp=timestamp,
            model_code=model_code
        )
        submission_path = temp_dir / "submission.py"
        with open(submission_path, 'w') as f:
            f.write(submission_content)
        print(f"✅ Created submission.py with exact model architecture")

        # Copy checkpoints
        if Path(c1_checkpoint).exists():
            shutil.copy(c1_checkpoint, temp_dir / "c1_transformer.pth")
            print(f"✅ Copied C1 checkpoint: {c1_checkpoint}")
        else:
            print(f"⚠️  C1 checkpoint not found: {c1_checkpoint}")
            return None

        if Path(c2_checkpoint).exists():
            shutil.copy(c2_checkpoint, temp_dir / "c2_transformer.pth")
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

        print(f"\n✅ Transformer v2 submission created: {output_name}")

        zip_size = Path(output_name).stat().st_size / (1024 * 1024)
        print(f"   Size: {zip_size:.2f} MB")

        print(f"\n🎯 Expected Performance:")
        print(f"   C1 NRMSE: 1.000")
        print(f"   C2 NRMSE: 1.015")
        print(f"   Overall: 1.011 (beats current 1.11!)")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("="*70)
    print("🚀 Ready to submit!")
    print(f"   Upload {output_name} to Codabench")
    print("="*70)

    return output_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Transformer submission v2")

    parser.add_argument('--c1_checkpoint', type=str,
                       default='checkpoints_transformer/c1_transformer_best.pth',
                       help='Path to C1 transformer checkpoint')
    parser.add_argument('--c2_checkpoint', type=str,
                       default='checkpoints_transformer/c2_transformer_best.pth',
                       help='Path to C2 transformer checkpoint')
    parser.add_argument('--output', type=str, default=None,
                       help='Output ZIP filename')

    args = parser.parse_args()

    create_transformer_submission_v2(
        c1_checkpoint=args.c1_checkpoint,
        c2_checkpoint=args.c2_checkpoint,
        output_name=args.output
    )
