"""
Create submission ZIP for EEG Challenge 2025

Usage:
    python create_submission.py --model_c1 checkpoints/c1_best.pth --model_c2 checkpoints/c2_best.pth
"""

import argparse
import zipfile
import shutil
from pathlib import Path
from datetime import datetime


SUBMISSION_TEMPLATE = '''"""
EEG Challenge 2025 Submission
Generated: {timestamp}

Model: EEGNeX with Sigmoid-Inside-Classifier
Approach: Trained on HBN-EEG data
"""

import torch
import torch.nn as nn
from pathlib import Path


def load_model_path():
    """
    Helper function to load model path (required by competition)
    Returns the directory containing this submission.py file
    """
    return Path(__file__).parent


class EEGNeX(nn.Module):
    """EEGNeX model architecture"""

    def __init__(self, in_chans=129, n_classes=1, n_times=200,
                 challenge_name='c1', dropout=0.20,
                 output_range=(0.88, 1.12)):
        super().__init__()

        self.in_chans = in_chans
        self.n_classes = n_classes
        self.n_times = n_times
        self.challenge_name = challenge_name
        self.is_c1 = (challenge_name == 'c1')
        self.output_min, self.output_max = output_range

        # Temporal convolution
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(in_chans, 64, kernel_size=25, padding=12),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Spatial convolution
        self.spatial_conv = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=15, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Feature extraction
        self.feature_conv = nn.Sequential(
            nn.Conv1d(32, 16, kernel_size=10, padding=4),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout * 0.8)
        )

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)

        # Classifier with sigmoid INSIDE for C1
        if self.is_c1:
            self.classifier = nn.Sequential(
                nn.Linear(16, 8),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(8, 1),
                nn.Sigmoid()  # INSIDE architecture
            )
            self.use_c1_scaling = True
        else:
            self.classifier = nn.Linear(16, n_classes)
            self.use_c1_scaling = False

    def forward(self, x):
        """Forward pass"""
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.feature_conv(x)
        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.dropout(x)
        x = self.classifier(x)

        # Scale output for C1
        if self.use_c1_scaling:
            x = self.output_min + x * (self.output_max - self.output_min)

        return x


class Submission:
    """Submission class for inference"""

    def __init__(self, SFREQ, DEVICE):
        self.sfreq = SFREQ
        self.device = DEVICE
        self.model_path = load_model_path()

        print(f"🚀 EEG Challenge 2025 Submission")
        print(f"   Model: EEGNeX with Sigmoid-Inside-Classifier")
        print(f"   Model path: {{self.model_path}}")
        print(f"   Device: {{DEVICE}}")

    def get_model_challenge_1(self):
        """Load Challenge 1 model"""
        print("🧠 Loading Challenge 1 model...")

        n_times = int(2 * self.sfreq)

        model = EEGNeX(
            in_chans=129,
            n_classes=1,
            n_times=n_times,
            challenge_name='c1',
            dropout=0.20,
            output_range=(0.88, 1.12)
        ).to(self.device)

        # Load trained weights
        weights_path = self.model_path / "c1_weights.pth"
        if weights_path.exists():
            checkpoint = torch.load(weights_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded trained weights from epoch {{checkpoint['epoch']}}")
        else:
            print("⚠️  No trained weights found, using random initialization")

        model.eval()
        return model

    def get_model_challenge_2(self):
        """Load Challenge 2 model"""
        print("🧠 Loading Challenge 2 model...")

        n_times = int(2 * self.sfreq)

        model = EEGNeX(
            in_chans=129,
            n_classes=1,
            n_times=n_times,
            challenge_name='c2',
            dropout=0.20
        ).to(self.device)

        # Load trained weights
        weights_path = self.model_path / "c2_weights.pth"
        if weights_path.exists():
            checkpoint = torch.load(weights_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded trained weights from epoch {{checkpoint['epoch']}}")
        else:
            print("⚠️  No trained weights found, using random initialization")

        model.eval()
        return model
'''


def create_submission(model_c1_path, model_c2_path, output_name=None):
    """
    Create submission ZIP file

    Args:
        model_c1_path: Path to Challenge 1 model checkpoint
        model_c2_path: Path to Challenge 2 model checkpoint
        output_name: Output ZIP filename (auto-generated if None)
    """
    print("="*70)
    print("📦 Creating Submission Package")
    print("="*70)

    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    # Auto-generate output name if not provided
    if output_name is None:
        output_name = f"{timestamp}_trained_submission.zip"

    # Create temp directory
    temp_dir = Path(f"temp_submission_{timestamp}")
    temp_dir.mkdir(exist_ok=True)

    try:
        # Write submission.py
        submission_content = SUBMISSION_TEMPLATE.format(timestamp=timestamp)
        submission_path = temp_dir / "submission.py"
        with open(submission_path, 'w') as f:
            f.write(submission_content)
        print(f"✅ Created submission.py")

        # Copy model weights
        if Path(model_c1_path).exists():
            shutil.copy(model_c1_path, temp_dir / "c1_weights.pth")
            print(f"✅ Copied C1 weights: {model_c1_path}")
        else:
            print(f"⚠️  C1 weights not found: {model_c1_path}")

        if Path(model_c2_path).exists():
            shutil.copy(model_c2_path, temp_dir / "c2_weights.pth")
            print(f"✅ Copied C2 weights: {model_c2_path}")
        else:
            print(f"⚠️  C2 weights not found: {model_c2_path}")

        # Create ZIP
        with zipfile.ZipFile(output_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in temp_dir.rglob('*'):
                if file.is_file():
                    arcname = file.relative_to(temp_dir)
                    zipf.write(file, arcname)

        print(f"\n✅ Submission package created: {output_name}")

        # Show file sizes
        zip_size = Path(output_name).stat().st_size / 1024
        print(f"   Size: {zip_size:.1f} KB")

    finally:
        # Cleanup temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("="*70)
    print("🚀 Ready to submit!")
    print(f"   Upload {output_name} to Codabench")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create submission ZIP")

    parser.add_argument('--model_c1', type=str, required=True,
                        help='Path to Challenge 1 model checkpoint')
    parser.add_argument('--model_c2', type=str, required=True,
                        help='Path to Challenge 2 model checkpoint')
    parser.add_argument('--output', type=str, default=None,
                        help='Output ZIP filename (auto-generated if not provided)')

    args = parser.parse_args()

    create_submission(args.model_c1, args.model_c2, args.output)
