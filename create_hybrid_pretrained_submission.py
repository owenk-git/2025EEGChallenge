"""
Create hybrid submission:
- C1: Pretrained ResNet18 (Best NRMSE: 0.965)
- C2: Official EEGNeX method (Best NRMSE: 1.01)

This combines the best of both approaches for optimal score.

Usage:
    python create_hybrid_pretrained_submission.py
"""

import argparse
import zipfile
import shutil
import torch
from pathlib import Path
from datetime import datetime


HYBRID_SUBMISSION_TEMPLATE = '''"""
EEG Challenge 2025 - Hybrid Submission
Generated: {timestamp}

Strategy:
- C1: Pretrained ResNet18 with transfer learning (NRMSE: 0.965)
- C2: Official EEGNeX method (NRMSE: 1.01)
- Expected Overall: 0.3 × 0.965 + 0.7 × 1.01 = 0.997
"""

import torch
import torch.nn as nn
from pathlib import Path

try:
    from torchvision import models
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False


def load_model_path():
    """Return directory containing this submission.py"""
    return Path(__file__).parent


# ============================================================================
# C1: Pretrained ResNet18 Model
# ============================================================================

class EEGPretrainedCNN_C1(nn.Module):
    """Pretrained ResNet18 for C1 reaction time prediction"""

    def __init__(self, backbone='resnet18', n_channels=129, output_range=(0.5, 1.5)):
        super().__init__()

        if not TORCHVISION_AVAILABLE:
            raise ImportError("torchvision required for C1 model")

        self.output_min, self.output_max = output_range

        # Load pretrained ResNet18
        base_model = models.resnet18(pretrained=False)  # Will load our weights
        feature_dim = 512

        # Adapt first conv layer for EEG
        original_conv = base_model.conv1
        self.conv1 = nn.Conv2d(
            1, 64,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )

        # Copy remaining ResNet layers
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.avgpool = base_model.avgpool

        # Regression head for C1
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # x: (batch, 129, 200)
        # Reshape to image-like: (batch, 1, 129, 200)
        x = x.unsqueeze(1)

        # ResNet forward
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)

        # Clamp output to valid range
        x = torch.clamp(x, self.output_min, self.output_max)

        return x


# ============================================================================
# C2: Official EEGNeX Model
# ============================================================================

class EEGNeX(nn.Module):
    """Official EEGNeX architecture for C2"""

    def __init__(self, n_channels=129, n_times=200, n_classes=1, challenge_name='c2'):
        super().__init__()

        self.challenge_name = challenge_name
        self.is_c1 = (challenge_name == 'c1')

        # Temporal convolution
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 25), padding=(0, 12)),
            nn.BatchNorm2d(32),
            nn.GELU()
        )

        # Spatial depthwise convolution
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(n_channels, 1), groups=32),
            nn.BatchNorm2d(32),
            nn.GELU()
        )

        # Separable convolution blocks
        self.sep_conv1 = self._make_sep_conv(32, 64)
        self.sep_conv2 = self._make_sep_conv(64, 128)

        # Adaptive pooling and classifier
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, n_classes)

    def _make_sep_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 15), padding=(0, 7), groups=in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.sep_conv1(x)
        x = self.sep_conv2(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


# ============================================================================
# Submission Class
# ============================================================================

class Submission:
    """Hybrid submission combining best models for C1 and C2"""

    def __init__(self, SFREQ, DEVICE):
        self.sfreq = SFREQ
        self.device = DEVICE
        self.model_path = load_model_path()

        print(f"🎯 Hybrid Submission:")
        print(f"   C1: Pretrained ResNet18 (transfer learning)")
        print(f"   C2: Official EEGNeX")
        print(f"   Device: {{DEVICE}}")

    def get_model_challenge_1(self):
        """Load C1: Pretrained ResNet18"""
        print("📦 Loading C1: Pretrained ResNet18...")

        model = EEGPretrainedCNN_C1(
            backbone='resnet18',
            n_channels=129,
            output_range=(0.5, 1.5)
        )

        checkpoint_path = self.model_path / "c1_pretrained_resnet18.pth"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"C1 checkpoint not found: {{checkpoint_path}}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])

        model = model.to(self.device)
        model.eval()

        print(f"✅ C1 model loaded (Best NRMSE: 0.965)")
        return model

    def get_model_challenge_2(self):
        """Load C2: Official EEGNeX"""
        print("📦 Loading C2: Official EEGNeX...")

        model = EEGNeX(
            n_channels=129,
            n_times=200,
            n_classes=1,
            challenge_name='c2'
        )

        checkpoint_path = self.model_path / "c2_official_eegnex.pth"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"C2 checkpoint not found: {{checkpoint_path}}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])

        model = model.to(self.device)
        model.eval()

        print(f"✅ C2 model loaded (Best NRMSE: 1.01)")
        return model
'''


def create_hybrid_submission(c1_checkpoint, c2_checkpoint, output_name=None):
    """
    Create hybrid submission ZIP

    Args:
        c1_checkpoint: Path to C1 pretrained model checkpoint
        c2_checkpoint: Path to C2 official model checkpoint
        output_name: Output ZIP filename
    """
    print("="*70)
    print("📦 Creating Hybrid Pretrained Submission")
    print("="*70)
    print("Strategy:")
    print("  C1: Pretrained ResNet18 (NRMSE: 0.965)")
    print("  C2: Official EEGNeX (NRMSE: 1.01)")
    print("  Expected Overall: 0.997")
    print("="*70)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    if output_name is None:
        output_name = f"{timestamp}_hybrid_pretrained_submission.zip"

    temp_dir = Path(f"temp_hybrid_{timestamp}")
    temp_dir.mkdir(exist_ok=True)

    try:
        # Write submission.py
        submission_content = HYBRID_SUBMISSION_TEMPLATE.format(timestamp=timestamp)
        submission_path = temp_dir / "submission.py"
        with open(submission_path, 'w') as f:
            f.write(submission_content)
        print(f"✅ Created submission.py")

        # Copy C1 checkpoint
        if Path(c1_checkpoint).exists():
            shutil.copy(c1_checkpoint, temp_dir / "c1_pretrained_resnet18.pth")
            print(f"✅ Copied C1 checkpoint: {c1_checkpoint}")
        else:
            print(f"⚠️  C1 checkpoint not found: {c1_checkpoint}")
            return None

        # Copy C2 checkpoint
        if Path(c2_checkpoint).exists():
            shutil.copy(c2_checkpoint, temp_dir / "c2_official_eegnex.pth")
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

        print(f"\n✅ Hybrid submission created: {output_name}")

        zip_size = Path(output_name).stat().st_size / (1024 * 1024)
        print(f"   Size: {zip_size:.2f} MB")

        print(f"\n🎯 Expected Performance:")
        print(f"   C1: 0.965 (pretrained)")
        print(f"   C2: 1.01 (official)")
        print(f"   Overall: 0.997 ⭐ (beats current 1.11!)")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("="*70)
    print("🚀 Ready to submit!")
    print(f"   Upload {output_name} to Codabench")
    print("="*70)

    return output_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create hybrid pretrained submission")

    parser.add_argument('--c1_checkpoint', type=str,
                       default='checkpoints_pretrained/c1_pretrained_best.pth',
                       help='Path to C1 pretrained checkpoint')
    parser.add_argument('--c2_checkpoint', type=str,
                       default='checkpoints/c2_official_best.pth',
                       help='Path to C2 official checkpoint')
    parser.add_argument('--output', type=str, default=None,
                       help='Output ZIP filename')

    args = parser.parse_args()

    create_hybrid_submission(
        c1_checkpoint=args.c1_checkpoint,
        c2_checkpoint=args.c2_checkpoint,
        output_name=args.output
    )
