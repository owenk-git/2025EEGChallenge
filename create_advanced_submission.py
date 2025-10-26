"""
Create submission ZIP with proper Submission class structure
Matches format from working erp_mlp_submission.zip
"""

import argparse
from pathlib import Path
import zipfile
import shutil


def create_submission_py_content():
    """Generate submission.py with proper Submission class and embedded model code"""

    return '''"""
Submission for NeurIPS 2025 EEG Challenge
Trial-Level RT Predictor (C1) + Domain Adaptation EEGNeX (C2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


# ============================================================================
# TRIAL-LEVEL RT PREDICTOR (C1 Model)
# ============================================================================

class SpatialAttention(nn.Module):
    """Learn which EEG channels are most predictive of RT"""
    def __init__(self, n_channels=129):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(n_channels, n_channels // 4),
            nn.ReLU(),
            nn.Linear(n_channels // 4, n_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        channel_stats = x.mean(dim=2)
        attention_weights = self.attention(channel_stats)
        attention_weights = attention_weights.unsqueeze(2)
        return x * attention_weights


class PreStimulusEncoder(nn.Module):
    """Encode pre-stimulus period (-500ms to 0ms)"""
    def __init__(self, n_channels=129, time_points=50):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=11, padding=5),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.AvgPool1d(2),
            nn.Dropout(0.3),
            nn.Conv1d(64, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.AvgPool1d(2),
            nn.Dropout(0.3)
        )
        self.pool = nn.AdaptiveAvgPool1d(10)

    def forward(self, x_pre):
        x = self.conv(x_pre)
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        return x


class PostStimulusEncoder(nn.Module):
    """Encode post-stimulus period (0ms to +1500ms)"""
    def __init__(self, n_channels=129, time_points=150):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 128, kernel_size=15, padding=7),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.AvgPool1d(2),
            nn.Dropout(0.3),
            nn.Conv1d(128, 128, kernel_size=11, padding=5),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.AvgPool1d(2),
            nn.Dropout(0.3),
            nn.Conv1d(128, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.AvgPool1d(2),
            nn.Dropout(0.3)
        )
        self.pool = nn.AdaptiveAvgPool1d(20)

    def forward(self, x_post):
        x = self.conv(x_post)
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        return x


class TrialLevelRTPredictor(nn.Module):
    """Trial-Level RT Predictor for C1"""
    def __init__(self, n_channels=129, trial_length=200, pre_stim_points=50):
        super().__init__()
        self.n_channels = n_channels
        self.trial_length = trial_length
        self.pre_stim_points = pre_stim_points
        self.post_stim_points = trial_length - pre_stim_points

        self.spatial_attention = SpatialAttention(n_channels)
        self.pre_encoder = PreStimulusEncoder(n_channels, self.pre_stim_points)
        self.post_encoder = PostStimulusEncoder(n_channels, self.post_stim_points)

        total_features = 320 + 1280
        self.rt_head = nn.Sequential(
            nn.Linear(total_features, 512),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Linear(128, 32),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.spatial_attention(x)
        x_pre = x[:, :, :self.pre_stim_points]
        x_post = x[:, :, self.pre_stim_points:]
        pre_features = self.pre_encoder(x_pre)
        post_features = self.post_encoder(x_post)
        combined = torch.cat([pre_features, post_features], dim=1)
        rt_pred = self.rt_head(combined).squeeze(-1)
        return rt_pred


# ============================================================================
# DOMAIN ADAPTATION EEGNEX (C2 Model)
# ============================================================================

class SeparableConv1d(nn.Module):
    """Depthwise Separable Convolution"""
    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size,
                                   groups=in_channels, padding=kernel_size//2, bias=bias)
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return x


class EEGNeXBlock(nn.Module):
    """EEGNeX Block with residual connection"""
    def __init__(self, channels, kernel_size=5):
        super().__init__()
        self.sep_conv = SeparableConv1d(channels, channels, kernel_size)
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        residual = x
        x = self.sep_conv(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x + residual


class DomainAdaptationEEGNeX(nn.Module):
    """EEGNeX with Domain Adaptation for C2"""
    def __init__(self, n_channels=129, n_times=900, challenge='c2', output_range=(-3, 3)):
        super().__init__()
        self.n_channels = n_channels
        self.n_times = n_times
        self.challenge = challenge
        self.output_range = output_range

        # Spatial filtering
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(1, 32, (n_channels, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU()
        )

        # Temporal convolution
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(32, 64, (1, 25), padding=(0, 12), bias=False),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.3)
        )

        # Separable conv blocks
        self.sep_conv1 = nn.Sequential(
            SeparableConv1d(64, 128, kernel_size=5),
            nn.ELU(),
            nn.AvgPool1d(2),
            nn.Dropout(0.3)
        )

        self.sep_conv2 = nn.Sequential(
            SeparableConv1d(128, 128, kernel_size=5),
            nn.ELU(),
            nn.AvgPool1d(2),
            nn.Dropout(0.3)
        )

        self.sep_conv3 = EEGNeXBlock(128, kernel_size=5)

        # Adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool1d(28)
        self.feature_dim = 128 * 28

        # Task predictor
        self.task_predictor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_dim, 256),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # Add channel dimension
        x = x.unsqueeze(1)

        # Spatial filtering
        x = self.spatial_conv(x)

        # Temporal filtering
        x = self.temporal_conv(x)

        # Remove spatial dimension
        x = x.squeeze(2)

        # Separable convolutions
        x = self.sep_conv1(x)
        x = self.sep_conv2(x)
        x = self.sep_conv3(x)

        # Adaptive pooling
        x = self.adaptive_pool(x)

        # Task prediction
        predictions = self.task_predictor(x)
        predictions = predictions.squeeze(-1)

        # Clip predictions
        predictions = torch.clamp(predictions, self.output_range[0], self.output_range[1])

        return predictions


# ============================================================================
# SUBMISSION CLASS
# ============================================================================

def load_model_path():
    """Get the directory containing model files"""
    return Path(__file__).parent


class Submission:
    """Submission class for NeurIPS 2025 EEG Challenge"""

    def __init__(self, SFREQ, DEVICE):
        self.sfreq = SFREQ
        self.device = DEVICE
        self.model_path = load_model_path()

        print(f"🤖 Trial-Level + Domain Adaptation Submission")
        print(f"   Device: {DEVICE}")
        print(f"   Sample rate: {SFREQ} Hz")

    def get_model_challenge_1(self):
        """Load C1 model (Trial-Level RT Predictor)"""
        print("📦 Loading C1 Trial-Level RT Predictor...")

        model = TrialLevelRTPredictor(
            n_channels=129,
            trial_length=200,
            pre_stim_points=50
        )

        checkpoint_path = self.model_path / "c1_model.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"C1 checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

        model = model.to(self.device)
        model.eval()

        print(f"✅ C1 Trial-Level RT Predictor loaded")
        return model

    def get_model_challenge_2(self):
        """Load C2 model (Domain Adaptation EEGNeX)"""
        print("📦 Loading C2 Domain Adaptation EEGNeX...")

        model = DomainAdaptationEEGNeX(
            n_channels=129,
            n_times=900,
            challenge='c2',
            output_range=(-3, 3)
        )

        checkpoint_path = self.model_path / "c2_model.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"C2 checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

        model = model.to(self.device)
        model.eval()

        print(f"✅ C2 Domain Adaptation EEGNeX loaded")
        return model
'''


def create_submission(checkpoint_c1, checkpoint_c2, name='submission', output_dir='submissions'):
    """
    Create submission ZIP file with proper format

    Args:
        checkpoint_c1: Path to C1 checkpoint
        checkpoint_c2: Path to C2 checkpoint
        name: Submission name (default: submission)
        output_dir: Output directory (default: submissions)
    """
    print(f"Creating submission: {name}")

    # Create temp directory
    temp_dir = Path('temp_submission')
    temp_dir.mkdir(exist_ok=True)

    try:
        # Create submission.py with embedded model code
        submission_code = create_submission_py_content()
        submission_path = temp_dir / 'submission.py'
        with open(submission_path, 'w') as f:
            f.write(submission_code)
        print(f"  ✓ Created submission.py with Submission class")

        # Copy checkpoints with correct names
        shutil.copy(checkpoint_c1, temp_dir / 'c1_model.pt')
        print(f"  ✓ Copied C1 checkpoint: {checkpoint_c1}")

        shutil.copy(checkpoint_c2, temp_dir / 'c2_model.pt')
        print(f"  ✓ Copied C2 checkpoint: {checkpoint_c2}")

        # Create zip file
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        zip_path = output_dir / f'{name}.zip'

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in temp_dir.iterdir():
                zipf.write(file, file.name)

        print(f"  ✓ Created {zip_path}")

        # Cleanup
        shutil.rmtree(temp_dir)

        print(f"\n✅ Submission created: {zip_path}")
        print(f"   Contains:")
        print(f"   - submission.py (with Submission class)")
        print(f"   - c1_model.pt (Trial-Level RT Predictor)")
        print(f"   - c2_model.pt (Domain Adaptation EEGNeX)")
        print(f"\n   Ready to submit to competition!")

        return zip_path

    except Exception as e:
        # Cleanup on error
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise e


def main():
    parser = argparse.ArgumentParser(
        description='Create Submission ZIP',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 create_advanced_submission.py --checkpoint_c1 c1_best.pt --checkpoint_c2 c2_best.pt
  python3 create_advanced_submission.py --checkpoint_c1 c1_best.pt --checkpoint_c2 c2_best.pt --name my_submission
        """
    )
    parser.add_argument('--checkpoint_c1', type=str, required=True,
                       help='Path to C1 checkpoint (.pt file)')
    parser.add_argument('--checkpoint_c2', type=str, required=True,
                       help='Path to C2 checkpoint (.pt file)')
    parser.add_argument('--name', type=str, default='submission',
                       help='Submission name (creates {name}.zip, default: submission)')
    parser.add_argument('--output_dir', type=str, default='submissions',
                       help='Output directory (default: submissions)')

    args = parser.parse_args()

    # Verify checkpoints exist
    c1_path = Path(args.checkpoint_c1)
    c2_path = Path(args.checkpoint_c2)

    if not c1_path.exists():
        print(f"❌ C1 checkpoint not found: {c1_path}")
        print(f"\nPlease provide the path to your C1 model checkpoint.")
        print(f"Expected: Trial-Level RT Predictor checkpoint from DEBUG_C1_TRAINING.py")
        return

    if not c2_path.exists():
        print(f"❌ C2 checkpoint not found: {c2_path}")
        print(f"\nPlease provide the path to your C2 model checkpoint.")
        print(f"Expected: Domain Adaptation EEGNeX checkpoint from DEBUG_C2_TRAINING.py")
        return

    create_submission(
        checkpoint_c1=args.checkpoint_c1,
        checkpoint_c2=args.checkpoint_c2,
        name=args.name,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
