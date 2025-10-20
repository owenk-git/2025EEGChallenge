"""
Create Ensemble Submission for EEG Challenge

This script creates a submission.zip that contains multiple models
and ensembles their predictions.

Ensemble Methods:
1. Simple averaging
2. Weighted averaging (by validation performance)
3. Stacking (train meta-model)

Usage:
    # Simple averaging
    python create_ensemble_submission.py \
      --models checkpoints/c1_*.pth checkpoints/c2_*.pth \
      --method average

    # Weighted averaging
    python create_ensemble_submission.py \
      --models checkpoints/c1_*.pth checkpoints/c2_*.pth \
      --method weighted \
      --weights 0.25 0.20 0.30 0.15 0.10

    # From K-fold models
    python create_ensemble_submission.py \
      --models checkpoints_kfold/c1_fold*_best.pth \
      --method average
"""

import argparse
import torch
import zipfile
import shutil
from pathlib import Path
from datetime import datetime
import json


# Template for ensemble submission.py
ENSEMBLE_SUBMISSION_TEMPLATE = '''"""
Ensemble Submission for EEG Challenge {challenge}
Generated: {timestamp}

Ensemble method: {method}
Number of models: {n_models}
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path


class TemporalConvBlock(nn.Module):
    """Temporal convolution block"""
    def __init__(self, in_channels, out_channels, kernel_size, dropout=0.2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                             kernel_size=(1, kernel_size),
                             padding=(0, kernel_size // 2))
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class SpatialConvBlock(nn.Module):
    """Spatial (depthwise) convolution block"""
    def __init__(self, in_channels, depth_multiplier=2, dropout=0.2):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels * depth_multiplier,
                                   kernel_size=(in_channels, 1),
                                   groups=in_channels)
        self.bn = nn.BatchNorm2d(in_channels * depth_multiplier)
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class SeparableConv2d(nn.Module):
    """Separable convolution"""
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels,
                                   kernel_size=kernel_size,
                                   padding=(0, kernel_size[1] // 2),
                                   groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class EEGNeX(nn.Module):
    """EEGNeX architecture"""
    def __init__(self, in_chans=129, n_times=200, challenge_name='c1', dropout=0.2):
        super().__init__()

        # Temporal convolution
        self.temporal_conv = TemporalConvBlock(1, 8, kernel_size=64, dropout=dropout)

        # Spatial convolution
        self.spatial_conv = SpatialConvBlock(8, depth_multiplier=2, dropout=dropout)

        # Average pooling
        self.pool1 = nn.AvgPool2d((1, 4))

        # Separable convolutions
        self.separable_conv = nn.Sequential(
            SeparableConv2d(16, 16, kernel_size=(1, 16)),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Dropout(dropout)
        )

        self.pool2 = nn.AvgPool2d((1, 8))

        # Calculate flattened size
        n_features = 16 * (n_times // 32)

        # Classifier with sigmoid inside
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_features, 64),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Sigmoid inside classifier
        )

    def forward(self, x):
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.pool1(x)
        x = self.separable_conv(x)
        x = self.pool2(x)
        x = self.classifier(x)
        return x


def load_models(model_paths, challenge):
    """Load all models"""
    models = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for path in model_paths:
        model = EEGNeX(
            in_chans=129,
            n_times=200,
            challenge_name=challenge,
            dropout=0.2
        )

        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        models.append(model)

    return models


# Ensemble configuration
ENSEMBLE_METHOD = "{method}"
ENSEMBLE_WEIGHTS = {weights}
C1_MODEL_PATHS = {c1_paths}
C2_MODEL_PATHS = {c2_paths}


def predict_c1(eeg_data):
    """Challenge 1: Response time prediction with ensemble"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load models
    models = load_models(C1_MODEL_PATHS, 'c1')

    # Get predictions from all models
    predictions = []
    with torch.no_grad():
        eeg_tensor = torch.tensor(eeg_data, dtype=torch.float32).unsqueeze(0).to(device)

        for model in models:
            pred = model(eeg_tensor)
            predictions.append(pred.cpu().numpy())

    # Ensemble predictions
    predictions = np.array(predictions)  # Shape: (n_models, 1, 1)

    if ENSEMBLE_METHOD == "average":
        final_pred = np.mean(predictions, axis=0)
    elif ENSEMBLE_METHOD == "weighted":
        weights = np.array(ENSEMBLE_WEIGHTS).reshape(-1, 1, 1)
        final_pred = np.sum(predictions * weights, axis=0)
    else:
        # Default to average
        final_pred = np.mean(predictions, axis=0)

    return float(final_pred[0, 0])


def predict_c2(eeg_data):
    """Challenge 2: Externalizing factor prediction with ensemble"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load models
    models = load_models(C2_MODEL_PATHS, 'c2')

    # Get predictions from all models
    predictions = []
    with torch.no_grad():
        eeg_tensor = torch.tensor(eeg_data, dtype=torch.float32).unsqueeze(0).to(device)

        for model in models:
            pred = model(eeg_tensor)
            predictions.append(pred.cpu().numpy())

    # Ensemble predictions
    predictions = np.array(predictions)

    if ENSEMBLE_METHOD == "average":
        final_pred = np.mean(predictions, axis=0)
    elif ENSEMBLE_METHOD == "weighted":
        weights = np.array(ENSEMBLE_WEIGHTS).reshape(-1, 1, 1)
        final_pred = np.sum(predictions * weights, axis=0)
    else:
        final_pred = np.mean(predictions, axis=0)

    return float(final_pred[0, 0])
'''


def find_models_by_challenge(model_paths):
    """Separate model paths by challenge"""
    c1_models = []
    c2_models = []

    for path in model_paths:
        path_str = str(path)
        if 'c1' in path_str.lower():
            c1_models.append(str(path))
        elif 'c2' in path_str.lower():
            c2_models.append(str(path))

    return c1_models, c2_models


def create_ensemble_submission(model_paths, method='average', weights=None, output_name=None):
    """
    Create ensemble submission ZIP

    Args:
        model_paths: List of paths to model checkpoints
        method: 'average' or 'weighted'
        weights: List of weights for weighted averaging (must sum to 1.0)
        output_name: Custom output name
    """
    # Separate models by challenge
    c1_models, c2_models = find_models_by_challenge(model_paths)

    if not c1_models or not c2_models:
        print("⚠️  Warning: Missing models for one or both challenges")
        print(f"   C1 models: {len(c1_models)}")
        print(f"   C2 models: {len(c2_models)}")

    print(f"\n📦 Creating ensemble submission")
    print(f"   Method: {method}")
    print(f"   C1 models: {len(c1_models)}")
    print(f"   C2 models: {len(c2_models)}")

    # Validate weights
    if method == 'weighted':
        if weights is None:
            print("❌ Error: Weights required for weighted averaging")
            return

        if abs(sum(weights) - 1.0) > 0.01:
            print(f"⚠️  Warning: Weights sum to {sum(weights)}, normalizing to 1.0")
            weights = [w / sum(weights) for w in weights]

        print(f"   Weights: {weights}")

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    # Generate output name
    if output_name is None:
        output_name = f"{timestamp}_ensemble_{method}_{len(c1_models)}models.zip"

    # Create temporary directory
    temp_dir = Path(f"temp_submission_{timestamp}")
    temp_dir.mkdir(exist_ok=True)

    try:
        # Create submission.py
        submission_content = ENSEMBLE_SUBMISSION_TEMPLATE.format(
            challenge="1 and 2",
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            method=method,
            n_models=len(c1_models),
            weights=weights if weights else "None",
            c1_paths=c1_models,
            c2_paths=c2_models
        )

        submission_file = temp_dir / "submission.py"
        with open(submission_file, 'w') as f:
            f.write(submission_content)

        print(f"   ✅ Created submission.py")

        # Copy all model weights
        for i, model_path in enumerate(c1_models):
            dest = temp_dir / f"c1_model{i}.pth"
            shutil.copy(model_path, dest)
            print(f"   ✅ Copied C1 model {i+1}/{len(c1_models)}")

        for i, model_path in enumerate(c2_models):
            dest = temp_dir / f"c2_model{i}.pth"
            shutil.copy(model_path, dest)
            print(f"   ✅ Copied C2 model {i+1}/{len(c2_models)}")

        # Create ZIP file
        print(f"\n📦 Creating ZIP archive...")
        with zipfile.ZipFile(output_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in temp_dir.iterdir():
                zipf.write(file, file.name)

        print(f"\n✅ Ensemble submission created: {output_name}")

        # Calculate size
        size_mb = Path(output_name).stat().st_size / (1024 * 1024)
        print(f"   Size: {size_mb:.2f} MB")

        # List contents
        print(f"\n📋 Contents:")
        with zipfile.ZipFile(output_name, 'r') as zipf:
            for info in zipf.filelist:
                print(f"   - {info.filename} ({info.file_size / 1024:.1f} KB)")

    finally:
        # Cleanup temporary directory
        shutil.rmtree(temp_dir)

    print(f"\n🎯 Next step: Upload {output_name} to Codabench!")
    print(f"   https://www.codabench.org/competitions/9975/")

    return output_name


def main():
    parser = argparse.ArgumentParser(description="Create ensemble submission")

    parser.add_argument('--models', nargs='+', required=True,
                        help='Paths to model checkpoints (use wildcards: checkpoints/c1_*.pth)')
    parser.add_argument('--method', choices=['average', 'weighted'], default='average',
                        help='Ensemble method')
    parser.add_argument('--weights', nargs='+', type=float, default=None,
                        help='Weights for each model (for weighted averaging)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output filename')

    args = parser.parse_args()

    # Expand wildcards and find models
    model_paths = []
    for pattern in args.models:
        matching_files = list(Path('.').glob(pattern))
        model_paths.extend(matching_files)

    if not model_paths:
        print("❌ No models found matching the patterns")
        return

    print(f"\n📁 Found {len(model_paths)} models:")
    for path in model_paths:
        print(f"   - {path}")

    # Create ensemble submission
    create_ensemble_submission(
        model_paths=model_paths,
        method=args.method,
        weights=args.weights,
        output_name=args.output
    )


if __name__ == "__main__":
    main()
