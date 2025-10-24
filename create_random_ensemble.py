"""
Create ensemble of random initializations

Since random weights got C1: 0.93, try ensemble of multiple random models
Averaging might reduce variance and improve score!

Usage:
    python create_random_ensemble.py --n_models 10
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path
import zipfile
import shutil
from datetime import datetime

SUBMISSION_TEMPLATE_ENSEMBLE = '''"""
Ensemble of {n_models} random initializations

Random init got C1: 0.93 before, ensemble might be even better!
"""

import torch
import torch.nn as nn
from pathlib import Path
import numpy as np

class EEGNeXRandom(nn.Module):
    def __init__(self, in_chans=129, n_classes=1, n_times=200, dropout=0.20,
                 challenge_name='c1', seed=42):
        super().__init__()

        # Set seed for reproducible random init
        torch.manual_seed(seed)

        self.challenge_name = challenge_name
        self.is_c1 = (challenge_name == 'c1')

        self.temporal_conv = nn.Sequential(
            nn.Conv1d(in_chans, 64, kernel_size=25, padding=12),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.spatial_conv = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=15, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.feature_conv = nn.Sequential(
            nn.Conv1d(32, 16, kernel_size=10, padding=4),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout * 0.8)
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)

        if self.is_c1:
            self.classifier = nn.Sequential(
                nn.Linear(16, 8),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(8, 1),
                nn.Sigmoid()
            )
            self.use_c1_scaling = True
        else:
            self.classifier = nn.Linear(16, n_classes)
            self.use_c1_scaling = False

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                if self.is_c1:
                    nn.init.xavier_normal_(m.weight, gain=0.5)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.5)
                else:
                    nn.init.xavier_uniform_(m.weight, gain=0.1)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.feature_conv(x)
        x = self.global_pool(x).squeeze(-1)
        x = self.dropout(x)
        x = self.classifier(x)

        if self.use_c1_scaling:
            x = 0.5 + x * 1.0

        return x


class Submission:
    def __init__(self, SFREQ, DEVICE):
        self.sfreq = SFREQ
        self.device = DEVICE
        self.n_models = {n_models}
        self.seeds = {seeds}

    def get_model_challenge_1(self):
        """Ensemble of random C1 models"""
        models = []
        for seed in self.seeds:
            model = EEGNeXRandom(
                in_chans=129,
                n_classes=1,
                n_times=int(2 * self.sfreq),
                dropout=0.20,
                challenge_name='c1',
                seed=seed
            ).to(self.device)
            model.eval()
            models.append(model)

        # Return ensemble wrapper
        return EnsembleModel(models)

    def get_model_challenge_2(self):
        """Ensemble of random C2 models"""
        models = []
        for seed in self.seeds:
            model = EEGNeXRandom(
                in_chans=129,
                n_classes=1,
                n_times=int(2 * self.sfreq),
                dropout=0.20,
                challenge_name='c2',
                seed=seed
            ).to(self.device)
            model.eval()
            models.append(model)

        return EnsembleModel(models)


class EnsembleModel(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        # Average predictions from all models
        outputs = [model(x) for model in self.models]
        return torch.stack(outputs).mean(dim=0)

    def eval(self):
        for model in self.models:
            model.eval()
        return self
'''


def create_random_ensemble(n_models=10, output_name=None):
    """Create submission with ensemble of random models"""

    print("="*70)
    print(f"Creating Random Ensemble Submission ({n_models} models)")
    print("="*70)
    print("Strategy: Random init got C1: 0.93")
    print("Ensemble might reduce variance and improve!")

    # Generate seeds
    seeds = list(range(42, 42 + n_models))

    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    # Auto-generate output name
    if output_name is None:
        output_name = f"{timestamp}_random_ensemble_{n_models}.zip"

    # Create temp directory
    temp_dir = Path(f"temp_ensemble_{timestamp}")
    temp_dir.mkdir(exist_ok=True)

    try:
        # Write submission.py
        submission_content = SUBMISSION_TEMPLATE_ENSEMBLE.format(
            n_models=n_models,
            seeds=seeds
        )

        submission_path = temp_dir / "submission.py"
        with open(submission_path, 'w') as f:
            f.write(submission_content)

        print(f"✅ Created submission.py with {n_models} models")
        print(f"   Seeds: {seeds[:5]}...{seeds[-1]}")

        # Create ZIP
        with zipfile.ZipFile(output_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(submission_path, submission_path.name)

        print(f"\n✅ Ensemble submission created: {output_name}")

        # Show file size
        zip_size = Path(output_name).stat().st_size / 1024
        print(f"   Size: {zip_size:.1f} KB")

    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("="*70)
    print("🚀 Ready to submit!")
    print(f"   Upload {output_name} to Codabench")
    print(f"   Expected: Better than single random (C1: 0.93)")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create random ensemble submission")
    parser.add_argument('--n_models', type=int, default=10,
                        help='Number of random models to ensemble (default: 10)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output ZIP filename')

    args = parser.parse_args()
    create_random_ensemble(n_models=args.n_models, output_name=args.output)
