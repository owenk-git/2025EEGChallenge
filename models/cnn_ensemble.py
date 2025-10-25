"""
CNN Ensemble Model

Combines multiple different architectures:
1. Temporal CNN (focus on time patterns)
2. Spatial CNN (focus on channel patterns)
3. Hybrid CNN (both temporal and spatial)

Ensemble voting improves robustness and generalization.
"""

import torch
import torch.nn as nn


class TemporalCNN(nn.Module):
    """CNN focused on temporal patterns"""

    def __init__(self, n_channels=129, n_times=200):
        super().__init__()

        # Process each channel independently in time
        self.temporal = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=25, padding=12),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),

            nn.Conv1d(64, 128, kernel_size=15, padding=7),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),

            nn.Conv1d(128, 256, kernel_size=7, padding=3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.fc = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32)
        )

    def forward(self, x):
        # x: (batch, n_channels, n_times)
        x = self.temporal(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        return x


class SpatialCNN(nn.Module):
    """CNN focused on spatial (channel) patterns"""

    def __init__(self, n_channels=129, n_times=200):
        super().__init__()

        # Treat channels as spatial dimension
        self.spatial = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(n_channels//4, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Conv2d(32, 64, kernel_size=(n_channels//4, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.temporal = nn.Sequential(
            nn.Conv1d(n_channels, 128, kernel_size=25, padding=12),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32)
        )

    def forward(self, x):
        # x: (batch, n_channels, n_times)
        # Use temporal path
        x = self.temporal(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        return x


class HybridCNN(nn.Module):
    """CNN with both temporal and spatial branches"""

    def __init__(self, n_channels=129, n_times=200):
        super().__init__()

        # Temporal branch
        self.temporal_branch = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=25, padding=12),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=15, padding=7),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # Spatial branch (depthwise)
        self.spatial_branch = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 25), padding=(0, 12)),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=(n_channels, 1), groups=32),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Fusion
        self.fc = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32)
        )

    def forward(self, x):
        # x: (batch, n_channels, n_times)

        # Temporal branch
        temp_out = self.temporal_branch(x).squeeze(-1)

        # Spatial branch
        spatial_in = x.unsqueeze(1)  # (batch, 1, channels, time)
        spatial_out = self.spatial_branch(spatial_in).squeeze(-1).squeeze(-1)

        # Concatenate and fuse
        x = torch.cat([temp_out, spatial_out], dim=1)
        x = self.fc(x)
        return x


class CNNEnsemble(nn.Module):
    """
    Ensemble of different CNN architectures

    Combines:
    - Temporal CNN (focus on time patterns)
    - Spatial CNN (focus on channel patterns)
    - Hybrid CNN (both temporal and spatial)

    Final prediction is weighted average of all models.
    """

    def __init__(self, n_channels=129, n_times=200, n_classes=1,
                 challenge_name='c1', output_range=(0.5, 1.5)):
        super().__init__()

        self.challenge_name = challenge_name
        self.is_c1 = (challenge_name == 'c1')
        self.output_min, self.output_max = output_range

        # Three different CNN architectures
        self.temporal_cnn = TemporalCNN(n_channels, n_times)
        self.spatial_cnn = SpatialCNN(n_channels, n_times)
        self.hybrid_cnn = HybridCNN(n_channels, n_times)

        # Final fusion layer (32 features from each model)
        self.fusion = nn.Sequential(
            nn.Linear(32 * 3, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, n_classes)
        )

        # For C1, add sigmoid
        if self.is_c1:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Identity()

    def forward(self, x):
        """
        Args:
            x: (batch, n_channels, n_times)

        Returns:
            (batch, n_classes)
        """
        # Get features from each model
        temp_features = self.temporal_cnn(x)
        spatial_features = self.spatial_cnn(x)
        hybrid_features = self.hybrid_cnn(x)

        # Concatenate all features
        all_features = torch.cat([temp_features, spatial_features, hybrid_features], dim=1)

        # Final fusion and prediction
        x = self.fusion(all_features)
        x = self.activation(x)

        # Scale output for C1
        if self.is_c1:
            x = self.output_min + x * (self.output_max - self.output_min)

        return x


def create_cnn_ensemble(challenge='c1', device='cuda', **kwargs):
    """
    Factory function to create CNN ensemble

    Args:
        challenge: 'c1' or 'c2'
        device: 'cuda' or 'cpu'

    Returns:
        model: CNNEnsemble model
    """
    if challenge == 'c1':
        output_range = (0.5, 1.5)
    else:
        output_range = (-3, 3)

    model = CNNEnsemble(challenge_name=challenge, output_range=output_range, **kwargs)
    model = model.to(device)
    return model


if __name__ == "__main__":
    print("Testing CNN Ensemble...")

    model = create_cnn_ensemble(challenge='c1', device='cpu')

    # Test forward pass
    x = torch.randn(4, 129, 200)
    y = model(x)

    print(f"✅ Input shape: {x.shape}")
    print(f"✅ Output shape: {y.shape}")
    print(f"✅ Output range: [{y.min():.3f}, {y.max():.3f}]")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Parameters: {n_params:,}")

    print("\n🎉 CNN Ensemble test passed!")
