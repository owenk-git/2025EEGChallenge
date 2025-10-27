#!/usr/bin/env python3
"""
Multi-Scale CNN for RT Prediction

Uses parallel Conv1D branches with different kernel sizes to capture
different temporal patterns in EEG:
- Short kernels (3, 5, 7): Sharp ERPs, high-frequency
- Medium kernels (11, 15): Alpha/beta oscillations
- Long kernels (25, 31): Slow waves, theta/delta
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleConv1DBlock(nn.Module):
    """
    Parallel convolutions with different kernel sizes
    """
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 7, 15, 25]):
        super().__init__()

        self.branches = nn.ModuleList()

        # Distribute channels evenly, give remaining to first branch
        channels_per_branch = out_channels // len(kernel_sizes)
        remaining_channels = out_channels % len(kernel_sizes)

        for i, kernel_size in enumerate(kernel_sizes):
            # First branch gets extra channels if division isn't even
            branch_channels = channels_per_branch + (remaining_channels if i == 0 else 0)

            branch = nn.Sequential(
                nn.Conv1d(in_channels, branch_channels,
                         kernel_size=kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(branch_channels),
                nn.ELU(),
                nn.Dropout(0.3)
            )
            self.branches.append(branch)

    def forward(self, x):
        # Apply all branches in parallel
        branch_outputs = [branch(x) for branch in self.branches]

        # Concatenate along channel dimension
        return torch.cat(branch_outputs, dim=1)


class SpatialConv(nn.Module):
    """Spatial convolution across channels"""
    def __init__(self, n_channels, out_channels=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ELU()
        )

    def forward(self, x):
        return self.conv(x)


class MultiScaleCNN_RT(nn.Module):
    """
    Multi-Scale CNN for RT Prediction

    Architecture:
    1. Spatial filtering (across channels)
    2. Multi-scale temporal convolutions (parallel branches)
    3. Hierarchical feature extraction
    4. RT prediction head
    """
    def __init__(self, n_channels=129, trial_length=200, pre_stim_points=50):
        super().__init__()

        self.n_channels = n_channels
        self.trial_length = trial_length
        self.pre_stim_points = pre_stim_points

        # Spatial filtering (reduce channels)
        self.spatial_conv = SpatialConv(n_channels, 64)

        # Multi-scale block 1: Small to large kernels
        self.multiscale1 = MultiScaleConv1DBlock(
            64, 128,
            kernel_sizes=[3, 7, 15, 25]
        )
        self.pool1 = nn.AvgPool1d(2)

        # Multi-scale block 2: Focus on alpha/beta
        self.multiscale2 = MultiScaleConv1DBlock(
            128, 128,
            kernel_sizes=[5, 11, 21]
        )
        self.pool2 = nn.AvgPool1d(2)

        # Multi-scale block 3: Longer context
        self.multiscale3 = MultiScaleConv1DBlock(
            128, 64,
            kernel_sizes=[7, 15, 31]
        )
        self.pool3 = nn.AvgPool1d(2)

        # Adaptive pooling to fixed size
        self.adaptive_pool = nn.AdaptiveAvgPool1d(20)

        # RT prediction head
        self.rt_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 20, 256),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: (batch, channels, time)
        Returns: (batch, 1) - RT prediction
        """
        # Spatial filtering
        x = self.spatial_conv(x)  # (batch, 64, time)

        # Multi-scale temporal processing
        x = self.multiscale1(x)  # (batch, 128, time)
        x = self.pool1(x)        # (batch, 128, time/2)

        x = self.multiscale2(x)  # (batch, 128, time/2)
        x = self.pool2(x)        # (batch, 128, time/4)

        x = self.multiscale3(x)  # (batch, 64, time/4)
        x = self.pool3(x)        # (batch, 64, time/8)

        # Adaptive pooling
        x = self.adaptive_pool(x)  # (batch, 64, 20)

        # RT prediction
        rt_pred = self.rt_head(x)  # (batch, 1)

        return rt_pred


class InceptionStyleCNN_RT(nn.Module):
    """
    Inception-style CNN with different kernel sizes per layer

    More aggressive multi-scale approach inspired by Inception networks
    """
    def __init__(self, n_channels=129, trial_length=200):
        super().__init__()

        self.n_channels = n_channels
        self.trial_length = trial_length

        # Spatial filtering
        self.spatial = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ELU()
        )

        # Inception block 1: Very diverse kernels
        self.inception1_1x1 = nn.Conv1d(64, 32, kernel_size=1)
        self.inception1_3x3 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.inception1_5x5 = nn.Conv1d(64, 32, kernel_size=5, padding=2)
        self.inception1_7x7 = nn.Conv1d(64, 32, kernel_size=7, padding=3)
        self.inception1_bn = nn.BatchNorm1d(128)
        self.inception1_act = nn.ELU()
        self.pool1 = nn.AvgPool1d(2)

        # Inception block 2: Medium kernels
        self.inception2_3x3 = nn.Conv1d(128, 32, kernel_size=3, padding=1)
        self.inception2_7x7 = nn.Conv1d(128, 32, kernel_size=7, padding=3)
        self.inception2_11x11 = nn.Conv1d(128, 32, kernel_size=11, padding=5)
        self.inception2_15x15 = nn.Conv1d(128, 32, kernel_size=15, padding=7)
        self.inception2_bn = nn.BatchNorm1d(128)
        self.inception2_act = nn.ELU()
        self.pool2 = nn.AvgPool1d(2)

        # Inception block 3: Large kernels for slow patterns
        self.inception3_7x7 = nn.Conv1d(128, 16, kernel_size=7, padding=3)
        self.inception3_15x15 = nn.Conv1d(128, 16, kernel_size=15, padding=7)
        self.inception3_25x25 = nn.Conv1d(128, 16, kernel_size=25, padding=12)
        self.inception3_31x31 = nn.Conv1d(128, 16, kernel_size=31, padding=15)
        self.inception3_bn = nn.BatchNorm1d(64)
        self.inception3_act = nn.ELU()
        self.pool3 = nn.AvgPool1d(2)

        # Adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool1d(25)

        # RT prediction
        self.rt_head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(64 * 25, 256),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Spatial
        x = self.spatial(x)

        # Inception block 1
        x1 = self.inception1_1x1(x)
        x2 = self.inception1_3x3(x)
        x3 = self.inception1_5x5(x)
        x4 = self.inception1_7x7(x)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.inception1_bn(x)
        x = self.inception1_act(x)
        x = self.pool1(x)

        # Inception block 2
        x1 = self.inception2_3x3(x)
        x2 = self.inception2_7x7(x)
        x3 = self.inception2_11x11(x)
        x4 = self.inception2_15x15(x)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.inception2_bn(x)
        x = self.inception2_act(x)
        x = self.pool2(x)

        # Inception block 3
        x1 = self.inception3_7x7(x)
        x2 = self.inception3_15x15(x)
        x3 = self.inception3_25x25(x)
        x4 = self.inception3_31x31(x)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.inception3_bn(x)
        x = self.inception3_act(x)
        x = self.pool3(x)

        # Adaptive pooling
        x = self.adaptive_pool(x)

        # RT prediction
        rt_pred = self.rt_head(x)

        return rt_pred


if __name__ == '__main__':
    print("Testing Multi-Scale CNN models...\n")

    batch_size = 16
    n_channels = 129
    trial_length = 200

    # Test MultiScaleCNN_RT
    print("=" * 60)
    print("MultiScaleCNN_RT")
    print("=" * 60)
    model1 = MultiScaleCNN_RT(n_channels, trial_length)
    x = torch.randn(batch_size, n_channels, trial_length)
    out = model1(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"RT range: [{out.min().item():.3f}, {out.max().item():.3f}]")
    print(f"Parameters: {sum(p.numel() for p in model1.parameters()):,}")

    # Test InceptionStyleCNN_RT
    print("\n" + "=" * 60)
    print("InceptionStyleCNN_RT")
    print("=" * 60)
    model2 = InceptionStyleCNN_RT(n_channels, trial_length)
    out = model2(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"RT range: [{out.min().item():.3f}, {out.max().item():.3f}]")
    print(f"Parameters: {sum(p.numel() for p in model2.parameters()):,}")

    print("\n✅ Both models working!")
