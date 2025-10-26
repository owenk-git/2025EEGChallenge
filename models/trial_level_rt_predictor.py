#!/usr/bin/env python3
"""
TRIAL-LEVEL RT PREDICTOR

Simple but effective model that predicts RT from individual trials.

Key innovations:
1. Temporal split: Pre-stimulus (attention) vs Post-stimulus (ERP/motor)
2. Frequency awareness: Alpha (8-12Hz) and Beta (13-30Hz) extraction
3. Spatial attention: Learn which channels matter most
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    """
    Learn which EEG channels are most predictive of RT
    (e.g., frontal for attention, motor for execution)
    """
    def __init__(self, n_channels=129):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(n_channels, n_channels // 4),
            nn.ReLU(),
            nn.Linear(n_channels // 4, n_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: (batch, channels, time)
        Returns: (batch, channels, time) with channel-wise attention
        """
        # Global average over time
        channel_stats = x.mean(dim=2)  # (batch, channels)

        # Compute attention weights
        attention_weights = self.attention(channel_stats)  # (batch, channels)

        # Apply attention
        attention_weights = attention_weights.unsqueeze(2)  # (batch, channels, 1)
        return x * attention_weights


class PreStimulusEncoder(nn.Module):
    """
    Encode pre-stimulus period (−500ms to 0ms)

    Captures: Alpha power (attention/alertness state)
    """
    def __init__(self, n_channels=129, time_points=50):
        super().__init__()

        # Simple CNN for alpha extraction
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

        # Adaptive pooling to fixed size
        self.pool = nn.AdaptiveAvgPool1d(10)

    def forward(self, x_pre):
        """
        x_pre: (batch, channels, 50) - pre-stimulus period
        Returns: (batch, 320) - pre-stimulus features
        """
        x = self.conv(x_pre)  # (batch, 32, time)
        x = self.pool(x)  # (batch, 32, 10)
        x = x.reshape(x.size(0), -1)  # (batch, 320)
        return x


class PostStimulusEncoder(nn.Module):
    """
    Encode post-stimulus period (0ms to +1500ms)

    Captures: ERPs (P300, N200), motor preparation, beta suppression
    """
    def __init__(self, n_channels=129, time_points=150):
        super().__init__()

        # Deeper CNN for ERP extraction
        self.conv = nn.Sequential(
            # Early ERP components (N1, P1)
            nn.Conv1d(n_channels, 128, kernel_size=15, padding=7),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.AvgPool1d(2),
            nn.Dropout(0.3),

            # P300 window (~300-500ms)
            nn.Conv1d(128, 128, kernel_size=11, padding=5),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.AvgPool1d(2),
            nn.Dropout(0.3),

            # Motor execution
            nn.Conv1d(128, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.AvgPool1d(2),
            nn.Dropout(0.3)
        )

        # Adaptive pooling
        self.pool = nn.AdaptiveAvgPool1d(20)

    def forward(self, x_post):
        """
        x_post: (batch, channels, 150) - post-stimulus period
        Returns: (batch, 1280) - post-stimulus features
        """
        x = self.conv(x_post)  # (batch, 64, time)
        x = self.pool(x)  # (batch, 64, 20)
        x = x.reshape(x.size(0), -1)  # (batch, 1280)
        return x


class TrialLevelRTPredictor(nn.Module):
    """
    Trial-Level RT Predictor

    Input: Single trial (129 channels, 200 time points)
           - Pre-stimulus: [-500ms, 0ms] = 50 points
           - Post-stimulus: [0ms, +1500ms] = 150 points

    Output: RT prediction for THIS trial
    """
    def __init__(self, n_channels=129, trial_length=200, pre_stim_points=50):
        super().__init__()

        self.n_channels = n_channels
        self.trial_length = trial_length
        self.pre_stim_points = pre_stim_points
        self.post_stim_points = trial_length - pre_stim_points

        # Spatial attention (which channels matter)
        self.spatial_attention = SpatialAttention(n_channels)

        # Pre-stimulus encoder (attention state)
        self.pre_encoder = PreStimulusEncoder(n_channels, self.pre_stim_points)

        # Post-stimulus encoder (ERP, motor)
        self.post_encoder = PostStimulusEncoder(n_channels, self.post_stim_points)

        # Fusion and RT prediction
        total_features = 320 + 1280  # pre + post
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
            nn.Sigmoid()  # Output [0, 1] normalized RT
        )

    def forward(self, x):
        """
        x: (batch, channels, time)

        Returns: (batch,) - RT predictions
        """
        # Apply spatial attention
        x = self.spatial_attention(x)

        # Split into pre and post stimulus
        x_pre = x[:, :, :self.pre_stim_points]
        x_post = x[:, :, self.pre_stim_points:]

        # Encode separately
        pre_features = self.pre_encoder(x_pre)
        post_features = self.post_encoder(x_post)

        # Concatenate
        combined = torch.cat([pre_features, post_features], dim=1)

        # Predict RT
        rt_pred = self.rt_head(combined).squeeze(-1)

        return rt_pred


class RecordingLevelAggregator:
    """
    Aggregate trial-level predictions to recording-level

    In training: Predict RT for each trial
    In inference: Average predictions across all trials in recording
    """
    @staticmethod
    def aggregate(trial_predictions):
        """
        trial_predictions: (n_trials,) - RT predictions for all trials in recording

        Returns: scalar - recording-level RT prediction
        """
        # Use median (more robust than mean)
        return torch.median(trial_predictions)


if __name__ == '__main__':
    # Test model
    print("Testing Trial-Level RT Predictor...\n")

    batch_size = 16
    n_channels = 129
    trial_length = 200

    # Create model
    model = TrialLevelRTPredictor(
        n_channels=n_channels,
        trial_length=trial_length,
        pre_stim_points=50
    )

    # Test forward pass
    x = torch.randn(batch_size, n_channels, trial_length)
    rt_pred = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {rt_pred.shape}")
    print(f"RT range: [{rt_pred.min().item():.3f}, {rt_pred.max().item():.3f}]")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    print("\n✅ Model working!")
