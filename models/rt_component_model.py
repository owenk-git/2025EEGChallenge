#!/usr/bin/env python3
"""
Multi-Component RT Model

Models RT as sum of components:
- Attention state (pre-stimulus alpha power)
- Decision time (P300 amplitude/latency)
- Motor execution (beta suppression)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionStateEncoder(nn.Module):
    """
    Encode pre-stimulus attention state from alpha power (8-12 Hz)
    High alpha = low attention = slower RT
    """
    def __init__(self, n_channels=129, pre_stim_points=50):
        super().__init__()

        # Focus on alpha band
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=11, padding=5),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.AvgPool1d(2),
            nn.Dropout(0.3),
        )

        self.pool = nn.AdaptiveAvgPool1d(10)

        # Predict attention contribution to RT
        self.attention_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 10, 64),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()  # [0, 1] contribution
        )

    def forward(self, x_pre):
        """
        x_pre: (batch, channels, 50) - pre-stimulus period
        Returns: (batch, 1) - attention state contribution
        """
        x = self.conv(x_pre)
        x = self.pool(x)
        attention_contrib = self.attention_head(x)
        return attention_contrib


class DecisionTimeEncoder(nn.Module):
    """
    Encode decision time from P300 component (300-500ms post-stimulus)
    Larger/earlier P300 = faster decision = faster RT
    """
    def __init__(self, n_channels=129, post_stim_points=150):
        super().__init__()

        # Focus on P300 window
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.AvgPool1d(2),
            nn.Dropout(0.3),

            nn.Conv1d(64, 32, kernel_size=11, padding=5),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.AvgPool1d(2),
            nn.Dropout(0.3),
        )

        self.pool = nn.AdaptiveAvgPool1d(15)

        # Predict decision time contribution
        self.decision_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 15, 64),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()  # [0, 1] contribution
        )

    def forward(self, x_post):
        """
        x_post: (batch, channels, 150) - post-stimulus period
        Returns: (batch, 1) - decision time contribution
        """
        x = self.conv(x_post)
        x = self.pool(x)
        decision_contrib = self.decision_head(x)
        return decision_contrib


class MotorExecutionEncoder(nn.Module):
    """
    Encode motor execution from beta suppression (13-30 Hz)
    Strong beta suppression = active motor preparation = faster execution
    """
    def __init__(self, n_channels=129, post_stim_points=150):
        super().__init__()

        # Focus on beta band and motor regions
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.AvgPool1d(2),
            nn.Dropout(0.3),

            nn.Conv1d(64, 32, kernel_size=11, padding=5),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.AvgPool1d(2),
            nn.Dropout(0.3),
        )

        self.pool = nn.AdaptiveAvgPool1d(15)

        # Predict motor execution contribution
        self.motor_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 15, 64),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()  # [0, 1] contribution
        )

    def forward(self, x_post):
        """
        x_post: (batch, channels, 150) - post-stimulus period
        Returns: (batch, 1) - motor execution contribution
        """
        x = self.conv(x_post)
        x = self.pool(x)
        motor_contrib = self.motor_head(x)
        return motor_contrib


class RTComponentModel(nn.Module):
    """
    Multi-Component RT Predictor

    RT = baseline + attention_weight * attention + decision_weight * decision + motor_weight * motor

    The model learns both the component contributions and their weights
    """
    def __init__(self, n_channels=129, trial_length=200, pre_stim_points=50):
        super().__init__()

        self.n_channels = n_channels
        self.trial_length = trial_length
        self.pre_stim_points = pre_stim_points
        self.post_stim_points = trial_length - pre_stim_points

        # Component encoders
        self.attention_encoder = AttentionStateEncoder(n_channels, pre_stim_points)
        self.decision_encoder = DecisionTimeEncoder(n_channels, self.post_stim_points)
        self.motor_encoder = MotorExecutionEncoder(n_channels, self.post_stim_points)

        # Learnable component weights
        self.component_weights = nn.Sequential(
            nn.Linear(3, 16),
            nn.ELU(),
            nn.Linear(16, 3),
            nn.Softmax(dim=-1)  # Weights sum to 1
        )

        # Baseline RT (learnable)
        self.baseline = nn.Parameter(torch.tensor([0.5]))

        # Final scaling (to match target range)
        self.output_scale = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x):
        """
        x: (batch, channels, time)
        Returns: (batch, 1) - RT prediction
        """
        # Split into pre and post stimulus
        x_pre = x[:, :, :self.pre_stim_points]
        x_post = x[:, :, self.pre_stim_points:]

        # Compute component contributions
        attention_contrib = self.attention_encoder(x_pre)  # (batch, 1)
        decision_contrib = self.decision_encoder(x_post)   # (batch, 1)
        motor_contrib = self.motor_encoder(x_post)         # (batch, 1)

        # Stack components
        components = torch.cat([attention_contrib, decision_contrib, motor_contrib], dim=1)  # (batch, 3)

        # Compute component weights (adaptive per sample)
        weights = self.component_weights(components)  # (batch, 3)

        # Weighted sum of components
        weighted_components = (components * weights).sum(dim=1, keepdim=True)  # (batch, 1)

        # Final RT = baseline + scaled weighted components
        rt_pred = self.baseline + self.output_scale * weighted_components

        return rt_pred

    def get_component_weights(self):
        """Get the learned component weights (for analysis)"""
        # Create dummy input to extract weights
        with torch.no_grad():
            dummy_components = torch.ones(1, 3)
            weights = self.component_weights(dummy_components)
        return {
            'attention': weights[0, 0].item(),
            'decision': weights[0, 1].item(),
            'motor': weights[0, 2].item()
        }


if __name__ == '__main__':
    # Test model
    print("Testing RT Component Model...\n")

    batch_size = 16
    n_channels = 129
    trial_length = 200

    # Create model
    model = RTComponentModel(
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

    # Get component weights
    weights = model.get_component_weights()
    print(f"\nComponent weights:")
    print(f"  Attention: {weights['attention']:.3f}")
    print(f"  Decision:  {weights['decision']:.3f}")
    print(f"  Motor:     {weights['motor']:.3f}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    print("\n✅ Model working!")
