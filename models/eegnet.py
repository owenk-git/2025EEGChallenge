"""
EEGNeX Model Architecture
Based on successful Sub 3 submission (1.14 overall score)

Key Innovation: Sigmoid activation INSIDE classifier architecture
"""

import torch
import torch.nn as nn


class EEGNeX(nn.Module):
    """
    EEGNeX architecture with sigmoid-inside-classifier for Challenge 1

    Architecture:
    - Temporal conv: 129 → 64 channels
    - Spatial conv: 64 → 32 channels
    - Feature conv: 32 → 16 channels
    - Classifier: 16 → 8 → 1

    Parameters:
    -----------
    in_chans : int
        Number of EEG channels (default: 129)
    n_classes : int
        Number of output classes (default: 1)
    n_times : int
        Number of time points (default: 200 for 100 Hz, 2 sec)
    challenge_name : str
        'c1' or 'c2' - determines classifier architecture
    dropout : float
        Dropout rate (default: 0.20)
    output_range : tuple
        (min, max) for C1 output scaling (default: (0.88, 1.12))
    """

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

        # Classifier - KEY: Sigmoid INSIDE for C1
        if self.is_c1:
            self.classifier = nn.Sequential(
                nn.Linear(16, 8),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(8, 1),
                nn.Sigmoid()  # ← INSIDE architecture (proven approach)
            )
            self.use_c1_scaling = True
        else:
            # Challenge 2: simple classifier
            self.classifier = nn.Linear(16, n_classes)
            self.use_c1_scaling = False

        self._init_weights()

    def _init_weights(self):
        """Initialize network weights"""
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
                        nn.init.constant_(m.bias, 0)
                else:
                    nn.init.normal_(m.weight, 0, 0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: (batch_size, in_chans, n_times)

        Returns:
            predictions: (batch_size, n_classes)
        """
        # Convolutional layers
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.feature_conv(x)

        # Global pooling
        x = self.global_pool(x)
        x = x.squeeze(-1)

        # Dropout before classifier
        x = self.dropout(x)

        # Classifier (sigmoid already inside for C1)
        x = self.classifier(x)

        # Scale output for C1 (from [0, 1] to [output_min, output_max])
        if self.use_c1_scaling:
            x = self.output_min + x * (self.output_max - self.output_min)

        return x


def create_model(challenge='c1', device='cuda', **kwargs):
    """
    Factory function to create model

    Args:
        challenge: 'c1' or 'c2'
        device: 'cuda' or 'cpu'
        **kwargs: additional model arguments

    Returns:
        model: EEGNeX model
    """
    model = EEGNeX(challenge_name=challenge, **kwargs)
    model = model.to(device)
    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing EEGNeX model...")

    # Challenge 1
    model_c1 = create_model(challenge='c1', device='cpu')
    x = torch.randn(4, 129, 200)  # batch=4, channels=129, time=200
    y = model_c1(x)
    print(f"C1 output shape: {y.shape}")
    print(f"C1 output range: [{y.min():.3f}, {y.max():.3f}]")

    # Challenge 2
    model_c2 = create_model(challenge='c2', device='cpu')
    y2 = model_c2(x)
    print(f"C2 output shape: {y2.shape}")

    print("✅ Model test passed!")
