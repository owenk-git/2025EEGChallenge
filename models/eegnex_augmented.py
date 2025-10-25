"""
Improved EEGNeX with Data Augmentation

Enhancements:
1. Time warping augmentation
2. Channel dropout augmentation
3. Noise injection
4. Improved architecture with residual connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class EEGAugmentation(nn.Module):
    """
    EEG data augmentation module

    Applies various augmentations during training to improve generalization
    """

    def __init__(self, p_time_warp=0.3, p_channel_drop=0.2, p_noise=0.3):
        super().__init__()
        self.p_time_warp = p_time_warp
        self.p_channel_drop = p_channel_drop
        self.p_noise = p_noise

    def time_warp(self, x):
        """Randomly speed up or slow down the signal"""
        batch, channels, time = x.shape

        # Random warp factor (0.8 to 1.2)
        warp_factor = 0.8 + random.random() * 0.4

        # Interpolate
        new_time = int(time * warp_factor)
        x_warped = F.interpolate(x, size=new_time, mode='linear', align_corners=False)

        # Crop or pad to original length
        if new_time > time:
            start = random.randint(0, new_time - time)
            x_warped = x_warped[:, :, start:start+time]
        elif new_time < time:
            pad_size = time - new_time
            pad_left = random.randint(0, pad_size)
            pad_right = pad_size - pad_left
            x_warped = F.pad(x_warped, (pad_left, pad_right), mode='replicate')

        return x_warped

    def channel_dropout(self, x):
        """Randomly drop some channels"""
        batch, channels, time = x.shape

        # Drop 10-30% of channels
        n_drop = int(channels * (0.1 + random.random() * 0.2))
        drop_indices = random.sample(range(channels), n_drop)

        x = x.clone()
        x[:, drop_indices, :] = 0

        return x

    def add_noise(self, x):
        """Add Gaussian noise"""
        noise_level = 0.01 + random.random() * 0.04  # 1-5% noise
        noise = torch.randn_like(x) * noise_level * x.std()
        return x + noise

    def forward(self, x, training=True):
        """
        Apply augmentations during training

        Args:
            x: (batch, channels, time)
            training: Whether in training mode

        Returns:
            Augmented x
        """
        if not training:
            return x

        # Apply each augmentation with probability
        if random.random() < self.p_time_warp:
            x = self.time_warp(x)

        if random.random() < self.p_channel_drop:
            x = self.channel_dropout(x)

        if random.random() < self.p_noise:
            x = self.add_noise(x)

        return x


class EEGNeXImproved(nn.Module):
    """
    Improved EEGNeX with:
    - Data augmentation
    - Residual connections
    - Better regularization
    """

    def __init__(self, n_channels=129, n_times=200, n_classes=1, challenge_name='c1',
                 output_range=(0.5, 1.5), use_augmentation=True):
        super().__init__()

        self.challenge_name = challenge_name
        self.is_c1 = (challenge_name == 'c1')
        self.output_min, self.output_max = output_range
        self.use_augmentation = use_augmentation

        # Data augmentation
        if use_augmentation:
            self.augmentation = EEGAugmentation()

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

        # Separable convolution blocks with residual connections
        self.sep_conv1 = self._make_sep_conv_residual(32, 64)
        self.sep_conv2 = self._make_sep_conv_residual(64, 128)
        self.sep_conv3 = self._make_sep_conv_residual(128, 128)  # Extra block

        # Adaptive pooling and classifier
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Enhanced classifier
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes)
        )

        # For C1, add sigmoid
        if self.is_c1:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Identity()

    def _make_sep_conv_residual(self, in_channels, out_channels):
        """Separable convolution with residual connection"""
        return nn.ModuleDict({
            'conv': nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=(1, 15), padding=(0, 7), groups=in_channels),
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
                nn.Dropout(0.3)
            ),
            'residual': nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        })

    def _apply_sep_conv_residual(self, x, block):
        """Apply separable conv with residual"""
        residual = block['residual'](x)
        x = block['conv'](x)
        return x + residual

    def forward(self, x):
        # Apply augmentation
        if self.use_augmentation and self.training:
            x = self.augmentation(x, training=True)

        # Add channel dimension
        x = x.unsqueeze(1)  # (batch, 1, channels, time)

        # Temporal and spatial processing
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)

        # Separable convolutions with residuals
        x = self._apply_sep_conv_residual(x, self.sep_conv1)
        x = self._apply_sep_conv_residual(x, self.sep_conv2)
        x = self._apply_sep_conv_residual(x, self.sep_conv3)

        # Pooling and classification
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.activation(x)

        # Scale output for C1
        if self.is_c1:
            x = self.output_min + x * (self.output_max - self.output_min)

        return x


def create_eegnex_improved(challenge='c1', device='cuda', **kwargs):
    """
    Factory function to create improved EEGNeX

    Args:
        challenge: 'c1' or 'c2'
        device: 'cuda' or 'cpu'

    Returns:
        model: EEGNeXImproved model
    """
    if challenge == 'c1':
        output_range = (0.5, 1.5)
    else:
        output_range = (-3, 3)

    model = EEGNeXImproved(challenge_name=challenge, output_range=output_range, **kwargs)
    model = model.to(device)
    return model


if __name__ == "__main__":
    print("Testing Improved EEGNeX...")

    model = create_eegnex_improved(challenge='c1', device='cpu')

    # Test forward pass
    x = torch.randn(4, 129, 200)
    y = model(x)

    print(f"✅ Input shape: {x.shape}")
    print(f"✅ Output shape: {y.shape}")
    print(f"✅ Output range: [{y.min():.3f}, {y.max():.3f}]")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Parameters: {n_params:,}")

    print("\n🎉 Improved EEGNeX test passed!")
