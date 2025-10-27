#!/usr/bin/env python3
"""
STFT-based 2D CNN for RT Prediction

Uses Short-Time Fourier Transform to convert EEG from time domain
to time-frequency domain, then applies 2D CNN to extract features.

Time-frequency representation is crucial for EEG because:
- Alpha oscillations (8-12 Hz): Related to attention/alertness
- Beta oscillations (13-30 Hz): Related to motor preparation
- Theta oscillations (4-8 Hz): Related to cognitive processing
- Gamma (>30 Hz): High-frequency neural activity

STFT captures WHEN and WHICH frequencies are present, which is
more informative than raw time-series for oscillatory signals.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class STFTLayer(nn.Module):
    """
    STFT layer that converts time-series to time-frequency representation

    Input: (batch, channels, time)
    Output: (batch, channels, freq_bins, time_bins)
    """
    def __init__(self, n_fft=64, hop_length=16, win_length=64, freq_bins=33):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.freq_bins = freq_bins

        # Register window as buffer (not a parameter)
        window = torch.hann_window(win_length)
        self.register_buffer('window', window)

    def forward(self, x):
        """
        x: (batch, channels, time)
        Returns: (batch, channels, freq_bins, time_bins)
        """
        batch, channels, time = x.shape

        # Apply STFT to each channel
        stft_list = []
        for ch in range(channels):
            # STFT for this channel
            ch_data = x[:, ch, :]  # (batch, time)

            # torch.stft expects (batch, time) and returns (batch, freq, time, 2)
            # where last dim is [real, imag]
            stft_out = torch.stft(
                ch_data,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                return_complex=False,
                normalized=False,
                onesided=True
            )  # (batch, freq, time, 2)

            # Compute magnitude
            magnitude = torch.sqrt(stft_out[..., 0]**2 + stft_out[..., 1]**2)  # (batch, freq, time)

            stft_list.append(magnitude)

        # Stack all channels: (batch, channels, freq, time)
        stft_all = torch.stack(stft_list, dim=1)

        return stft_all


class STFT_2DCNN_RT(nn.Module):
    """
    STFT + 2D CNN for RT Prediction

    Pipeline:
    1. STFT: Convert (channels, time) → (channels, freq, time)
    2. 2D Conv: Process time-frequency images
    3. Global pooling + FC: Predict RT
    """
    def __init__(self, n_channels=129, trial_length=200, n_fft=64, hop_length=16):
        super().__init__()

        self.n_channels = n_channels
        self.trial_length = trial_length

        # STFT layer
        self.stft = STFTLayer(n_fft=n_fft, hop_length=hop_length, win_length=n_fft)

        # After STFT: (batch, 129 channels, 33 freq_bins, ~12 time_bins)
        # Treat as multi-channel 2D image

        # 2D CNN to process time-frequency representation
        # Conv across both frequency and time
        self.conv1 = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=(5, 3), padding=(2, 1)),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.3)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.3)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Dropout(0.3)
        )

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))

        # RT prediction head
        self.rt_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
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
        # STFT: (batch, channels, time) → (batch, channels, freq, time)
        x_stft = self.stft(x)

        # Apply log compression (common for spectrograms)
        x_stft = torch.log1p(x_stft)

        # 2D CNN
        x = self.conv1(x_stft)
        x = self.conv2(x)
        x = self.conv3(x)

        # Global pooling
        x = self.global_pool(x)

        # RT prediction
        rt_pred = self.rt_head(x)

        return rt_pred


class FrequencyBandCNN_RT(nn.Module):
    """
    Explicit Frequency Band CNN

    Instead of learning from full spectrogram, explicitly extract
    and process frequency bands separately:
    - Delta: 1-4 Hz
    - Theta: 4-8 Hz
    - Alpha: 8-12 Hz
    - Beta: 13-30 Hz
    - Gamma: 30-50 Hz
    """
    def __init__(self, n_channels=129, trial_length=200, sfreq=100):
        super().__init__()

        self.n_channels = n_channels
        self.trial_length = trial_length
        self.sfreq = sfreq

        # STFT layer
        self.stft = STFTLayer(n_fft=64, hop_length=16, win_length=64)

        # Frequency band indices (for 100 Hz sampling, n_fft=64)
        # freq_resolution = sfreq / n_fft = 100 / 64 = 1.56 Hz per bin
        freq_resolution = sfreq / 64

        # Band definitions (in bins)
        self.delta_bins = (int(1/freq_resolution), int(4/freq_resolution))    # 1-4 Hz
        self.theta_bins = (int(4/freq_resolution), int(8/freq_resolution))    # 4-8 Hz
        self.alpha_bins = (int(8/freq_resolution), int(12/freq_resolution))   # 8-12 Hz
        self.beta_bins = (int(13/freq_resolution), int(30/freq_resolution))   # 13-30 Hz
        self.gamma_bins = (int(30/freq_resolution), int(50/freq_resolution))  # 30-50 Hz

        # Separate CNN for each band
        self.delta_cnn = self._make_band_cnn(n_channels, 16)
        self.theta_cnn = self._make_band_cnn(n_channels, 16)
        self.alpha_cnn = self._make_band_cnn(n_channels, 32)  # Alpha is important for RT
        self.beta_cnn = self._make_band_cnn(n_channels, 32)   # Beta is important for motor
        self.gamma_cnn = self._make_band_cnn(n_channels, 16)

        # Fusion head
        total_features = 16 + 16 + 32 + 32 + 16  # = 112
        self.fusion = nn.Sequential(
            nn.Linear(total_features, 128),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def _make_band_cnn(self, n_channels, out_features):
        """Create CNN for a frequency band"""
        return nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(32 * 2 * 2, out_features),
            nn.ELU(),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        """
        x: (batch, channels, time)
        Returns: (batch, 1) - RT prediction
        """
        # STFT
        x_stft = self.stft(x)  # (batch, channels, freq, time)
        x_stft = torch.log1p(x_stft)

        # Extract frequency bands
        delta = x_stft[:, :, self.delta_bins[0]:self.delta_bins[1], :]
        theta = x_stft[:, :, self.theta_bins[0]:self.theta_bins[1], :]
        alpha = x_stft[:, :, self.alpha_bins[0]:self.alpha_bins[1], :]
        beta = x_stft[:, :, self.beta_bins[0]:self.beta_bins[1], :]
        gamma = x_stft[:, :, self.gamma_bins[0]:self.gamma_bins[1], :]

        # Process each band
        delta_feat = self.delta_cnn(delta)
        theta_feat = self.theta_cnn(theta)
        alpha_feat = self.alpha_cnn(alpha)
        beta_feat = self.beta_cnn(beta)
        gamma_feat = self.gamma_cnn(gamma)

        # Concatenate all band features
        all_features = torch.cat([delta_feat, theta_feat, alpha_feat, beta_feat, gamma_feat], dim=1)

        # Predict RT
        rt_pred = self.fusion(all_features)

        return rt_pred


if __name__ == '__main__':
    print("Testing STFT-based 2D CNN models...\n")

    batch_size = 8
    n_channels = 129
    trial_length = 200

    # Test STFT_2DCNN_RT
    print("=" * 60)
    print("STFT_2DCNN_RT")
    print("=" * 60)
    model1 = STFT_2DCNN_RT(n_channels, trial_length)
    x = torch.randn(batch_size, n_channels, trial_length)
    out = model1(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"RT range: [{out.min().item():.3f}, {out.max().item():.3f}]")
    print(f"Parameters: {sum(p.numel() for p in model1.parameters()):,}")

    # Test FrequencyBandCNN_RT
    print("\n" + "=" * 60)
    print("FrequencyBandCNN_RT")
    print("=" * 60)
    model2 = FrequencyBandCNN_RT(n_channels, trial_length, sfreq=100)
    out = model2(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"RT range: [{out.min().item():.3f}, {out.max().item():.3f}]")
    print(f"Parameters: {sum(p.numel() for p in model2.parameters()):,}")

    print(f"\nFrequency band ranges (bins):")
    print(f"  Delta (1-4 Hz): {model2.delta_bins}")
    print(f"  Theta (4-8 Hz): {model2.theta_bins}")
    print(f"  Alpha (8-12 Hz): {model2.alpha_bins}")
    print(f"  Beta (13-30 Hz): {model2.beta_bins}")
    print(f"  Gamma (30-50 Hz): {model2.gamma_bins}")

    print("\n✅ Both STFT models working!")
