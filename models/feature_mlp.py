"""
Feature-based MLP for EEG

Combines:
1. Hand-crafted EEG features (band power, spectral, time-domain)
2. PyTorch neural network for prediction

This avoids XGBoost/sklearn dependency while using domain knowledge.
"""

import torch
import torch.nn as nn
import numpy as np


def compute_psd_torch(signal, sfreq=100, nperseg=256):
    """
    Compute PSD using PyTorch operations (no scipy dependency)

    Args:
        signal: (n_samples,) tensor
        sfreq: Sampling frequency
        nperseg: Segment length

    Returns:
        freqs, psd
    """
    nperseg = min(nperseg, len(signal))

    # Simple periodogram using FFT
    window = torch.hann_window(nperseg, device=signal.device)
    windowed = signal[:nperseg] * window

    fft_result = torch.fft.rfft(windowed)
    psd = (fft_result.abs() ** 2) / nperseg
    freqs = torch.fft.rfftfreq(nperseg, 1/sfreq)

    return freqs, psd


def extract_band_power_torch(eeg_data, sfreq=100):
    """
    Extract band power features using PyTorch

    Args:
        eeg_data: (n_channels, n_times) tensor

    Returns:
        features: (n_features,) tensor
    """
    n_channels, n_times = eeg_data.shape
    device = eeg_data.device

    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 50)
    }

    features = []

    for band_name, (low, high) in bands.items():
        band_powers = []

        for ch in range(n_channels):
            freqs, psd = compute_psd_torch(eeg_data[ch], sfreq, nperseg=min(256, n_times))

            # Find frequency indices
            idx_band = (freqs >= low) & (freqs <= high)

            if idx_band.sum() > 0:
                # Trapezoidal integration
                band_power = torch.trapz(psd[idx_band], freqs[idx_band])
                band_powers.append(band_power)

        if len(band_powers) > 0:
            band_powers = torch.stack(band_powers)
            features.extend([
                band_powers.mean(),
                band_powers.std(),
                band_powers.max()
            ])
        else:
            features.extend([torch.tensor(0.0, device=device)] * 3)

    # Relative power
    total_power = sum(features[i*3] for i in range(len(bands)))
    if total_power > 0:
        for i in range(len(bands)):
            features.append(features[i*3] / total_power)
    else:
        features.extend([torch.tensor(0.0, device=device)] * len(bands))

    return torch.stack(features)


def extract_spectral_features_torch(eeg_data, sfreq=100):
    """Extract spectral features using PyTorch"""
    n_channels, n_times = eeg_data.shape
    device = eeg_data.device

    spectral_entropies = []
    peak_freqs = []
    spectral_centroids = []

    for ch in range(n_channels):
        freqs, psd = compute_psd_torch(eeg_data[ch], sfreq, nperseg=min(256, n_times))

        # Normalize PSD
        psd_sum = psd.sum()
        if psd_sum > 0:
            psd_norm = psd / psd_sum
        else:
            psd_norm = psd + 1e-10

        # Spectral entropy
        spectral_entropy = -(psd_norm * torch.log2(psd_norm + 1e-10)).sum()
        spectral_entropies.append(spectral_entropy)

        # Peak frequency
        peak_freq = freqs[psd.argmax()]
        peak_freqs.append(peak_freq)

        # Spectral centroid
        spectral_centroid = (freqs * psd_norm).sum()
        spectral_centroids.append(spectral_centroid)

    spectral_entropies = torch.stack(spectral_entropies)
    peak_freqs = torch.stack(peak_freqs)
    spectral_centroids = torch.stack(spectral_centroids)

    features = torch.stack([
        spectral_entropies.mean(),
        spectral_entropies.std(),
        peak_freqs.mean(),
        peak_freqs.std(),
        spectral_centroids.mean(),
        spectral_centroids.std()
    ])

    return features


def extract_time_features_torch(eeg_data):
    """Extract time-domain features using PyTorch"""
    n_channels, n_times = eeg_data.shape

    # Basic statistics
    features = [
        eeg_data.mean(dim=1).mean(),
        eeg_data.mean(dim=1).std(),
        eeg_data.std(dim=1).mean(),
        eeg_data.std(dim=1).std()
    ]

    # Hjorth parameters
    activities = []
    mobilities = []
    complexities = []

    for ch in range(n_channels):
        x = eeg_data[ch]
        dx = x[1:] - x[:-1]
        ddx = dx[1:] - dx[:-1]

        activity = x.var()
        activities.append(activity)

        var_x = x.var()
        var_dx = dx.var()
        mobility = torch.sqrt(var_dx / var_x) if var_x > 0 else torch.tensor(0.0, device=eeg_data.device)
        mobilities.append(mobility)

        var_ddx = ddx.var()
        complexity = (torch.sqrt(var_ddx / var_dx) / mobility) if (mobility > 0 and var_dx > 0) else torch.tensor(0.0, device=eeg_data.device)
        complexities.append(complexity)

    activities = torch.stack(activities)
    mobilities = torch.stack(mobilities)
    complexities = torch.stack(complexities)

    features.extend([
        activities.mean(),
        mobilities.mean(),
        complexities.mean()
    ])

    # Zero-crossing rate
    zero_crossings = []
    for ch in range(n_channels):
        signs = torch.sign(eeg_data[ch])
        zc = (signs[1:] != signs[:-1]).sum().float() / n_times
        zero_crossings.append(zc)

    zero_crossings = torch.stack(zero_crossings)
    features.extend([
        zero_crossings.mean(),
        zero_crossings.std()
    ])

    return torch.stack(features)


def extract_all_features_torch(eeg_data, sfreq=100):
    """
    Extract all features from EEG data

    Args:
        eeg_data: (n_channels, n_times) tensor

    Returns:
        features: (n_features,) tensor
    """
    band_features = extract_band_power_torch(eeg_data, sfreq)
    spectral_features = extract_spectral_features_torch(eeg_data, sfreq)
    time_features = extract_time_features_torch(eeg_data)

    all_features = torch.cat([band_features, spectral_features, time_features])
    return all_features


class FeatureMLP(nn.Module):
    """
    MLP that operates on hand-crafted EEG features

    This combines domain knowledge (feature engineering) with
    neural network learning, avoiding XGBoost dependency.
    """

    def __init__(self, n_channels=129, sfreq=100, challenge_name='c1', output_range=(0.5, 1.5)):
        super().__init__()

        self.n_channels = n_channels
        self.sfreq = sfreq
        self.challenge_name = challenge_name
        self.is_c1 = (challenge_name == 'c1')
        self.output_min, self.output_max = output_range

        # Compute feature dimension (this is fixed based on extract_all_features_torch)
        # Band power: 5 bands * 3 stats + 5 relative = 20
        # Spectral: 6 features
        # Time: 9 features
        # Total: 35 features
        self.n_features = 35

        # MLP for feature processing
        self.mlp = nn.Sequential(
            nn.Linear(self.n_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(32, 1)
        )

        # For C1, add sigmoid to constrain output
        if self.is_c1:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Identity()

    def forward(self, x):
        """
        Args:
            x: (batch, n_channels, n_times) EEG data

        Returns:
            (batch, 1) predictions
        """
        batch_size = x.shape[0]

        # Extract features for each sample in batch
        features_list = []
        for i in range(batch_size):
            features = extract_all_features_torch(x[i], self.sfreq)
            features_list.append(features)

        features = torch.stack(features_list)  # (batch, n_features)

        # Pass through MLP
        output = self.mlp(features)
        output = self.activation(output)

        # Scale output for C1
        if self.is_c1:
            output = self.output_min + output * (self.output_max - self.output_min)

        return output


def create_feature_mlp(challenge='c1', device='cuda', **kwargs):
    """
    Factory function to create FeatureMLP

    Args:
        challenge: 'c1' or 'c2'
        device: 'cuda' or 'cpu'

    Returns:
        model: FeatureMLP model
    """
    if challenge == 'c1':
        output_range = (0.5, 1.5)
    else:
        output_range = (-3, 3)

    model = FeatureMLP(challenge_name=challenge, output_range=output_range, **kwargs)
    model = model.to(device)
    return model


if __name__ == "__main__":
    print("Testing Feature MLP...")

    model = create_feature_mlp(challenge='c1', device='cpu')

    # Test forward pass
    x = torch.randn(4, 129, 200)
    y = model(x)

    print(f"✅ Input shape: {x.shape}")
    print(f"✅ Output shape: {y.shape}")
    print(f"✅ Output range: [{y.min():.3f}, {y.max():.3f}]")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Parameters: {n_params:,}")

    print("\n🎉 Feature MLP test passed!")
