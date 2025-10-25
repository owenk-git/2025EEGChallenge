"""
ERP-Based Feature Extraction for Reaction Time Prediction

Uses cognitive neuroscience insights:
- P300 latency correlates strongly with RT (r=0.6-0.8)
- N2 latency relates to response inhibition
- Pre-stimulus alpha power predicts attention state
- Motor preparation signals (beta desynchronization)

This is THE standard approach in neuroscience for RT prediction.
"""

import torch
import torch.nn as nn
import numpy as np


def extract_p300_features(eeg_data, sfreq=100):
    """
    Extract P300 Event-Related Potential features

    P300 is a positive deflection ~300-600ms after stimulus
    Earlier P300 = faster reaction time

    Args:
        eeg_data: (n_channels, n_times) tensor
        sfreq: Sampling frequency (100 Hz)

    Returns:
        features: P300 latency, amplitude, area
    """
    device = eeg_data.device
    n_channels, n_times = eeg_data.shape

    # P300 window: 300-600ms (assuming stimulus at t=0)
    # At 100Hz: 30-60 samples
    # But our data is 2 seconds, so we need to find stimulus onset
    # Assume stimulus is in first 500ms (0-50 samples)

    # For each channel, find peak in 300-600ms window
    # Approximate: look at second half of epoch
    p300_window_start = n_times // 3  # ~0.66s
    p300_window_end = 2 * n_times // 3  # ~1.33s

    p300_latencies = []
    p300_amplitudes = []
    p300_areas = []

    for ch in range(n_channels):
        signal = eeg_data[ch]

        # P300 is positive deflection - find max in window
        window_signal = signal[p300_window_start:p300_window_end]

        if len(window_signal) > 0:
            # Latency (time of peak)
            peak_idx = window_signal.argmax()
            latency = (p300_window_start + peak_idx) / sfreq  # Convert to seconds
            p300_latencies.append(latency)

            # Amplitude (height of peak)
            amplitude = window_signal[peak_idx]
            p300_amplitudes.append(amplitude)

            # Area under curve (integral)
            area = window_signal.sum() / sfreq
            p300_areas.append(area)

    if len(p300_latencies) > 0:
        p300_latencies = torch.stack(p300_latencies)
        p300_amplitudes = torch.stack(p300_amplitudes)
        p300_areas = torch.stack(p300_areas)

        features = torch.stack([
            p300_latencies.mean(),      # Average P300 latency
            p300_latencies.std(),       # Variability
            p300_amplitudes.mean(),     # Average amplitude
            p300_amplitudes.max(),      # Max amplitude
            p300_areas.mean(),          # Average area
        ]).to(device)
    else:
        features = torch.zeros(5, device=device)

    return features


def extract_n2_features(eeg_data, sfreq=100):
    """
    Extract N2 Event-Related Potential features

    N2 is a negative deflection ~200-350ms after stimulus
    Related to response inhibition and conflict detection
    """
    device = eeg_data.device
    n_channels, n_times = eeg_data.shape

    # N2 window: 200-350ms
    n2_window_start = n_times // 6  # ~0.33s
    n2_window_end = n_times // 3    # ~0.66s

    n2_latencies = []
    n2_amplitudes = []

    for ch in range(n_channels):
        signal = eeg_data[ch]
        window_signal = signal[n2_window_start:n2_window_end]

        if len(window_signal) > 0:
            # N2 is negative deflection - find min
            peak_idx = window_signal.argmin()
            latency = (n2_window_start + peak_idx) / sfreq
            n2_latencies.append(latency)

            amplitude = window_signal[peak_idx].abs()
            n2_amplitudes.append(amplitude)

    if len(n2_latencies) > 0:
        n2_latencies = torch.stack(n2_latencies)
        n2_amplitudes = torch.stack(n2_amplitudes)

        features = torch.stack([
            n2_latencies.mean(),
            n2_amplitudes.mean(),
            n2_amplitudes.max()
        ]).to(device)
    else:
        features = torch.zeros(3, device=device)

    return features


def extract_prestimulus_alpha(eeg_data, sfreq=100):
    """
    Extract pre-stimulus alpha power

    Alpha power (8-13 Hz) before stimulus predicts:
    - Attention state
    - Arousal level
    - Response readiness

    Higher alpha = slower RT (drowsy, inattentive)
    """
    device = eeg_data.device
    n_channels, n_times = eeg_data.shape

    # Pre-stimulus period: first 500ms
    prestim_end = min(50, n_times // 4)  # First 500ms or 1/4 of signal

    alpha_powers = []

    for ch in range(n_channels):
        signal = eeg_data[ch, :prestim_end]

        if len(signal) > 16:
            # FFT to get alpha power
            fft_result = torch.fft.rfft(signal)
            psd = (fft_result.abs() ** 2) / len(signal)
            freqs = torch.fft.rfftfreq(len(signal), 1/sfreq).to(device)

            # Alpha band: 8-13 Hz
            alpha_idx = (freqs >= 8) & (freqs <= 13)
            if alpha_idx.sum() > 0:
                alpha_power = psd[alpha_idx].mean()
                alpha_powers.append(alpha_power)

    if len(alpha_powers) > 0:
        alpha_powers = torch.stack(alpha_powers)
        features = torch.stack([
            alpha_powers.mean(),
            alpha_powers.std(),
            alpha_powers.max()
        ]).to(device)
    else:
        features = torch.zeros(3, device=device)

    return features


def extract_motor_preparation(eeg_data, sfreq=100):
    """
    Extract motor preparation features

    Beta desynchronization (13-30 Hz) indicates motor preparation
    Stronger desynchronization = faster response
    """
    device = eeg_data.device
    n_channels, n_times = eeg_data.shape

    # Motor preparation period: middle portion before response
    prep_start = n_times // 3
    prep_end = 2 * n_times // 3

    beta_powers = []

    for ch in range(n_channels):
        signal = eeg_data[ch, prep_start:prep_end]

        if len(signal) > 16:
            fft_result = torch.fft.rfft(signal)
            psd = (fft_result.abs() ** 2) / len(signal)
            freqs = torch.fft.rfftfreq(len(signal), 1/sfreq).to(device)

            # Beta band: 13-30 Hz
            beta_idx = (freqs >= 13) & (freqs <= 30)
            if beta_idx.sum() > 0:
                beta_power = psd[beta_idx].mean()
                beta_powers.append(beta_power)

    if len(beta_powers) > 0:
        beta_powers = torch.stack(beta_powers)
        features = torch.stack([
            beta_powers.mean(),
            beta_powers.std()
        ]).to(device)
    else:
        features = torch.zeros(2, device=device)

    return features


def extract_all_erp_features(eeg_data, sfreq=100):
    """
    Extract all ERP and cognitive features

    Total features: 5 (P300) + 3 (N2) + 3 (alpha) + 2 (beta) = 13 features
    """
    p300_feats = extract_p300_features(eeg_data, sfreq)
    n2_feats = extract_n2_features(eeg_data, sfreq)
    alpha_feats = extract_prestimulus_alpha(eeg_data, sfreq)
    beta_feats = extract_motor_preparation(eeg_data, sfreq)

    all_features = torch.cat([p300_feats, n2_feats, alpha_feats, beta_feats])
    return all_features


class ERPMLP(nn.Module):
    """
    MLP using Event-Related Potential features for RT prediction

    Based on cognitive neuroscience literature showing:
    - P300 latency correlates 0.6-0.8 with RT
    - Pre-stimulus alpha predicts attention
    - Motor preparation signals predict response speed

    This should MASSIVELY improve C1 generalization!
    """

    def __init__(self, n_channels=129, sfreq=100, challenge_name='c1', output_range=(0.5, 1.5)):
        super().__init__()

        self.n_channels = n_channels
        self.sfreq = sfreq
        self.challenge_name = challenge_name
        self.is_c1 = (challenge_name == 'c1')
        self.output_min, self.output_max = output_range

        # 13 ERP features
        self.n_features = 13

        # Deeper MLP for ERP features
        self.mlp = nn.Sequential(
            nn.Linear(self.n_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(16, 1)
        )

        if self.is_c1:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Identity()

    def forward(self, x):
        """
        Args:
            x: (batch, n_channels, n_times)

        Returns:
            (batch, 1) predictions
        """
        batch_size = x.shape[0]

        # Extract ERP features for each sample
        features_list = []
        for i in range(batch_size):
            features = extract_all_erp_features(x[i], self.sfreq)
            features_list.append(features)

        features = torch.stack(features_list)  # (batch, n_features)

        # Pass through MLP
        output = self.mlp(features)
        output = self.activation(output)

        # Scale output for C1
        if self.is_c1:
            output = self.output_min + output * (self.output_max - self.output_min)

        return output


def create_erp_mlp(challenge='c1', device='cuda', **kwargs):
    """Factory function to create ERP MLP"""
    if challenge == 'c1':
        output_range = (0.5, 1.5)
    else:
        output_range = (-3, 3)

    model = ERPMLP(challenge_name=challenge, output_range=output_range, **kwargs)
    model = model.to(device)
    return model


if __name__ == "__main__":
    print("Testing ERP MLP...")

    model = create_erp_mlp(challenge='c1', device='cpu')

    # Test forward pass
    x = torch.randn(4, 129, 200)
    y = model(x)

    print(f"✅ Input shape: {x.shape}")
    print(f"✅ Output shape: {y.shape}")
    print(f"✅ Output range: [{y.min():.3f}, {y.max():.3f}]")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Parameters: {n_params:,}")

    print("\n🎉 ERP MLP test passed!")
    print("\n🧠 Uses neuroscience-based features:")
    print("   - P300 latency (strong RT correlate)")
    print("   - N2 amplitude (response inhibition)")
    print("   - Pre-stimulus alpha (attention)")
    print("   - Motor beta (preparation)")
