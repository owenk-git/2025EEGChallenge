"""
Comprehensive EEG Feature Extraction for Competition

Extracts multiple categories of features:
1. Band Power (delta, theta, alpha, beta, gamma)
2. Spectral Features (peak frequency, spectral entropy)
3. Time-domain Features (entropy, complexity)
4. Connectivity Features (coherence, phase locking)
5. Statistical Features (mean, std, skew, kurtosis)

Usage:
    from features.eeg_features import extract_all_features
    features = extract_all_features(eeg_data, sfreq=100)
"""

import numpy as np
from scipy import signal, stats
from scipy.fft import fft, fftfreq
import warnings
warnings.filterwarnings('ignore')


def extract_band_power(eeg_data, sfreq=100):
    """
    Extract power in different frequency bands

    Bands:
    - Delta: 0.5-4 Hz (deep sleep)
    - Theta: 4-8 Hz (drowsiness, meditation)
    - Alpha: 8-13 Hz (relaxed, eyes closed)
    - Beta: 13-30 Hz (active thinking)
    - Gamma: 30-50 Hz (high-level cognition)

    Args:
        eeg_data: (n_channels, n_times) EEG data
        sfreq: sampling frequency

    Returns:
        dict with band powers for each channel
    """
    n_channels, n_times = eeg_data.shape

    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 50)
    }

    features = {}

    for band_name, (low, high) in bands.items():
        band_powers = []

        for ch in range(n_channels):
            # Compute power spectral density
            freqs, psd = signal.welch(eeg_data[ch], sfreq, nperseg=min(256, n_times))

            # Find frequencies in band
            idx_band = np.logical_and(freqs >= low, freqs <= high)

            # Integrate power in band
            band_power = np.trapz(psd[idx_band], freqs[idx_band])
            band_powers.append(band_power)

        # Aggregate across channels
        features[f'{band_name}_power_mean'] = np.mean(band_powers)
        features[f'{band_name}_power_std'] = np.std(band_powers)
        features[f'{band_name}_power_max'] = np.max(band_powers)

    # Relative band powers
    total_power = sum([features[f'{b}_power_mean'] for b in bands.keys()])
    if total_power > 0:
        for band in bands.keys():
            features[f'{band}_power_relative'] = features[f'{band}_power_mean'] / total_power

    return features


def extract_spectral_features(eeg_data, sfreq=100):
    """
    Extract spectral features from EEG

    Features:
    - Spectral entropy
    - Peak frequency
    - Spectral edge frequency
    - Spectral centroid
    """
    n_channels, n_times = eeg_data.shape

    features = {}

    spectral_entropies = []
    peak_freqs = []
    spectral_centroids = []

    for ch in range(n_channels):
        # Compute PSD
        freqs, psd = signal.welch(eeg_data[ch], sfreq, nperseg=min(256, n_times))

        # Normalize PSD
        psd_norm = psd / np.sum(psd)

        # Spectral entropy
        spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
        spectral_entropies.append(spectral_entropy)

        # Peak frequency
        peak_freq = freqs[np.argmax(psd)]
        peak_freqs.append(peak_freq)

        # Spectral centroid
        spectral_centroid = np.sum(freqs * psd_norm)
        spectral_centroids.append(spectral_centroid)

    features['spectral_entropy_mean'] = np.mean(spectral_entropies)
    features['spectral_entropy_std'] = np.std(spectral_entropies)
    features['peak_freq_mean'] = np.mean(peak_freqs)
    features['peak_freq_std'] = np.std(peak_freqs)
    features['spectral_centroid_mean'] = np.mean(spectral_centroids)
    features['spectral_centroid_std'] = np.std(spectral_centroids)

    return features


def extract_time_features(eeg_data):
    """
    Extract time-domain features

    Features:
    - Sample entropy
    - Hjorth parameters (activity, mobility, complexity)
    - Zero-crossing rate
    - Statistical moments
    """
    n_channels, n_times = eeg_data.shape

    features = {}

    # Statistical moments across channels
    features['mean_mean'] = np.mean(np.mean(eeg_data, axis=1))
    features['mean_std'] = np.std(np.mean(eeg_data, axis=1))
    features['std_mean'] = np.mean(np.std(eeg_data, axis=1))
    features['std_std'] = np.std(np.std(eeg_data, axis=1))

    # Skewness and kurtosis
    skews = [stats.skew(eeg_data[ch]) for ch in range(n_channels)]
    kurts = [stats.kurtosis(eeg_data[ch]) for ch in range(n_channels)]

    features['skew_mean'] = np.mean(skews)
    features['skew_std'] = np.std(skews)
    features['kurtosis_mean'] = np.mean(kurts)
    features['kurtosis_std'] = np.std(kurts)

    # Hjorth parameters
    activities = []
    mobilities = []
    complexities = []

    for ch in range(n_channels):
        x = eeg_data[ch]
        dx = np.diff(x)
        ddx = np.diff(dx)

        # Activity (variance)
        activity = np.var(x)
        activities.append(activity)

        # Mobility
        mobility = np.sqrt(np.var(dx) / np.var(x)) if np.var(x) > 0 else 0
        mobilities.append(mobility)

        # Complexity
        complexity = (np.sqrt(np.var(ddx) / np.var(dx)) / mobility) if (mobility > 0 and np.var(dx) > 0) else 0
        complexities.append(complexity)

    features['hjorth_activity_mean'] = np.mean(activities)
    features['hjorth_mobility_mean'] = np.mean(mobilities)
    features['hjorth_complexity_mean'] = np.mean(complexities)

    # Zero crossing rate
    zero_crossings = []
    for ch in range(n_channels):
        zc = np.sum(np.diff(np.sign(eeg_data[ch])) != 0)
        zero_crossings.append(zc / n_times)

    features['zero_crossing_rate_mean'] = np.mean(zero_crossings)
    features['zero_crossing_rate_std'] = np.std(zero_crossings)

    return features


def extract_connectivity_features(eeg_data, sfreq=100):
    """
    Extract connectivity features between channels

    Features:
    - Mean coherence
    - Phase locking value
    - Correlation
    """
    n_channels, n_times = eeg_data.shape

    features = {}

    # Sample subset of channel pairs (too expensive to do all)
    n_pairs = min(50, n_channels * (n_channels - 1) // 2)

    coherences = []
    correlations = []

    np.random.seed(42)
    pairs = []
    for _ in range(n_pairs):
        i, j = np.random.choice(n_channels, 2, replace=False)
        pairs.append((i, j))

    for i, j in pairs:
        # Correlation
        corr = np.corrcoef(eeg_data[i], eeg_data[j])[0, 1]
        correlations.append(abs(corr))

        # Coherence
        try:
            freqs, coh = signal.coherence(eeg_data[i], eeg_data[j], sfreq, nperseg=min(64, n_times))
            coherences.append(np.mean(coh))
        except:
            pass

    features['correlation_mean'] = np.mean(correlations) if correlations else 0
    features['correlation_std'] = np.std(correlations) if correlations else 0
    features['coherence_mean'] = np.mean(coherences) if coherences else 0
    features['coherence_std'] = np.std(coherences) if coherences else 0

    return features


def extract_all_features(eeg_data, sfreq=100, include_connectivity=True):
    """
    Extract all features from EEG data

    Args:
        eeg_data: (n_channels, n_times) numpy array
        sfreq: sampling frequency in Hz
        include_connectivity: whether to include connectivity features (slower)

    Returns:
        dict of features
    """
    features = {}

    # Band power features
    features.update(extract_band_power(eeg_data, sfreq))

    # Spectral features
    features.update(extract_spectral_features(eeg_data, sfreq))

    # Time-domain features
    features.update(extract_time_features(eeg_data))

    # Connectivity features (optional, slower)
    if include_connectivity:
        features.update(extract_connectivity_features(eeg_data, sfreq))

    return features


if __name__ == "__main__":
    # Test feature extraction
    print("Testing EEG feature extraction...")

    # Create dummy EEG data
    n_channels = 129
    n_times = 200
    sfreq = 100

    eeg_data = np.random.randn(n_channels, n_times)

    # Extract features
    features = extract_all_features(eeg_data, sfreq, include_connectivity=True)

    print(f"\n✅ Extracted {len(features)} features:")
    for name, value in list(features.items())[:10]:
        print(f"  {name}: {value:.4f}")

    print(f"\n📊 Feature categories:")
    band_features = [k for k in features.keys() if any(b in k for b in ['delta', 'theta', 'alpha', 'beta', 'gamma'])]
    spectral_features = [k for k in features.keys() if 'spectral' in k or 'peak' in k or 'centroid' in k]
    time_features = [k for k in features.keys() if any(t in k for t in ['mean', 'std', 'skew', 'hjorth', 'zero'])]
    connectivity_features = [k for k in features.keys() if any(c in k for c in ['correlation', 'coherence'])]

    print(f"  Band power: {len(band_features)}")
    print(f"  Spectral: {len(spectral_features)}")
    print(f"  Time-domain: {len(time_features)}")
    print(f"  Connectivity: {len(connectivity_features)}")
    print(f"  Total: {len(features)}")
