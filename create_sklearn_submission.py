"""
Create submission using sklearn RandomForest models

This creates a submission.py that:
1. Extracts features from EEG data on-the-fly
2. Uses trained RandomForest models for prediction
3. Works with competition evaluation framework
4. Uses only numpy and sklearn (more likely to be available)

Usage:
    python create_sklearn_submission.py --output sklearn_submission.zip
"""

import argparse
import zipfile
import shutil
import pickle
from pathlib import Path
from datetime import datetime


SKLEARN_SUBMISSION_TEMPLATE = '''"""
EEG Challenge 2025 - Random Forest with Feature Engineering
Generated: {timestamp}

Approach: Classical ML with hand-crafted EEG features
- Band power (delta, theta, alpha, beta, gamma) using numpy FFT
- Spectral features (entropy, peak frequency)
- Time-domain features (Hjorth parameters, statistics)
- Uses only numpy and sklearn (no scipy dependency)
"""

import numpy as np
import pickle
from pathlib import Path


def load_model_path():
    """Return directory containing this submission.py"""
    return Path(__file__).parent


def compute_psd_numpy(signal, sfreq=100, nperseg=256):
    """
    Compute power spectral density using numpy FFT

    Alternative to scipy.signal.welch that uses only numpy
    """
    # Ensure nperseg doesn't exceed signal length
    nperseg = min(nperseg, len(signal))

    # Simple periodogram using FFT
    windowed = signal[:nperseg] * np.hanning(nperseg)
    fft_result = np.fft.rfft(windowed)
    psd = np.abs(fft_result) ** 2 / nperseg
    freqs = np.fft.rfftfreq(nperseg, 1/sfreq)

    return freqs, psd


def extract_band_power(eeg_data, sfreq=100):
    """Extract power in frequency bands using numpy FFT"""
    n_channels, n_times = eeg_data.shape

    bands = {{
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 50)
    }}

    features = {{}}

    for band_name, (low, high) in bands.items():
        band_powers = []

        for ch in range(n_channels):
            freqs, psd = compute_psd_numpy(eeg_data[ch], sfreq, nperseg=min(256, n_times))
            idx_band = np.logical_and(freqs >= low, freqs <= high)

            if np.any(idx_band):
                # Trapezoidal integration
                band_power = np.trapz(psd[idx_band], freqs[idx_band])
                band_powers.append(band_power)

        if len(band_powers) > 0:
            features[f'{{band_name}}_power_mean'] = np.mean(band_powers)
            features[f'{{band_name}}_power_std'] = np.std(band_powers)
            features[f'{{band_name}}_power_max'] = np.max(band_powers)
        else:
            features[f'{{band_name}}_power_mean'] = 0
            features[f'{{band_name}}_power_std'] = 0
            features[f'{{band_name}}_power_max'] = 0

    # Relative power
    total_power = sum([features[f'{{b}}_power_mean'] for b in bands.keys()])
    if total_power > 0:
        for band in bands.keys():
            features[f'{{band}}_power_relative'] = features[f'{{band}}_power_mean'] / total_power
    else:
        for band in bands.keys():
            features[f'{{band}}_power_relative'] = 0

    return features


def compute_skewness(x):
    """Compute skewness using numpy"""
    mean = np.mean(x)
    std = np.std(x)
    if std == 0:
        return 0
    return np.mean(((x - mean) / std) ** 3)


def compute_kurtosis(x):
    """Compute kurtosis using numpy"""
    mean = np.mean(x)
    std = np.std(x)
    if std == 0:
        return 0
    return np.mean(((x - mean) / std) ** 4) - 3  # Excess kurtosis


def extract_spectral_features(eeg_data, sfreq=100):
    """Extract spectral features using numpy FFT"""
    n_channels, n_times = eeg_data.shape
    features = {{}}

    spectral_entropies = []
    peak_freqs = []
    spectral_centroids = []

    for ch in range(n_channels):
        freqs, psd = compute_psd_numpy(eeg_data[ch], sfreq, nperseg=min(256, n_times))

        # Normalize PSD
        psd_sum = np.sum(psd)
        if psd_sum > 0:
            psd_norm = psd / psd_sum
        else:
            psd_norm = psd + 1e-10

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
    """Extract time-domain features using numpy"""
    n_channels, n_times = eeg_data.shape
    features = {{}}

    # Basic statistics
    features['mean_mean'] = np.mean(np.mean(eeg_data, axis=1))
    features['mean_std'] = np.std(np.mean(eeg_data, axis=1))
    features['std_mean'] = np.mean(np.std(eeg_data, axis=1))
    features['std_std'] = np.std(np.std(eeg_data, axis=1))

    # Skewness and kurtosis
    skews = [compute_skewness(eeg_data[ch]) for ch in range(n_channels)]
    kurts = [compute_kurtosis(eeg_data[ch]) for ch in range(n_channels)]

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

        activity = np.var(x)
        activities.append(activity)

        mobility = np.sqrt(np.var(dx) / np.var(x)) if np.var(x) > 0 else 0
        mobilities.append(mobility)

        complexity = (np.sqrt(np.var(ddx) / np.var(dx)) / mobility) if (mobility > 0 and np.var(dx) > 0) else 0
        complexities.append(complexity)

    features['hjorth_activity_mean'] = np.mean(activities)
    features['hjorth_mobility_mean'] = np.mean(mobilities)
    features['hjorth_complexity_mean'] = np.mean(complexities)

    # Zero-crossing rate
    zero_crossings = []
    for ch in range(n_channels):
        zc = np.sum(np.diff(np.sign(eeg_data[ch])) != 0)
        zero_crossings.append(zc / n_times)

    features['zero_crossing_rate_mean'] = np.mean(zero_crossings)
    features['zero_crossing_rate_std'] = np.std(zero_crossings)

    return features


def extract_all_features(eeg_data, sfreq=100):
    """Extract all features from EEG data"""
    features = {{}}
    features.update(extract_band_power(eeg_data, sfreq))
    features.update(extract_spectral_features(eeg_data, sfreq))
    features.update(extract_time_features(eeg_data))
    return features


class RandomForestPredictor:
    """Wrapper for RandomForest prediction"""

    def __init__(self, model_path, sfreq=100):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        self.sfreq = sfreq
        print(f"✅ Loaded RandomForest model from {{model_path}}")

    def predict(self, eeg_data):
        """
        Predict from EEG data

        Args:
            eeg_data: (n_channels, n_times) numpy array

        Returns:
            prediction: scalar value
        """
        # Extract features
        features = extract_all_features(eeg_data, self.sfreq)

        # Convert to numpy array (must match training order)
        feature_vector = np.array(list(features.values())).reshape(1, -1)

        # Predict
        prediction = self.model.predict(feature_vector)[0]

        return prediction


class Submission:
    """Submission class for competition"""

    def __init__(self, SFREQ, DEVICE):
        self.sfreq = SFREQ
        self.device = DEVICE
        self.model_path = load_model_path()

        print(f"🌲 RandomForest Submission with Feature Engineering")
        print(f"   Device: {{DEVICE}}")
        print(f"   Sample rate: {{SFREQ}} Hz")

    def get_model_challenge_1(self):
        """Load Challenge 1 RandomForest model"""
        print("🌲 Loading Challenge 1 RandomForest model...")

        model_path = self.model_path / "rf_c1.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {{model_path}}")

        return RandomForestPredictor(model_path, self.sfreq)

    def get_model_challenge_2(self):
        """Load Challenge 2 RandomForest model"""
        print("🌲 Loading Challenge 2 RandomForest model...")

        model_path = self.model_path / "rf_c2.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {{model_path}}")

        return RandomForestPredictor(model_path, self.sfreq)
'''


def create_sklearn_submission(model_c1_path, model_c2_path, output_name=None):
    """
    Create sklearn RandomForest submission ZIP

    Args:
        model_c1_path: Path to C1 RandomForest model
        model_c2_path: Path to C2 RandomForest model
        output_name: Output ZIP filename
    """
    print("="*70)
    print("📦 Creating sklearn RandomForest Submission")
    print("="*70)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    if output_name is None:
        output_name = f"{timestamp}_sklearn_submission.zip"

    temp_dir = Path(f"temp_sklearn_{timestamp}")
    temp_dir.mkdir(exist_ok=True)

    try:
        # Write submission.py
        submission_content = SKLEARN_SUBMISSION_TEMPLATE.format(timestamp=timestamp)
        submission_path = temp_dir / "submission.py"
        with open(submission_path, 'w') as f:
            f.write(submission_content)
        print(f"✅ Created submission.py")

        # Copy RandomForest models
        if Path(model_c1_path).exists():
            shutil.copy(model_c1_path, temp_dir / "rf_c1.pkl")
            print(f"✅ Copied C1 RandomForest model: {model_c1_path}")
        else:
            print(f"⚠️  C1 model not found: {model_c1_path}")

        if Path(model_c2_path).exists():
            shutil.copy(model_c2_path, temp_dir / "rf_c2.pkl")
            print(f"✅ Copied C2 RandomForest model: {model_c2_path}")
        else:
            print(f"⚠️  C2 model not found: {model_c2_path}")

        # Create ZIP
        with zipfile.ZipFile(output_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in temp_dir.rglob('*'):
                if file.is_file():
                    arcname = file.relative_to(temp_dir)
                    zipf.write(file, arcname)

        print(f"\n✅ sklearn submission created: {output_name}")

        zip_size = Path(output_name).stat().st_size / 1024
        print(f"   Size: {zip_size:.1f} KB")

        print(f"\n🎯 This submission uses:")
        print(f"   - Feature engineering (band power, spectral, time-domain)")
        print(f"   - sklearn RandomForest (more compatible than XGBoost)")
        print(f"   - Only numpy and sklearn (no scipy dependency)")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("="*70)
    print("🚀 Ready to submit!")
    print(f"   Upload {output_name} to Codabench")
    print("="*70)

    return output_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create sklearn submission")

    parser.add_argument('--model_c1', type=str, default='models_sklearn/rf_c1.pkl',
                       help='Path to C1 RandomForest model')
    parser.add_argument('--model_c2', type=str, default='models_sklearn/rf_c2.pkl',
                       help='Path to C2 RandomForest model')
    parser.add_argument('--output', type=str, default=None,
                       help='Output ZIP filename')

    args = parser.parse_args()

    create_sklearn_submission(
        model_c1_path=args.model_c1,
        model_c2_path=args.model_c2,
        output_name=args.output
    )
