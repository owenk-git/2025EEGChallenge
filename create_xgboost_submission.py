"""
Create submission using XGBoost models

This creates a submission.py that:
1. Extracts features from EEG data on-the-fly
2. Uses trained XGBoost models for prediction
3. Works with competition evaluation framework

Usage:
    python create_xgboost_submission.py --output xgboost_submission.zip
"""

import argparse
import zipfile
import shutil
from pathlib import Path
from datetime import datetime


XGBOOST_SUBMISSION_TEMPLATE = '''"""
EEG Challenge 2025 - XGBoost with Feature Engineering
Generated: {timestamp}

Approach: Classical ML with hand-crafted EEG features
- Band power (delta, theta, alpha, beta, gamma)
- Spectral features (entropy, peak frequency)
- Time-domain features (Hjorth parameters, statistics)
"""

import numpy as np
import xgboost as xgb
from scipy import signal, stats
from pathlib import Path


def load_model_path():
    """Return directory containing this submission.py"""
    return Path(__file__).parent


def extract_band_power(eeg_data, sfreq=100):
    """Extract power in frequency bands"""
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
            freqs, psd = signal.welch(eeg_data[ch], sfreq, nperseg=min(256, n_times))
            idx_band = np.logical_and(freqs >= low, freqs <= high)
            band_power = np.trapz(psd[idx_band], freqs[idx_band])
            band_powers.append(band_power)

        features[f'{{band_name}}_power_mean'] = np.mean(band_powers)
        features[f'{{band_name}}_power_std'] = np.std(band_powers)
        features[f'{{band_name}}_power_max'] = np.max(band_powers)

    total_power = sum([features[f'{{b}}_power_mean'] for b in bands.keys()])
    if total_power > 0:
        for band in bands.keys():
            features[f'{{band}}_power_relative'] = features[f'{{band}}_power_mean'] / total_power

    return features


def extract_spectral_features(eeg_data, sfreq=100):
    """Extract spectral features"""
    n_channels, n_times = eeg_data.shape
    features = {{}}

    spectral_entropies = []
    peak_freqs = []
    spectral_centroids = []

    for ch in range(n_channels):
        freqs, psd = signal.welch(eeg_data[ch], sfreq, nperseg=min(256, n_times))
        psd_norm = psd / np.sum(psd)

        spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
        spectral_entropies.append(spectral_entropy)

        peak_freq = freqs[np.argmax(psd)]
        peak_freqs.append(peak_freq)

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
    """Extract time-domain features"""
    n_channels, n_times = eeg_data.shape
    features = {{}}

    features['mean_mean'] = np.mean(np.mean(eeg_data, axis=1))
    features['mean_std'] = np.std(np.mean(eeg_data, axis=1))
    features['std_mean'] = np.mean(np.std(eeg_data, axis=1))
    features['std_std'] = np.std(np.std(eeg_data, axis=1))

    skews = [stats.skew(eeg_data[ch]) for ch in range(n_channels)]
    kurts = [stats.kurtosis(eeg_data[ch]) for ch in range(n_channels)]

    features['skew_mean'] = np.mean(skews)
    features['skew_std'] = np.std(skews)
    features['kurtosis_mean'] = np.mean(kurts)
    features['kurtosis_std'] = np.std(kurts)

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


class XGBoostPredictor:
    """Wrapper for XGBoost prediction"""

    def __init__(self, model_path, sfreq=100):
        self.model = xgb.Booster()
        self.model.load_model(str(model_path))
        self.sfreq = sfreq
        print(f"✅ Loaded XGBoost model from {{model_path}}")

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

        # Create DMatrix
        dmatrix = xgb.DMatrix(feature_vector)

        # Predict
        prediction = self.model.predict(dmatrix)[0]

        return prediction


class Submission:
    """Submission class for competition"""

    def __init__(self, SFREQ, DEVICE):
        self.sfreq = SFREQ
        self.device = DEVICE
        self.model_path = load_model_path()

        print(f"🌲 XGBoost Submission with Feature Engineering")
        print(f"   Device: {{DEVICE}}")
        print(f"   Sample rate: {{SFREQ}} Hz")

    def get_model_challenge_1(self):
        """Load Challenge 1 XGBoost model"""
        print("🌲 Loading Challenge 1 XGBoost model...")

        model_path = self.model_path / "xgboost_c1.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {{model_path}}")

        return XGBoostPredictor(model_path, self.sfreq)

    def get_model_challenge_2(self):
        """Load Challenge 2 XGBoost model"""
        print("🌲 Loading Challenge 2 XGBoost model...")

        model_path = self.model_path / "xgboost_c2.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {{model_path}}")

        return XGBoostPredictor(model_path, self.sfreq)
'''


def create_xgboost_submission(model_c1_path, model_c2_path, output_name=None):
    """
    Create XGBoost submission ZIP

    Args:
        model_c1_path: Path to C1 XGBoost model
        model_c2_path: Path to C2 XGBoost model
        output_name: Output ZIP filename
    """
    print("="*70)
    print("📦 Creating XGBoost Submission")
    print("="*70)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    if output_name is None:
        output_name = f"{timestamp}_xgboost_submission.zip"

    temp_dir = Path(f"temp_xgboost_{timestamp}")
    temp_dir.mkdir(exist_ok=True)

    try:
        # Write submission.py
        submission_content = XGBOOST_SUBMISSION_TEMPLATE.format(timestamp=timestamp)
        submission_path = temp_dir / "submission.py"
        with open(submission_path, 'w') as f:
            f.write(submission_content)
        print(f"✅ Created submission.py")

        # Copy XGBoost models
        if Path(model_c1_path).exists():
            shutil.copy(model_c1_path, temp_dir / "xgboost_c1.json")
            print(f"✅ Copied C1 XGBoost model: {model_c1_path}")
        else:
            print(f"⚠️  C1 model not found: {model_c1_path}")

        if Path(model_c2_path).exists():
            shutil.copy(model_c2_path, temp_dir / "xgboost_c2.json")
            print(f"✅ Copied C2 XGBoost model: {model_c2_path}")
        else:
            print(f"⚠️  C2 model not found: {model_c2_path}")

        # Create ZIP
        with zipfile.ZipFile(output_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in temp_dir.rglob('*'):
                if file.is_file():
                    arcname = file.relative_to(temp_dir)
                    zipf.write(file, arcname)

        print(f"\n✅ XGBoost submission created: {output_name}")

        zip_size = Path(output_name).stat().st_size / 1024
        print(f"   Size: {zip_size:.1f} KB")

        print(f"\n🎯 This submission uses:")
        print(f"   - Feature engineering (band power, spectral, time-domain)")
        print(f"   - XGBoost gradient boosting")
        print(f"   - Often beats deep learning on small EEG datasets!")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("="*70)
    print("🚀 Ready to submit!")
    print(f"   Upload {output_name} to Codabench")
    print("="*70)

    return output_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create XGBoost submission")

    parser.add_argument('--model_c1', type=str, default='models_xgboost/xgboost_c1.json',
                       help='Path to C1 XGBoost model')
    parser.add_argument('--model_c2', type=str, default='models_xgboost/xgboost_c2.json',
                       help='Path to C2 XGBoost model')
    parser.add_argument('--output', type=str, default=None,
                       help='Output ZIP filename')

    args = parser.parse_args()

    create_xgboost_submission(
        model_c1_path=args.model_c1,
        model_c2_path=args.model_c2,
        output_name=args.output
    )
