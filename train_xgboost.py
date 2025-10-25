"""
Train XGBoost models using extracted EEG features

This approach:
1. Extracts comprehensive EEG features (band power, spectral, connectivity)
2. Trains XGBoost regressor
3. Often beats deep learning on small EEG datasets

Usage:
    python train_xgboost.py --challenge 1
    python train_xgboost.py --challenge 2
"""

import argparse
import numpy as np
import xgboost as xgb
from pathlib import Path
import pickle
from tqdm import tqdm

from features.eeg_features import extract_all_features

try:
    from data.official_eegdash_loader import create_official_eegdash_loaders
    OFFICIAL_LOADER_AVAILABLE = True
except ImportError:
    OFFICIAL_LOADER_AVAILABLE = False
    from data.official_dataset_example import create_official_dataloaders_with_split


def extract_features_from_loader(dataloader, sfreq=100, max_samples=None):
    """
    Extract features from all samples in dataloader

    Args:
        dataloader: PyTorch DataLoader
        sfreq: sampling frequency
        max_samples: maximum number of samples to process (for debugging)

    Returns:
        X: (n_samples, n_features) numpy array
        y: (n_samples,) numpy array
    """
    X_list = []
    y_list = []

    print(f"📊 Extracting features from {len(dataloader)} batches...")

    n_samples = 0
    for batch_idx, (data, target) in enumerate(tqdm(dataloader, desc="Extracting features")):
        # data: (batch_size, n_channels, n_times)
        # target: (batch_size, 1)

        batch_size = data.shape[0]

        for i in range(batch_size):
            eeg_data = data[i].numpy()  # (n_channels, n_times)

            # Extract all features
            features = extract_all_features(eeg_data, sfreq, include_connectivity=False)  # Skip connectivity for speed

            # Convert to numpy array
            feature_vector = np.array(list(features.values()))

            X_list.append(feature_vector)
            y_list.append(target[i].item())

            n_samples += 1
            if max_samples and n_samples >= max_samples:
                break

        if max_samples and n_samples >= max_samples:
            break

    X = np.array(X_list)
    y = np.array(y_list)

    print(f"✅ Extracted features: X shape = {X.shape}, y shape = {y.shape}")

    return X, y


def train_xgboost(X_train, y_train, X_val, y_val):
    """
    Train XGBoost model

    Args:
        X_train, y_train: training data
        X_val, y_val: validation data

    Returns:
        trained XGBoost model
    """
    print("\n🌲 Training XGBoost...")

    # Create DMatrix (XGBoost's internal data structure)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # XGBoost parameters
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.1,  # L1 regularization
        'reg_lambda': 1.0,  # L2 regularization
        'seed': 42
    }

    # Train with early stopping
    evals = [(dtrain, 'train'), (dval, 'val')]

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=10
    )

    # Get best iteration
    best_iteration = model.best_iteration
    print(f"\n✅ Best iteration: {best_iteration}")

    # Evaluate
    y_pred_train = model.predict(dtrain, iteration_range=(0, best_iteration))
    y_pred_val = model.predict(dval, iteration_range=(0, best_iteration))

    # Compute NRMSE
    train_rmse = np.sqrt(np.mean((y_train - y_pred_train) ** 2))
    val_rmse = np.sqrt(np.mean((y_val - y_pred_val) ** 2))

    train_nrmse = train_rmse / np.std(y_train) if np.std(y_train) > 0 else float('inf')
    val_nrmse = val_rmse / np.std(y_val) if np.std(y_val) > 0 else float('inf')

    print(f"\n📊 Training Results:")
    print(f"   Train RMSE: {train_rmse:.4f}, NRMSE: {train_nrmse:.4f}")
    print(f"   Val RMSE:   {val_rmse:.4f}, NRMSE: {val_nrmse:.4f}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost on EEG features")

    parser.add_argument('-c', '--challenge', type=int, required=True, choices=[1, 2])
    parser.add_argument('--max_samples', type=int, default=None, help='Max samples for debugging')
    parser.add_argument('--output_dir', type=str, default='models_xgboost')

    args = parser.parse_args()

    print("="*70)
    print(f"🌲 XGBoost Training - Challenge {args.challenge}")
    print("="*70)
    print("Approach: Feature Engineering + Gradient Boosting")
    print("="*70)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load data
    print(f"\n📊 Loading data...")

    if OFFICIAL_LOADER_AVAILABLE:
        train_loader, val_loader = create_official_eegdash_loaders(
            challenge=f'c{args.challenge}',
            batch_size=32,
            mini=False,
            release="R11",
            num_workers=0  # Sequential for feature extraction
        )
    else:
        from data.official_dataset_example import create_official_dataloaders_with_split
        train_loader, val_loader = create_official_dataloaders_with_split(
            task="contrastChangeDetection",
            challenge=f'c{args.challenge}',
            batch_size=32,
            mini=False,
            num_workers=0,
            val_split=0.2
        )

    # Extract features
    X_train, y_train = extract_features_from_loader(train_loader, max_samples=args.max_samples)
    X_val, y_val = extract_features_from_loader(val_loader, max_samples=args.max_samples)

    # Train XGBoost
    model = train_xgboost(X_train, y_train, X_val, y_val)

    # Save model
    model_path = output_dir / f'xgboost_c{args.challenge}.json'
    model.save_model(str(model_path))
    print(f"\n✅ Model saved to: {model_path}")

    # Save feature names (for later use in submission)
    # Extract from a dummy sample to get feature names
    dummy_eeg = np.random.randn(129, 200)
    dummy_features = extract_all_features(dummy_eeg, include_connectivity=False)
    feature_names = list(dummy_features.keys())

    feature_names_path = output_dir / f'feature_names_c{args.challenge}.pkl'
    with open(feature_names_path, 'wb') as f:
        pickle.dump(feature_names, f)
    print(f"✅ Feature names saved to: {feature_names_path}")

    print("\n" + "="*70)
    print("✅ XGBoost Training Complete!")
    print("="*70)
    print(f"\nTo create submission:")
    print(f"  python create_xgboost_submission.py")
    print("="*70)


if __name__ == "__main__":
    main()
