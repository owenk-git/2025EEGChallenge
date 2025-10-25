"""
Train sklearn RandomForest models on EEG features

Similar to XGBoost but uses sklearn RandomForest which is:
- More likely to be available in Docker image
- Still excellent for tabular data
- Less likely to overfit than deep learning

Usage:
    python train_sklearn.py --challenge 1
    python train_sklearn.py --challenge 2
"""

import argparse
import numpy as np
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

# Import feature extraction from existing code
from features.eeg_features import extract_all_features
from data.official_eegdash_loader import create_official_eegdash_loaders


def extract_features_from_loader(dataloader, sfreq=100, max_samples=None):
    """
    Extract features from all samples in dataloader

    Args:
        dataloader: PyTorch DataLoader
        sfreq: Sampling frequency
        max_samples: Limit number of samples (for debugging)

    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
    """
    X_list = []
    y_list = []

    print(f"Extracting features from {len(dataloader)} batches...")

    for batch_idx, (data, target) in enumerate(tqdm(dataloader)):
        if max_samples and batch_idx * dataloader.batch_size >= max_samples:
            break

        batch_size = data.shape[0]

        for i in range(batch_size):
            # Extract features for this sample
            eeg_data = data[i].numpy()  # Shape: (n_channels, n_times)
            features = extract_all_features(eeg_data, sfreq, include_connectivity=False)

            # Convert to numpy array
            feature_vector = np.array(list(features.values()))
            X_list.append(feature_vector)

            # Target
            y_list.append(target[i].item())

    X = np.array(X_list)
    y = np.array(y_list)

    print(f"✅ Extracted {X.shape[0]} samples with {X.shape[1]} features")

    return X, y


def train_random_forest(X_train, y_train, X_val, y_val, n_estimators=100, max_depth=20):
    """
    Train Random Forest model

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        n_estimators: Number of trees
        max_depth: Max depth of trees

    Returns:
        model: Trained RandomForest model
        train_rmse, val_rmse: RMSE scores
    """
    print(f"\n🌲 Training RandomForest...")
    print(f"   n_estimators: {n_estimators}")
    print(f"   max_depth: {max_depth}")

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    # Train
    model.fit(X_train, y_train)

    # Evaluate
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

    # Normalize by std
    train_nrmse = train_rmse / np.std(y_train)
    val_nrmse = val_rmse / np.std(y_val)

    print(f"\n📊 Training Results:")
    print(f"   Train RMSE: {train_rmse:.4f}, NRMSE: {train_nrmse:.4f}")
    print(f"   Val RMSE:   {val_rmse:.4f}, NRMSE: {val_nrmse:.4f}")

    return model, train_rmse, val_rmse


def main():
    parser = argparse.ArgumentParser(description="Train sklearn RandomForest")

    parser.add_argument('--challenge', type=int, required=True, choices=[1, 2],
                       help='Challenge number (1 or 2)')
    parser.add_argument('--n_estimators', type=int, default=100,
                       help='Number of trees')
    parser.add_argument('--max_depth', type=int, default=20,
                       help='Max depth of trees')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Max samples to use (for debugging)')

    args = parser.parse_args()

    print("="*70)
    print(f"🌲 Training RandomForest for Challenge {args.challenge}")
    print("="*70)

    # Create output directory
    output_dir = Path('models_sklearn')
    output_dir.mkdir(exist_ok=True)

    # Load data
    print(f"\n📁 Loading Challenge {args.challenge} data...")
    train_loader, val_loader = create_official_eegdash_loaders(
        challenge=f'c{args.challenge}',
        batch_size=32,
        num_workers=4,
        val_split=0.2
    )

    print(f"✅ Loaded data:")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches:   {len(val_loader)}")

    # Extract features
    print(f"\n🔄 Extracting training features...")
    X_train, y_train = extract_features_from_loader(train_loader, max_samples=args.max_samples)

    print(f"\n🔄 Extracting validation features...")
    X_val, y_val = extract_features_from_loader(val_loader, max_samples=args.max_samples)

    # Train model
    model, train_rmse, val_rmse = train_random_forest(
        X_train, y_train,
        X_val, y_val,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth
    )

    # Save model
    model_path = output_dir / f'rf_c{args.challenge}.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n✅ Model saved to: {model_path}")

    print("\n" + "="*70)
    print("✅ RandomForest Training Complete!")
    print("="*70)
    print("\nTo create submission:")
    print("  python create_sklearn_submission.py")
    print("="*70)


if __name__ == "__main__":
    main()
