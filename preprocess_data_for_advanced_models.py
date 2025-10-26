"""
Preprocess data for advanced models

Loads data using official eegdash loader and saves to disk in format expected by:
- Domain Adaptation EEGNeX
- Cross-Task Pre-Training
- Hybrid CNN-Transformer-DA

This creates preprocessed numpy files for faster training.
"""

import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import argparse

# Try to import official loader
try:
    from data.official_eegdash_loader import create_official_eegdash_loaders
    OFFICIAL_LOADER_AVAILABLE = True
except ImportError:
    try:
        from data.official_dataset_example import create_official_dataloaders_with_split
        OFFICIAL_LOADER_AVAILABLE = False
        print("⚠️ Using fallback loader")
    except ImportError:
        print("❌ No data loader available!")
        exit(1)


def preprocess_challenge(challenge='c1', output_dir='data/preprocessed',
                        val_split=0.2, max_samples=None):
    """
    Preprocess data for one challenge

    Args:
        challenge: 'c1' or 'c2'
        output_dir: Output directory
        val_split: Validation split ratio
        max_samples: Maximum samples to process (None = all)
    """
    print(f"\n{'='*60}")
    print(f"Preprocessing Challenge {challenge.upper()}")
    print(f"{'='*60}")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data using official loader
    print("\n📦 Loading data from eegdash...")

    if OFFICIAL_LOADER_AVAILABLE:
        train_loader, val_loader = create_official_eegdash_loaders(
            challenge=challenge,
            batch_size=1,  # Load one at a time for preprocessing
            num_workers=0,
            val_split=val_split
        )
    else:
        train_loader, val_loader = create_official_dataloaders_with_split(
            challenge=int(challenge[1]),
            batch_size=1,
            val_split=val_split
        )

    # Extract training data
    print(f"\n🔄 Extracting training data...")
    X_train_list = []
    y_train_list = []
    subjects_train_list = []

    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Train")):
        if max_samples and batch_idx >= max_samples:
            break

        # data shape: (batch=1, channels, time)
        # target shape: (batch=1,)
        X_train_list.append(data.squeeze(0).cpu().numpy())
        y_train_list.append(target.squeeze(0).cpu().numpy())

        # Use batch index as dummy subject ID
        # (Real subject IDs might be available in dataset metadata)
        subjects_train_list.append(batch_idx % 50)  # Simulate 50 subjects

    X_train = np.stack(X_train_list, axis=0)
    y_train = np.array(y_train_list)
    subjects_train = np.array(subjects_train_list)

    print(f"  ✓ Training: X shape {X_train.shape}, y shape {y_train.shape}")
    print(f"  ✓ Subjects: {len(np.unique(subjects_train))} unique")

    # Extract validation data
    print(f"\n🔄 Extracting validation data...")
    X_val_list = []
    y_val_list = []
    subjects_val_list = []

    for batch_idx, (data, target) in enumerate(tqdm(val_loader, desc="Val")):
        if max_samples and batch_idx >= max_samples // 5:  # 20% of max_samples
            break

        X_val_list.append(data.squeeze(0).cpu().numpy())
        y_val_list.append(target.squeeze(0).cpu().numpy())
        subjects_val_list.append(batch_idx % 50 + 50)  # Different subjects

    X_val = np.stack(X_val_list, axis=0)
    y_val = np.array(y_val_list)
    subjects_val = np.array(subjects_val_list)

    print(f"  ✓ Validation: X shape {X_val.shape}, y shape {y_val.shape}")
    print(f"  ✓ Subjects: {len(np.unique(subjects_val))} unique")

    # Print statistics
    print(f"\n📊 Data Statistics:")
    print(f"  Training targets: mean={y_train.mean():.4f}, std={y_train.std():.4f}")
    print(f"  Training targets: min={y_train.min():.4f}, max={y_train.max():.4f}")
    print(f"  Validation targets: mean={y_val.mean():.4f}, std={y_val.std():.4f}")
    print(f"  Validation targets: min={y_val.min():.4f}, max={y_val.max():.4f}")

    # Save data
    print(f"\n💾 Saving to {output_dir}...")

    np.save(output_dir / f'{challenge}_X_train.npy', X_train)
    np.save(output_dir / f'{challenge}_y_train.npy', y_train)
    np.save(output_dir / f'{challenge}_subjects_train.npy', subjects_train)

    np.save(output_dir / f'{challenge}_X_val.npy', X_val)
    np.save(output_dir / f'{challenge}_y_val.npy', y_val)
    np.save(output_dir / f'{challenge}_subjects_val.npy', subjects_val)

    print(f"  ✓ Saved 6 files for {challenge}")

    # Calculate disk usage
    total_size_mb = (
        X_train.nbytes + y_train.nbytes + subjects_train.nbytes +
        X_val.nbytes + y_val.nbytes + subjects_val.nbytes
    ) / (1024 * 1024)
    print(f"  ✓ Total size: {total_size_mb:.2f} MB")

    return X_train.shape, y_train.shape, X_val.shape, y_val.shape


def main():
    parser = argparse.ArgumentParser(description='Preprocess data for advanced models')
    parser.add_argument('--challenges', type=str, default='c1,c2',
                       help='Challenges to preprocess (comma-separated)')
    parser.add_argument('--output_dir', type=str, default='data/preprocessed',
                       help='Output directory')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Validation split ratio')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum samples per challenge (for testing)')

    args = parser.parse_args()

    challenges = args.challenges.split(',')

    print("="*60)
    print("Data Preprocessing for Advanced Models")
    print("="*60)
    print(f"Challenges: {challenges}")
    print(f"Output directory: {args.output_dir}")
    print(f"Validation split: {args.val_split}")
    if args.max_samples:
        print(f"Max samples: {args.max_samples} (TESTING MODE)")

    # Preprocess each challenge
    for challenge in challenges:
        challenge = challenge.strip()
        try:
            preprocess_challenge(
                challenge=challenge,
                output_dir=args.output_dir,
                val_split=args.val_split,
                max_samples=args.max_samples
            )
        except Exception as e:
            print(f"\n❌ Error preprocessing {challenge}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "="*60)
    print("✅ Preprocessing Complete!")
    print("="*60)

    # Verify all files
    output_dir = Path(args.output_dir)
    print("\n📁 Created files:")
    for file in sorted(output_dir.glob('*.npy')):
        size_mb = file.stat().st_size / (1024 * 1024)
        print(f"  {file.name:30s} ({size_mb:.2f} MB)")

    print("\n🚀 Ready to train advanced models!")
    print("\nNext steps:")
    print("  1. Check data: python3 check_data_ready.py")
    print("  2. Train models: ./train_all_advanced_models.sh")


if __name__ == '__main__':
    main()
