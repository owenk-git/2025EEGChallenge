"""
Check if preprocessed data is ready for training advanced models
"""

import numpy as np
from pathlib import Path


def check_data():
    """Check if all required data files exist"""
    data_dir = Path('data/preprocessed')

    print("Checking preprocessed data...")
    print("=" * 50)

    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        print("Please preprocess data first!")
        return False

    required_files = [
        'c1_X_train.npy',
        'c1_y_train.npy',
        'c1_X_val.npy',
        'c1_y_val.npy',
        'c2_X_train.npy',
        'c2_y_train.npy',
        'c2_X_val.npy',
        'c2_y_val.npy'
    ]

    all_exist = True
    for filename in required_files:
        filepath = data_dir / filename
        if filepath.exists():
            # Load and check shape
            data = np.load(filepath)
            print(f"✓ {filename:25s} shape: {data.shape}")
        else:
            print(f"❌ {filename:25s} NOT FOUND")
            all_exist = False

    print("=" * 50)

    # Check for optional subject files
    print("\nChecking optional files:")
    optional_files = [
        'c1_subjects_train.npy',
        'c1_subjects_val.npy',
        'c2_subjects_train.npy',
        'c2_subjects_val.npy'
    ]

    for filename in optional_files:
        filepath = data_dir / filename
        if filepath.exists():
            data = np.load(filepath)
            print(f"✓ {filename:25s} shape: {data.shape}")
        else:
            print(f"⚠ {filename:25s} NOT FOUND (will use dummy subjects)")

    print("=" * 50)

    if all_exist:
        print("\n✓ All required data files found!")
        print("  Ready to train models.")
        print("\nRun:")
        print("  ./train_all_advanced_models.sh")
        print("or")
        print("  python3 train_domain_adaptation.py --challenge c1 --epochs 100")
        return True
    else:
        print("\n❌ Missing required data files!")
        print("  Please run data preprocessing first.")
        return False


if __name__ == '__main__':
    check_data()
