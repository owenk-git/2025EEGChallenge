"""
Quick diagnostic to check what targets we're actually getting

Run this to see:
1. Are targets all the same?
2. What values are targets?
3. Is official loader working?
"""

import torch
import numpy as np

# Try official loader
try:
    from data.official_eegdash_loader import create_official_eegdash_loaders
    print("Testing official loader...")

    train_loader, val_loader = create_official_eegdash_loaders(
        challenge='c1',
        batch_size=32,
        mini=True,  # Use mini for quick test
        num_workers=0
    )

    print("\n✅ Loaders created")

    # Check train targets
    all_train_targets = []
    for data, target in train_loader:
        all_train_targets.append(target.numpy())
        if len(all_train_targets) >= 3:  # Just check first 3 batches
            break

    train_targets = np.concatenate(all_train_targets)

    print(f"\n📊 Train targets (first {len(train_targets)} samples):")
    print(f"   Values: {train_targets[:20].flatten()}")
    print(f"   Mean: {train_targets.mean():.4f}")
    print(f"   Std:  {train_targets.std():.4f}")
    print(f"   Min:  {train_targets.min():.4f}")
    print(f"   Max:  {train_targets.max():.4f}")
    print(f"   Unique values: {len(np.unique(train_targets))}")

    # Check val targets
    all_val_targets = []
    for data, target in val_loader:
        all_val_targets.append(target.numpy())

    val_targets = np.concatenate(all_val_targets)

    print(f"\n📊 Val targets (all {len(val_targets)} samples):")
    print(f"   Values: {val_targets[:20].flatten()}")
    print(f"   Mean: {val_targets.mean():.4f}")
    print(f"   Std:  {val_targets.std():.4f}")
    print(f"   Min:  {val_targets.min():.4f}")
    print(f"   Max:  {val_targets.max():.4f}")
    print(f"   Unique values: {len(np.unique(val_targets))}")

    if val_targets.std() == 0:
        print(f"\n❌ PROBLEM: All val targets are {val_targets[0]}")
        print(f"   This causes NRMSE = inf")
        print(f"   Official RT extraction likely failed!")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
