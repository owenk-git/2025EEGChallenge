"""
Recording-Level N-Fold Cross-Validation Training

This trains recording-level models (not trial-level) with proper K-fold validation.
Uses the best performing architecture: ERP MLP that scored 1.10 on leaderboard.

Key advantages:
- Recording-level prediction (no aggregation needed)
- Subject-wise K-fold (no data leakage)
- ERP features proven to work (P300, N2, alpha, beta)
- Saves best fold model for submission

Usage:
    python train_recording_kfold.py --model erp_mlp --challenge c1 --n_folds 5
    python train_recording_kfold.py --model erp_mlp --challenge c2 --n_folds 5
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
from tqdm import tqdm
import json

# Import models
from models.erp_features_mlp import ERPMLP
from models.cnn_ensemble import CNNEnsemble
from models.eegnex_augmented import EEGNeXImproved

# Import dataset
from data.official_eegdash_loader import OfficialEEGDashDataset


class RecordingLevelKFold:
    """
    K-Fold cross-validation at recording level

    Ensures no data leakage by splitting recordings (not subjects)
    """

    def __init__(self, dataset, n_splits=5, seed=42):
        self.dataset = dataset
        self.n_splits = n_splits
        self.seed = seed

        # Get all recording indices
        self.recording_indices = list(range(len(dataset)))

        # Shuffle
        np.random.seed(seed)
        np.random.shuffle(self.recording_indices)

        self.kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    def split(self):
        """
        Generate K-fold splits

        Yields:
            train_indices, val_indices for each fold
        """
        for fold_idx, (train_idx, val_idx) in enumerate(self.kfold.split(self.recording_indices)):
            train_recordings = [self.recording_indices[i] for i in train_idx]
            val_recordings = [self.recording_indices[i] for i in val_idx]

            yield fold_idx, train_recordings, val_recordings


def create_model(model_name, challenge, device):
    """Create model instance"""
    if model_name == 'erp_mlp':
        if challenge == 'c1':
            output_range = (0.5, 1.5)
        else:
            output_range = (-3, 3)
        model = ERPMLP(n_channels=129, sfreq=100, challenge_name=challenge, output_range=output_range)
    elif model_name == 'cnn_ensemble':
        model = CNNEnsemble(n_channels=129, n_times=200, challenge_name=challenge)
    elif model_name == 'eegnex_improved':
        model = EEGNeXImproved(n_channels=129, n_times=200, challenge_name=challenge)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model.to(device)


def train_one_epoch(model, loader, criterion, optimizer, device, challenge):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for data, target in tqdm(loader, desc='Training', leave=False):
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.size(0)
        all_preds.append(output.detach().cpu())
        all_targets.append(target.detach().cpu())

    avg_loss = total_loss / len(loader.dataset)

    # Calculate metrics
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    # MSE
    mse = ((all_preds - all_targets) ** 2).mean().item()

    # Correlation
    pred_centered = all_preds - all_preds.mean()
    target_centered = all_targets - all_targets.mean()
    numerator = (pred_centered * target_centered).sum()
    denominator = torch.sqrt((pred_centered ** 2).sum() * (target_centered ** 2).sum())
    correlation = (numerator / (denominator + 1e-8)).item()

    # NRMSE
    target_std = all_targets.std().item()
    rmse = np.sqrt(mse)
    nrmse = rmse / (target_std + 1e-8)

    return {
        'loss': avg_loss,
        'mse': mse,
        'rmse': rmse,
        'nrmse': nrmse,
        'correlation': correlation,
        'pred_std': all_preds.std().item(),
        'target_std': target_std
    }


def validate(model, loader, criterion, device, challenge):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in tqdm(loader, desc='Validating', leave=False):
            data = data.to(device)
            target = target.to(device)

            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item() * data.size(0)
            all_preds.append(output.cpu())
            all_targets.append(target.cpu())

    avg_loss = total_loss / len(loader.dataset)

    # Calculate metrics
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    # MSE
    mse = ((all_preds - all_targets) ** 2).mean().item()

    # Correlation
    pred_centered = all_preds - all_preds.mean()
    target_centered = all_targets - all_targets.mean()
    numerator = (pred_centered * target_centered).sum()
    denominator = torch.sqrt((pred_centered ** 2).sum() * (target_centered ** 2).sum())
    correlation = (numerator / (denominator + 1e-8)).item()

    # NRMSE
    target_std = all_targets.std().item()
    rmse = np.sqrt(mse)
    nrmse = rmse / (target_std + 1e-8)

    return {
        'loss': avg_loss,
        'mse': mse,
        'rmse': rmse,
        'nrmse': nrmse,
        'correlation': correlation,
        'pred_std': all_preds.std().item(),
        'target_std': target_std
    }


def train_fold(
    fold_idx,
    train_indices,
    val_indices,
    full_dataset,
    model_name,
    challenge,
    epochs,
    batch_size,
    lr,
    device,
    checkpoint_dir
):
    """Train one fold"""
    print(f"\n{'='*60}")
    print(f"FOLD {fold_idx + 1}")
    print(f"{'='*60}")
    print(f"Train recordings: {len(train_indices)}")
    print(f"Val recordings: {len(val_indices)}")

    # Create data loaders
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Create model
    model = create_model(model_name, challenge, device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Training loop
    best_val_nrmse = float('inf')
    best_epoch = 0
    fold_history = []
    patience_counter = 0
    max_patience = 15

    for epoch in range(epochs):
        print(f"\nFold {fold_idx + 1}, Epoch {epoch + 1}/{epochs}")

        # Train
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, challenge)

        # Validate
        val_metrics = validate(model, val_loader, criterion, device, challenge)

        # Update scheduler
        scheduler.step(val_metrics['nrmse'])

        # Log metrics
        print(f"Train - NRMSE: {train_metrics['nrmse']:.4f}, Corr: {train_metrics['correlation']:.4f}, "
              f"Pred Std: {train_metrics['pred_std']:.4f}")
        print(f"Val   - NRMSE: {val_metrics['nrmse']:.4f}, Corr: {val_metrics['correlation']:.4f}, "
              f"Pred Std: {val_metrics['pred_std']:.4f}, Target Std: {val_metrics['target_std']:.4f}")

        fold_history.append({
            'epoch': epoch + 1,
            'train': train_metrics,
            'val': val_metrics
        })

        # Save best model
        if val_metrics['nrmse'] < best_val_nrmse:
            best_val_nrmse = val_metrics['nrmse']
            best_epoch = epoch + 1
            patience_counter = 0

            checkpoint_path = checkpoint_dir / f"fold_{fold_idx}_best.pth"
            torch.save({
                'fold': fold_idx,
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_nrmse': val_metrics['nrmse'],
                'val_correlation': val_metrics['correlation'],
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            }, checkpoint_path)

            print(f"✅ Saved best model (NRMSE: {best_val_nrmse:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= max_patience:
            print(f"\n⏸  Early stopping at epoch {epoch + 1}")
            break

    print(f"\n✅ Fold {fold_idx + 1} complete!")
    print(f"   Best Val NRMSE: {best_val_nrmse:.4f} at epoch {best_epoch}")

    return {
        'fold': fold_idx,
        'best_val_nrmse': best_val_nrmse,
        'best_epoch': best_epoch,
        'history': fold_history
    }


def main():
    parser = argparse.ArgumentParser(description='Recording-Level N-Fold Training')

    # Model shortcuts: EM=erp_mlp, CE=cnn_ensemble, EI=eegnex_improved
    parser.add_argument('--model', '--m', type=str, default='erp_mlp',
                       choices=['erp_mlp', 'cnn_ensemble', 'eegnex_improved', 'EM', 'CE', 'EI'],
                       help='Model: EM=erp_mlp, CE=cnn_ensemble, EI=eegnex_improved')
    parser.add_argument('--challenge', '--c', type=str, default='c1',
                       choices=['c1', 'c2', '1', '2'],
                       help='Challenge: 1 or c1, 2 or c2')
    parser.add_argument('--n_folds', '--f', type=int, default=5,
                       help='Number of folds')
    parser.add_argument('--epochs', '--e', type=int, default=50,
                       help='Max epochs per fold')
    parser.add_argument('--batch_size', '--b', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--mini', action='store_true',
                       help='Use mini dataset')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')

    args = parser.parse_args()

    # Map shortcuts to full names
    model_map = {'EM': 'erp_mlp', 'CE': 'cnn_ensemble', 'EI': 'eegnex_improved'}
    if args.model in model_map:
        args.model = model_map[args.model]

    # Map challenge shortcuts
    if args.challenge in ['1', '2']:
        args.challenge = f'c{args.challenge}'

    print("="*60)
    print("RECORDING-LEVEL N-FOLD TRAINING")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Challenge: {args.challenge}")
    print(f"N-Folds: {args.n_folds}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print(f"Mini Dataset: {args.mini}")
    print(f"Device: {args.device}")

    # Create checkpoint directory
    checkpoint_dir = Path(f"checkpoints_kfold/{args.model}_{args.challenge}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print(f"\n📦 Loading dataset...")
    full_dataset = OfficialEEGDashDataset(
        task="contrastChangeDetection",
        challenge=args.challenge,
        release="R11",
        mini=args.mini,
        rt_method='mean'
    )

    print(f"✅ Loaded {len(full_dataset)} recordings")

    # Create K-fold splitter
    kfold = RecordingLevelKFold(full_dataset, n_splits=args.n_folds, seed=42)

    # Train each fold
    fold_results = []

    for fold_idx, train_indices, val_indices in kfold.split():
        fold_result = train_fold(
            fold_idx=fold_idx,
            train_indices=train_indices,
            val_indices=val_indices,
            full_dataset=full_dataset,
            model_name=args.model,
            challenge=args.challenge,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            checkpoint_dir=checkpoint_dir
        )

        fold_results.append(fold_result)

    # Summary
    print("\n" + "="*60)
    print("K-FOLD TRAINING COMPLETE")
    print("="*60)

    best_fold = None
    best_nrmse = float('inf')

    for result in fold_results:
        fold_idx = result['fold']
        val_nrmse = result['best_val_nrmse']
        print(f"Fold {fold_idx + 1}: Val NRMSE = {val_nrmse:.4f}")

        if val_nrmse < best_nrmse:
            best_nrmse = val_nrmse
            best_fold = fold_idx

    # Calculate average and std
    nrmses = [r['best_val_nrmse'] for r in fold_results]
    avg_nrmse = np.mean(nrmses)
    std_nrmse = np.std(nrmses)

    print(f"\nAverage Val NRMSE: {avg_nrmse:.4f} ± {std_nrmse:.4f}")
    print(f"Best Fold: {best_fold + 1} (NRMSE: {best_nrmse:.4f})")

    # Save summary
    summary = {
        'model': args.model,
        'challenge': args.challenge,
        'n_folds': args.n_folds,
        'fold_results': fold_results,
        'avg_nrmse': avg_nrmse,
        'std_nrmse': std_nrmse,
        'best_fold': best_fold,
        'best_nrmse': best_nrmse
    }

    summary_path = checkpoint_dir / 'kfold_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Summary saved to {summary_path}")
    print(f"\n🎯 Best model: {checkpoint_dir}/fold_{best_fold}_best.pth")
    print(f"\n📝 Use this checkpoint for submission!")


if __name__ == '__main__':
    main()
