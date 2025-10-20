"""
K-Fold Cross-Validation Training for EEG Challenge

This script implements K-fold CV with subject-wise splitting to:
1. Get robust performance estimates
2. Train multiple models for ensembling
3. Reduce variance in evaluation

Usage:
    python train_kfold.py -c 1 -o -e 100 --n_folds 5
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import json
import numpy as np
from datetime import datetime

from models.eegnet import create_model
from utils.metrics import compute_all_metrics, normalized_rmse

# Import official dataset
try:
    from data.official_dataset_example import OfficialEEGDataset
    OFFICIAL_AVAILABLE = True
except ImportError:
    OFFICIAL_AVAILABLE = False


def create_kfold_splits(dataset, n_folds=5, random_seed=42):
    """
    Create K-fold splits with subject-wise splitting

    Returns:
        List of (train_indices, val_indices) tuples
    """
    # Get unique subjects
    subject_ids = dataset.eeg_dataset.description.iloc[dataset.valid_indices]['subject'].values
    unique_subjects = np.unique(subject_ids)

    # Shuffle subjects
    np.random.seed(random_seed)
    np.random.shuffle(unique_subjects)

    # Split subjects into K folds
    n_subjects = len(unique_subjects)
    fold_size = n_subjects // n_folds

    folds = []
    for fold_idx in range(n_folds):
        # Determine validation subjects for this fold
        val_start = fold_idx * fold_size
        val_end = val_start + fold_size if fold_idx < n_folds - 1 else n_subjects
        val_subjects = set(unique_subjects[val_start:val_end])
        train_subjects = set(unique_subjects) - val_subjects

        # Create indices based on subjects
        train_indices = [i for i, idx in enumerate(dataset.valid_indices)
                        if dataset.eeg_dataset.description.iloc[idx]['subject'] in train_subjects]
        val_indices = [i for i, idx in enumerate(dataset.valid_indices)
                      if dataset.eeg_dataset.description.iloc[idx]['subject'] in val_subjects]

        folds.append((train_indices, val_indices))

        print(f"Fold {fold_idx + 1}/{n_folds}:")
        print(f"  Train: {len(train_indices)} recordings from {len(train_subjects)} subjects")
        print(f"  Val:   {len(val_indices)} recordings from {len(val_subjects)} subjects")

    return folds


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc="Training")
    for data, target in pbar:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({'loss': loss.item()})

    return total_loss / num_batches


def validate_with_metrics(model, dataloader, criterion, device):
    """Validate and compute comprehensive metrics"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for data, target in tqdm(dataloader, desc="Validation"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            all_predictions.append(output.cpu())
            all_targets.append(target.cpu())

    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    metrics = compute_all_metrics(all_predictions, all_targets)
    avg_loss = total_loss / len(dataloader)

    return avg_loss, metrics, all_predictions, all_targets


def train_fold(fold_idx, train_dataset, val_dataset, args, device):
    """Train a single fold"""
    print(f"\n{'='*70}")
    print(f"Training Fold {fold_idx + 1}/{args.n_folds}")
    print(f"{'='*70}\n")

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Create model
    model = create_model(
        challenge=f'c{args.challenge}',
        device=device,
        dropout=args.dropout
    )

    # Optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    # Training loop
    best_nrmse = float('inf')
    best_metrics = {}
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\nFold {fold_idx + 1}, Epoch {epoch}/{args.epochs}")

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"  Train loss: {train_loss:.4f}")

        # Validate
        val_loss, val_metrics, predictions, targets = validate_with_metrics(
            model, val_loader, criterion, device
        )

        print(f"  Val NRMSE: {val_metrics['nrmse']:.4f} ⭐")
        print(f"  Val Pearson: {val_metrics['pearson_r']:.4f}")
        print(f"  Val R²: {val_metrics['r2']:.4f}")

        # Learning rate scheduling
        scheduler.step(val_metrics['nrmse'])

        # Save best model
        if val_metrics['nrmse'] < best_nrmse:
            improvement = best_nrmse - val_metrics['nrmse']
            best_nrmse = val_metrics['nrmse']
            best_metrics = val_metrics
            best_epoch = epoch

            # Save checkpoint
            checkpoint_dir = Path(args.checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"c{args.challenge}_fold{fold_idx}_best.pth"

            torch.save({
                'fold': fold_idx,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_nrmse': best_nrmse,
                'val_metrics': best_metrics,
            }, checkpoint_path)

            print(f"  ✅ Best model saved! (improved: {improvement:.4f})")

    print(f"\nFold {fold_idx + 1} complete!")
    print(f"  Best val NRMSE: {best_nrmse:.4f} (epoch {best_epoch})")

    return best_nrmse, best_metrics, best_epoch


def main():
    parser = argparse.ArgumentParser(description="K-Fold Cross-Validation Training")

    # Data args
    parser.add_argument('-c', '--challenge', type=int, required=True, choices=[1, 2])
    parser.add_argument('-o', '--use_official', action='store_true',
                        help='Use official EEGChallengeDataset')
    parser.add_argument('-m', '--official_mini', action='store_true',
                        help='Use mini dataset')
    parser.add_argument('--max', '--max_subjects', type=int, default=None,
                        dest='max_subjects', help='Max subjects')

    # K-Fold args
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of folds (default: 5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for fold splitting')

    # Training args
    parser.add_argument('-e', '--epochs', type=int, default=50)
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--drop', '--dropout', type=float, default=0.2,
                        dest='dropout')
    parser.add_argument('-w', '--num_workers', type=int, default=4)

    # Checkpoint args
    parser.add_argument('--ckpt', '--checkpoint_dir', type=str, default='./checkpoints_kfold',
                        dest='checkpoint_dir')

    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load full dataset
    if not OFFICIAL_AVAILABLE:
        print("❌ Official dataset not available. Install: pip install eegdash braindecode")
        return

    print(f"📦 Loading full dataset for K-Fold CV")
    dataset = OfficialEEGDataset(
        task='contrastChangeDetection',
        challenge=f'c{args.challenge}',
        release='all',
        mini=args.official_mini,
        max_subjects=args.max_subjects
    )

    # Create K-fold splits
    print(f"\n📊 Creating {args.n_folds}-fold splits (subject-wise)")
    folds = create_kfold_splits(dataset, n_folds=args.n_folds, random_seed=args.seed)

    # Train each fold
    fold_results = []
    for fold_idx, (train_indices, val_indices) in enumerate(folds):
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)

        best_nrmse, best_metrics, best_epoch = train_fold(
            fold_idx, train_dataset, val_dataset, args, device
        )

        fold_results.append({
            'fold': fold_idx,
            'best_nrmse': best_nrmse,
            'best_metrics': best_metrics,
            'best_epoch': best_epoch,
        })

    # Summary
    print("\n" + "="*70)
    print("K-FOLD CROSS-VALIDATION SUMMARY")
    print("="*70)

    nrmse_scores = [r['best_nrmse'] for r in fold_results]
    mean_nrmse = np.mean(nrmse_scores)
    std_nrmse = np.std(nrmse_scores)

    print(f"\nIndividual Fold Results:")
    for i, result in enumerate(fold_results):
        print(f"  Fold {i+1}: NRMSE = {result['best_nrmse']:.4f} (epoch {result['best_epoch']})")

    print(f"\n📊 Overall Performance:")
    print(f"  Mean NRMSE: {mean_nrmse:.4f} ± {std_nrmse:.4f}")
    print(f"  Min NRMSE:  {min(nrmse_scores):.4f}")
    print(f"  Max NRMSE:  {max(nrmse_scores):.4f}")

    # Save results
    results_dir = Path("results_kfold")
    results_dir.mkdir(exist_ok=True)

    results_path = results_dir / f"c{args.challenge}_kfold_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'n_folds': args.n_folds,
            'challenge': args.challenge,
            'config': vars(args),
            'fold_results': fold_results,
            'summary': {
                'mean_nrmse': mean_nrmse,
                'std_nrmse': std_nrmse,
                'min_nrmse': min(nrmse_scores),
                'max_nrmse': max(nrmse_scores),
            }
        }, f, indent=2)

    print(f"\n💾 Results saved to: {results_path}")
    print("\n✅ K-Fold Cross-Validation Complete!")
    print(f"\n📁 Fold models saved in: {args.checkpoint_dir}/")
    print(f"   Use these models for ensemble submission!")
    print("="*70)


if __name__ == "__main__":
    main()
