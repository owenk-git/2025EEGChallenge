#!/usr/bin/env python3
"""
K-Fold Cross-Validation for C1 with PROPER validation strategy

Uses RECORDING-LEVEL splits to prevent data leakage
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json

from models.trial_level_rt_predictor import TrialLevelRTPredictor
from data.trial_level_loader import TrialLevelDataset


class RecordingLevelKFold:
    """
    K-Fold that splits by RECORDING, not by trial

    This ensures:
    - No trial from same recording appears in both train and val
    - Tests true generalization to new recordings
    """
    def __init__(self, dataset, n_splits=5, seed=42):
        self.dataset = dataset
        self.n_splits = n_splits
        self.seed = seed

        # Group trials by recording
        self.recording_to_trials = {}
        for idx, (_, _, trial_info) in enumerate(dataset.trials):
            rec_idx = trial_info['recording_idx']
            if rec_idx not in self.recording_to_trials:
                self.recording_to_trials[rec_idx] = []
            self.recording_to_trials[rec_idx].append(idx)

        self.recording_indices = list(self.recording_to_trials.keys())

        # Shuffle recordings
        np.random.seed(seed)
        np.random.shuffle(self.recording_indices)

        print(f"\n📊 K-Fold Setup:")
        print(f"   Total recordings: {len(self.recording_indices)}")
        print(f"   Total trials: {len(dataset.trials)}")
        print(f"   Avg trials per recording: {len(dataset.trials) / len(self.recording_indices):.1f}")
        print(f"   K-folds: {n_splits}")

    def split(self):
        """Generate train/val splits"""
        fold_size = len(self.recording_indices) // self.n_splits

        for fold in range(self.n_splits):
            # Val recordings for this fold
            val_start = fold * fold_size
            val_end = (fold + 1) * fold_size if fold < self.n_splits - 1 else len(self.recording_indices)
            val_recordings = self.recording_indices[val_start:val_end]

            # Train recordings (all others)
            train_recordings = [r for r in self.recording_indices if r not in val_recordings]

            # Get trial indices for train and val
            train_indices = []
            for rec_idx in train_recordings:
                train_indices.extend(self.recording_to_trials[rec_idx])

            val_indices = []
            for rec_idx in val_recordings:
                val_indices.extend(self.recording_to_trials[rec_idx])

            print(f"\n   Fold {fold + 1}/{self.n_splits}:")
            print(f"      Train: {len(train_recordings)} recordings, {len(train_indices)} trials")
            print(f"      Val:   {len(val_recordings)} recordings, {len(val_indices)} trials")

            yield train_indices, val_indices


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).squeeze()

        optimizer.zero_grad()
        predictions = model(X_batch).squeeze()
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(predictions.detach().cpu().numpy())
        all_targets.extend(y_batch.detach().cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
    nrmse = rmse / np.std(all_targets)
    corr = np.corrcoef(all_preds, all_targets)[0, 1] if len(all_preds) > 1 else 0.0

    return total_loss / len(train_loader), nrmse, corr, all_preds, all_targets


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).squeeze()

            predictions = model(X_batch).squeeze()
            loss = criterion(predictions, y_batch)

            total_loss += loss.item()
            all_preds.extend(predictions.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
    nrmse = rmse / np.std(all_targets)
    corr = np.corrcoef(all_preds, all_targets)[0, 1] if len(all_preds) > 1 else 0.0

    return total_loss / len(val_loader), nrmse, corr, all_preds, all_targets


def train_fold(model, train_loader, val_loader, criterion, optimizer, scheduler,
               device, epochs, patience=10):
    """Train one fold"""
    best_val_nrmse = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        train_loss, train_nrmse, train_corr, train_preds, train_targets = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_nrmse, val_corr, val_preds, val_targets = validate(
            model, val_loader, criterion, device
        )

        scheduler.step(val_nrmse)

        # Early stopping
        if val_nrmse < best_val_nrmse:
            best_val_nrmse = val_nrmse
            patience_counter = 0
            best_state = model.state_dict()
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"      Epoch {epoch+1}: Train NRMSE={train_nrmse:.4f}, Val NRMSE={val_nrmse:.4f} (best={best_val_nrmse:.4f})")

        if patience_counter >= patience:
            print(f"      Early stopping at epoch {epoch+1}")
            break

    # Load best model
    model.load_state_dict(best_state)

    return best_val_nrmse, model


def main():
    parser = argparse.ArgumentParser(description='K-Fold Cross-Validation for C1')
    parser.add_argument('--n_folds', type=int, default=5,
                       help='Number of folds')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Max epochs per fold')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--save_all_folds', action='store_true',
                       help='Save model from each fold')

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load full dataset
    print("\n📦 Loading dataset...")
    full_dataset = TrialLevelDataset(challenge='c1', mini=False)

    # Create K-Fold splitter (RECORDING-LEVEL)
    kfold = RecordingLevelKFold(full_dataset, n_splits=args.n_folds)

    # Store results
    fold_results = []
    all_fold_models = []

    print(f"\n{'='*70}")
    print(f"K-Fold Cross-Validation with RECORDING-LEVEL splits")
    print(f"{'='*70}\n")

    # Train each fold
    for fold_idx, (train_indices, val_indices) in enumerate(kfold.split()):
        print(f"\n{'='*70}")
        print(f"FOLD {fold_idx + 1}/{args.n_folds}")
        print(f"{'='*70}")

        # Create data loaders
        train_subset = Subset(full_dataset, train_indices)
        val_subset = Subset(full_dataset, val_indices)

        train_loader = DataLoader(train_subset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size,
                               shuffle=False, num_workers=4, pin_memory=True)

        # Create fresh model for this fold
        model = TrialLevelRTPredictor(
            n_channels=129,
            trial_length=200,
            pre_stim_points=50
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        # Train this fold
        best_val_nrmse, trained_model = train_fold(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            device, args.epochs, args.patience
        )

        print(f"\n   ✅ Fold {fold_idx + 1} complete! Best Val NRMSE: {best_val_nrmse:.4f}")

        fold_results.append({
            'fold': fold_idx + 1,
            'val_nrmse': best_val_nrmse,
            'n_train': len(train_indices),
            'n_val': len(val_indices)
        })

        all_fold_models.append(trained_model.state_dict())

        # Save individual fold if requested
        if args.save_all_folds:
            checkpoint_dir = Path(args.checkpoint_dir)
            checkpoint_dir.mkdir(exist_ok=True)
            fold_path = checkpoint_dir / f'c1_kfold_fold{fold_idx+1}.pt'
            torch.save({
                'fold': fold_idx + 1,
                'model_state_dict': trained_model.state_dict(),
                'val_nrmse': best_val_nrmse,
            }, fold_path)
            print(f"   Saved: {fold_path}")

    # Summary
    print(f"\n{'='*70}")
    print(f"K-FOLD CROSS-VALIDATION RESULTS")
    print(f"{'='*70}\n")

    fold_nrmses = [r['val_nrmse'] for r in fold_results]
    mean_nrmse = np.mean(fold_nrmses)
    std_nrmse = np.std(fold_nrmses)

    print("Per-fold results:")
    for result in fold_results:
        print(f"   Fold {result['fold']}: Val NRMSE = {result['val_nrmse']:.4f}")

    print(f"\n📊 Overall Statistics:")
    print(f"   Mean Val NRMSE: {mean_nrmse:.4f} ± {std_nrmse:.4f}")
    print(f"   Best fold: {min(fold_nrmses):.4f}")
    print(f"   Worst fold: {max(fold_nrmses):.4f}")

    # Save best fold model
    best_fold_idx = np.argmin(fold_nrmses)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)

    best_path = checkpoint_dir / 'c1_kfold_best.pt'
    torch.save({
        'fold': best_fold_idx + 1,
        'model_state_dict': all_fold_models[best_fold_idx],
        'val_nrmse': fold_nrmses[best_fold_idx],
        'mean_cv_nrmse': mean_nrmse,
        'std_cv_nrmse': std_nrmse,
        'all_fold_nrmses': fold_nrmses,
    }, best_path)
    print(f"\n✅ Saved best fold model: {best_path}")

    # Save results JSON
    results_path = checkpoint_dir / 'kfold_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'n_folds': args.n_folds,
            'mean_nrmse': float(mean_nrmse),
            'std_nrmse': float(std_nrmse),
            'fold_results': fold_results
        }, f, indent=2)
    print(f"✅ Saved results: {results_path}")

    print(f"\n{'='*70}")
    print(f"✅ K-Fold validation complete!")
    print(f"   TRUE generalization estimate: {mean_nrmse:.4f} ± {std_nrmse:.4f}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
