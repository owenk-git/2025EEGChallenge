#!/usr/bin/env python3
"""
K-Fold Cross-Validation with Correlation Loss

Combines:
- Recording-level K-fold (proper validation)
- Correlation-maximizing loss (fix low correlation)
- Variance preservation (fix collapse)
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


class ImprovedLoss(nn.Module):
    """
    Combined loss to fix all C1 problems:
    - MSE (accuracy)
    - Correlation (predictions must track targets)
    - Variance preservation (prevent collapse)
    """
    def __init__(self, lambda_corr=2.0, lambda_std=1.0):
        super().__init__()
        self.lambda_corr = lambda_corr
        self.lambda_std = lambda_std

    def forward(self, predictions, targets):
        # MSE loss
        mse_loss = nn.MSELoss()(predictions, targets)

        # Pearson correlation
        pred_mean = predictions.mean()
        target_mean = targets.mean()
        pred_centered = predictions - pred_mean
        target_centered = targets - target_mean
        numerator = (pred_centered * target_centered).sum()
        denominator = torch.sqrt((pred_centered ** 2).sum() * (target_centered ** 2).sum())
        correlation = numerator / (denominator + 1e-8)
        corr_loss = 1.0 - correlation

        # Variance preservation
        pred_std = predictions.std()
        target_std = targets.std()
        std_loss = (pred_std - target_std) ** 2

        # Combined
        total_loss = mse_loss + self.lambda_corr * corr_loss + self.lambda_std * std_loss

        return total_loss, {
            'mse': mse_loss.item(),
            'correlation': correlation.item(),
            'pred_std': pred_std.item(),
            'target_std': target_std.item()
        }


class RecordingLevelKFold:
    """K-Fold with recording-level splits"""
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
        np.random.seed(seed)
        np.random.shuffle(self.recording_indices)

        print(f"\n📊 K-Fold Setup:")
        print(f"   Total recordings: {len(self.recording_indices)}")
        print(f"   Total trials: {len(dataset.trials)}")
        print(f"   K-folds: {n_splits}")

    def split(self):
        fold_size = len(self.recording_indices) // self.n_splits

        for fold in range(self.n_splits):
            val_start = fold * fold_size
            val_end = (fold + 1) * fold_size if fold < self.n_splits - 1 else len(self.recording_indices)
            val_recordings = self.recording_indices[val_start:val_end]
            train_recordings = [r for r in self.recording_indices if r not in val_recordings]

            train_indices = []
            for rec_idx in train_recordings:
                train_indices.extend(self.recording_to_trials[rec_idx])

            val_indices = []
            for rec_idx in val_recordings:
                val_indices.extend(self.recording_to_trials[rec_idx])

            yield train_indices, val_indices


def train_fold(model, train_loader, val_loader, criterion, optimizer, scheduler,
               device, epochs, patience=10):
    """Train one fold"""
    best_val_nrmse = float('inf')
    best_val_corr = -1
    patience_counter = 0

    for epoch in range(epochs):
        # Train
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).squeeze()

            optimizer.zero_grad()
            predictions = model(X_batch).squeeze()
            loss, _ = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device).squeeze()
                predictions = model(X_batch).squeeze()
                all_preds.extend(predictions.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
        nrmse = rmse / np.std(all_targets)
        corr = np.corrcoef(all_preds, all_targets)[0, 1] if len(all_preds) > 1 else 0.0

        scheduler.step(nrmse)

        if nrmse < best_val_nrmse:
            best_val_nrmse = nrmse
            best_val_corr = corr
            patience_counter = 0
            best_state = model.state_dict()
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            pred_std = all_preds.std()
            target_std = all_targets.std()
            print(f"      Epoch {epoch+1}: Val NRMSE={nrmse:.4f}, Corr={corr:.3f}, Pred std={pred_std:.3f} (best={best_val_nrmse:.4f})")

        if patience_counter >= patience:
            print(f"      Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    return best_val_nrmse, best_val_corr, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lambda_corr', type=float, default=2.0)
    parser.add_argument('--lambda_std', type=float, default=1.0)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    print("\n📦 Loading dataset...")
    full_dataset = TrialLevelDataset(challenge='c1', mini=False)

    # K-Fold splitter
    kfold = RecordingLevelKFold(full_dataset, n_splits=args.n_folds)

    fold_results = []
    all_fold_models = []

    print(f"\n{'='*70}")
    print(f"K-Fold with Correlation Loss + Variance Preservation")
    print(f"Lambda Corr: {args.lambda_corr}, Lambda Std: {args.lambda_std}")
    print(f"{'='*70}\n")

    for fold_idx, (train_indices, val_indices) in enumerate(kfold.split()):
        print(f"\n{'='*70}")
        print(f"FOLD {fold_idx + 1}/{args.n_folds}")
        print(f"{'='*70}")

        train_subset = Subset(full_dataset, train_indices)
        val_subset = Subset(full_dataset, val_indices)

        train_loader = DataLoader(train_subset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size,
                               shuffle=False, num_workers=4, pin_memory=True)

        model = TrialLevelRTPredictor(
            n_channels=129,
            trial_length=200,
            pre_stim_points=50
        ).to(device)

        criterion = ImprovedLoss(lambda_corr=args.lambda_corr, lambda_std=args.lambda_std)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        best_val_nrmse, best_val_corr, trained_model = train_fold(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            device, args.epochs, args.patience
        )

        print(f"\n   ✅ Fold {fold_idx + 1}: Val NRMSE={best_val_nrmse:.4f}, Corr={best_val_corr:.3f}")

        fold_results.append({
            'fold': fold_idx + 1,
            'val_nrmse': best_val_nrmse,
            'val_corr': best_val_corr
        })

        all_fold_models.append(trained_model.state_dict())

    # Summary
    print(f"\n{'='*70}")
    print(f"K-FOLD RESULTS")
    print(f"{'='*70}\n")

    fold_nrmses = [r['val_nrmse'] for r in fold_results]
    fold_corrs = [r['val_corr'] for r in fold_results]

    mean_nrmse = np.mean(fold_nrmses)
    std_nrmse = np.std(fold_nrmses)
    mean_corr = np.mean(fold_corrs)

    print("Per-fold results:")
    for result in fold_results:
        print(f"   Fold {result['fold']}: NRMSE={result['val_nrmse']:.4f}, Corr={result['val_corr']:.3f}")

    print(f"\n📊 Overall:")
    print(f"   Mean Val NRMSE: {mean_nrmse:.4f} ± {std_nrmse:.4f}")
    print(f"   Mean Val Corr: {mean_corr:.3f}")
    print(f"   Best fold NRMSE: {min(fold_nrmses):.4f}")

    # Save best fold
    best_fold_idx = np.argmin(fold_nrmses)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)

    best_path = checkpoint_dir / 'c1_kfold_correlation_best.pt'
    torch.save({
        'fold': best_fold_idx + 1,
        'model_state_dict': all_fold_models[best_fold_idx],
        'val_nrmse': fold_nrmses[best_fold_idx],
        'val_corr': fold_corrs[best_fold_idx],
        'mean_cv_nrmse': mean_nrmse,
        'std_cv_nrmse': std_nrmse,
        'mean_cv_corr': mean_corr,
    }, best_path)
    print(f"\n✅ Saved: {best_path}")

    # Save results
    results_path = checkpoint_dir / 'kfold_correlation_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'mean_nrmse': float(mean_nrmse),
            'std_nrmse': float(std_nrmse),
            'mean_corr': float(mean_corr),
            'fold_results': fold_results,
            'lambda_corr': args.lambda_corr,
            'lambda_std': args.lambda_std,
        }, f, indent=2)
    print(f"✅ Saved: {results_path}")

    print(f"\n{'='*70}")
    print(f"✅ TRUE estimate: {mean_nrmse:.4f} ± {std_nrmse:.4f}")
    print(f"   Correlation: {mean_corr:.3f}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
