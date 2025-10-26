#!/usr/bin/env python3
"""
C1 Training with Combined Best Approaches

Combines:
- Distribution matching loss
- Mixup augmentation
- Confidence weighting
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

from models.trial_level_rt_predictor import TrialLevelRTPredictor
from data.trial_level_dataset import TrialLevelDataset


class CombinedLoss(nn.Module):
    """
    Combined loss with all improvements:
    - MSE (accuracy)
    - Distribution matching (preserve variance)
    - Confidence weighting (focus on extremes)
    """
    def __init__(self, lambda_mean=0.05, lambda_std=0.1, weight_strength=0.5):
        super().__init__()
        self.lambda_mean = lambda_mean
        self.lambda_std = lambda_std
        self.weight_strength = weight_strength

    def forward(self, predictions, targets):
        # Confidence weights (emphasize extreme RTs)
        target_mean = targets.mean()
        weights = 1.0 + self.weight_strength * torch.abs(targets - target_mean)

        # Weighted MSE
        squared_errors = (predictions - targets) ** 2
        weighted_mse = (weights * squared_errors).mean()

        # Distribution matching
        pred_mean = predictions.mean()
        pred_std = predictions.std()
        target_mean_val = targets.mean()
        target_std = targets.std()

        mean_loss = (pred_mean - target_mean_val) ** 2
        std_loss = (pred_std - target_std) ** 2

        # Combined loss
        total_loss = weighted_mse + self.lambda_mean * mean_loss + self.lambda_std * std_loss

        return total_loss, {
            'weighted_mse': weighted_mse.item(),
            'mean_diff': mean_loss.item(),
            'std_diff': std_loss.item(),
            'pred_std': pred_std.item(),
            'target_std': target_std.item()
        }


def mixup_data(x, y, alpha=0.3):
    """Apply mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss"""
    loss_a, _ = criterion(pred, y_a)
    loss_b, _ = criterion(pred, y_b)
    return lam * loss_a + (1 - lam) * loss_b


def train_epoch(model, train_loader, criterion, optimizer, device, mixup_alpha=0.3):
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []

    pbar = tqdm(train_loader, desc='Training')
    for X_batch, y_batch in pbar:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).squeeze()

        # Apply mixup with probability 0.5
        if np.random.random() > 0.5 and mixup_alpha > 0:
            X_mixed, y_a, y_b, lam = mixup_data(X_batch, y_batch, alpha=mixup_alpha)

            optimizer.zero_grad()
            predictions = model(X_mixed).squeeze()
            loss = mixup_criterion(criterion, predictions, y_a, y_b, lam)
        else:
            optimizer.zero_grad()
            predictions = model(X_batch).squeeze()
            loss, loss_dict = criterion(predictions, y_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # For stats, use original (non-mixed) batch
        with torch.no_grad():
            original_preds = model(X_batch).squeeze()
            all_preds.extend(original_preds.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())

        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Compute NRMSE
    rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
    nrmse = rmse / np.std(all_targets)

    # Compute correlation
    corr = np.corrcoef(all_preds, all_targets)[0, 1]

    return total_loss / len(train_loader), nrmse, corr, all_preds, all_targets


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in tqdm(val_loader, desc='Validation'):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).squeeze()

            predictions = model(X_batch).squeeze()
            loss, _ = criterion(predictions, y_batch)

            total_loss += loss.item()
            all_preds.extend(predictions.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Compute NRMSE
    rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
    nrmse = rmse / np.std(all_targets)

    # Compute correlation
    corr = np.corrcoef(all_preds, all_targets)[0, 1]

    return total_loss / len(val_loader), nrmse, corr, all_preds, all_targets


def main():
    parser = argparse.ArgumentParser(description='Train C1 with Combined Best Approaches')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lambda_mean', type=float, default=0.05)
    parser.add_argument('--lambda_std', type=float, default=0.1)
    parser.add_argument('--weight_strength', type=float, default=0.5)
    parser.add_argument('--mixup_alpha', type=float, default=0.3)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    print("Loading dataset...")
    full_dataset = TrialLevelDataset(challenge='c1', mini=False)

    # Train/val split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=4, pin_memory=True)

    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    # Create model
    print("Creating model...")
    model = TrialLevelRTPredictor(
        n_channels=129,
        trial_length=200,
        pre_stim_points=50
    ).to(device)

    # Create combined loss
    criterion = CombinedLoss(
        lambda_mean=args.lambda_mean,
        lambda_std=args.lambda_std,
        weight_strength=args.weight_strength
    )

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=7
    )

    # Training loop
    best_val_nrmse = float('inf')
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Training C1 with Combined Best Approaches")
    print(f"  - Distribution matching (λ_mean={args.lambda_mean}, λ_std={args.lambda_std})")
    print(f"  - Confidence weighting (strength={args.weight_strength})")
    print(f"  - Mixup augmentation (α={args.mixup_alpha})")
    print(f"{'='*70}\n")

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")

        # Train
        train_loss, train_nrmse, train_corr, train_preds, train_targets = train_epoch(
            model, train_loader, criterion, optimizer, device, mixup_alpha=args.mixup_alpha
        )

        # Validate
        val_loss, val_nrmse, val_corr, val_preds, val_targets = validate(
            model, val_loader, criterion, device
        )

        # Scheduler step
        scheduler.step(val_nrmse)

        # Print statistics
        print(f"\nEpoch {epoch+1} Results:")
        print(f"  Train - Loss: {train_loss:.4f}, NRMSE: {train_nrmse:.4f}, Corr: {train_corr:.4f}")
        print(f"  Train - Pred std: {train_preds.std():.4f}, Target std: {train_targets.std():.4f}")
        print(f"  Train - Pred range: [{train_preds.min():.3f}, {train_preds.max():.3f}]")
        print(f"  Val   - Loss: {val_loss:.4f}, NRMSE: {val_nrmse:.4f}, Corr: {val_corr:.4f}")
        print(f"  Val   - Pred std: {val_preds.std():.4f}, Target std: {val_targets.std():.4f}")
        print(f"  Val   - Pred range: [{val_preds.min():.3f}, {val_preds.max():.3f}]")

        # Save best model
        if val_nrmse < best_val_nrmse:
            best_val_nrmse = val_nrmse
            checkpoint_path = checkpoint_dir / 'c1_combined_best.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_nrmse': val_nrmse,
                'val_corr': val_corr,
                'hyperparameters': {
                    'lambda_mean': args.lambda_mean,
                    'lambda_std': args.lambda_std,
                    'weight_strength': args.weight_strength,
                    'mixup_alpha': args.mixup_alpha,
                }
            }, checkpoint_path)
            print(f"  ✓ Saved best model (NRMSE: {val_nrmse:.4f})")

        print()

    print(f"Training complete! Best Val NRMSE: {best_val_nrmse:.4f}")


if __name__ == '__main__':
    main()
