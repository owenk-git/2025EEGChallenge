#!/usr/bin/env python3
"""
Train Trial-Level RT Predictor with Diversity Loss

Adds regularization to encourage wider prediction range
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import datetime
import json
from tqdm import tqdm

from data.trial_level_loader import TrialLevelDataset
from models.trial_level_rt_predictor import TrialLevelRTPredictor


def compute_nrmse(predictions, targets):
    """Compute Normalized RMSE"""
    mse = ((predictions - targets) ** 2).mean()
    rmse = torch.sqrt(mse)
    nrmse = rmse / targets.std()
    return nrmse.item()


def train_with_diversity(
    challenge='c1',
    epochs=100,
    batch_size=32,
    lr=0.0001,
    lambda_diversity=0.1,
    mini=False,
    device='cuda'
):
    """
    Train trial-level model with diversity loss

    Args:
        challenge: 'c1' or 'c2'
        epochs: Number of epochs
        batch_size: Batch size
        lr: Learning rate
        lambda_diversity: Weight for diversity loss
        mini: Use mini dataset
        device: 'cuda' or 'cpu'
    """

    print("="*80)
    print(f"TRAINING TRIAL-LEVEL RT PREDICTOR WITH DIVERSITY LOSS")
    print("="*80)
    print(f"\nChallenge: {challenge}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Lambda diversity: {lambda_diversity}")
    print(f"Mini: {mini}")
    print(f"Device: {device}\n")

    # Load data
    print("Loading data...")
    full_dataset = TrialLevelDataset(challenge=challenge, mini=mini)

    # Split into train/val (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"✅ Train: {len(train_dataset)} trials")
    print(f"✅ Val: {len(val_dataset)} trials\n")

    # Create model
    model = TrialLevelRTPredictor(
        n_channels=129,
        trial_length=200,
        pre_stim_points=50
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Training loop
    best_nrmse = float('inf')
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)

    debug_log = {
        'config': {
            'challenge': challenge,
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr,
            'lambda_diversity': lambda_diversity,
            'mini': mini
        },
        'epochs': []
    }

    print("="*80)
    print("TRAINING")
    print("="*80)

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0
        train_task_loss = 0
        train_div_loss = 0
        all_train_preds = []
        all_train_targets = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for X_batch, y_batch in pbar:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            # Forward
            predictions = model(X_batch).squeeze()

            # Task loss (MSE)
            task_loss = nn.MSELoss()(predictions, y_batch)

            # Diversity loss (encourage variance)
            pred_mean = predictions.mean()
            pred_variance = ((predictions - pred_mean) ** 2).mean()
            diversity_loss = -torch.log(pred_variance + 1e-6)  # Maximize variance

            # Total loss
            total_loss = task_loss + lambda_diversity * diversity_loss

            # Backward
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += total_loss.item()
            train_task_loss += task_loss.item()
            train_div_loss += diversity_loss.item()

            all_train_preds.extend(predictions.detach().cpu().numpy())
            all_train_targets.extend(y_batch.detach().cpu().numpy())

            pbar.set_postfix({'loss': f'{total_loss.item():.4f}'})

        train_loss /= len(train_loader)
        train_task_loss /= len(train_loader)
        train_div_loss /= len(train_loader)

        all_train_preds = np.array(all_train_preds)
        all_train_targets = np.array(all_train_targets)

        train_nrmse = np.sqrt(np.mean((all_train_preds - all_train_targets)**2)) / np.std(all_train_targets)
        train_corr = np.corrcoef(all_train_preds, all_train_targets)[0, 1]

        # Validation
        model.eval()
        val_loss = 0
        all_val_preds = []
        all_val_targets = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                predictions = model(X_batch).squeeze()
                loss = nn.MSELoss()(predictions, y_batch)

                val_loss += loss.item()
                all_val_preds.extend(predictions.cpu().numpy())
                all_val_targets.extend(y_batch.cpu().numpy())

        val_loss /= len(val_loader)

        all_val_preds = np.array(all_val_preds)
        all_val_targets = np.array(all_val_targets)

        val_nrmse = np.sqrt(np.mean((all_val_preds - all_val_targets)**2)) / np.std(all_val_targets)
        val_corr = np.corrcoef(all_val_preds, all_val_targets)[0, 1]

        # Scheduler
        scheduler.step(val_nrmse)

        # Print progress
        print(f"\nEpoch {epoch}/{epochs}")
        print(f"  Train: Loss={train_loss:.4f}, Task={train_task_loss:.4f}, Div={train_div_loss:.4f}, NRMSE={train_nrmse:.4f}, Corr={train_corr:.3f}")
        print(f"  Val:   Loss={val_loss:.4f}, NRMSE={val_nrmse:.4f}, Corr={val_corr:.3f}")
        print(f"  📊 Train pred range: [{all_train_preds.min():.3f}, {all_train_preds.max():.3f}], std={all_train_preds.std():.3f}")
        print(f"  📊 Val pred range: [{all_val_preds.min():.3f}, {all_val_preds.max():.3f}], std={all_val_preds.std():.3f}")

        # Log epoch
        debug_log['epochs'].append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_task_loss': train_task_loss,
            'train_diversity_loss': train_div_loss,
            'train_nrmse': float(train_nrmse),
            'train_corr': float(train_corr),
            'train_pred_std': float(all_train_preds.std()),
            'val_loss': val_loss,
            'val_nrmse': float(val_nrmse),
            'val_corr': float(val_corr),
            'val_pred_std': float(all_val_preds.std())
        })

        # Save best model
        if val_nrmse < best_nrmse:
            best_nrmse = val_nrmse
            checkpoint_path = checkpoint_dir / f'trial_level_{challenge}_diversity_best.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_nrmse': best_nrmse,
                'lambda_diversity': lambda_diversity
            }, checkpoint_path)
            print(f"  ✓ Saved best model (NRMSE: {best_nrmse:.4f})")

    # Save debug log
    log_dir = Path('debug_logs')
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f'{challenge}_diversity_training_{timestamp}.json'

    with open(log_file, 'w') as f:
        json.dump(debug_log, f, indent=2)

    print(f"\n💾 Debug log saved: {log_file}")

    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Best Val NRMSE: {best_nrmse:.4f}")
    print(f"Model saved: {checkpoint_path}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Train trial-level model with diversity loss"
    )

    parser.add_argument('--challenge', type=str, default='c1', help='Challenge (c1/c2)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--lambda_diversity', type=float, default=0.1, help='Diversity loss weight')
    parser.add_argument('--mini', action='store_true', help='Use mini dataset')
    parser.add_argument('--device', type=str, default='cuda', help='Device')

    args = parser.parse_args()

    train_with_diversity(
        challenge=args.challenge,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_diversity=args.lambda_diversity,
        mini=args.mini,
        device=args.device
    )


if __name__ == "__main__":
    main()
