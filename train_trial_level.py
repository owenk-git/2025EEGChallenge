#!/usr/bin/env python3
"""
TRAIN TRIAL-LEVEL RT PREDICTOR

This is the breakthrough approach that should reduce NRMSE from 1.0-1.5 to 0.7-0.9!

Key innovation: Predict RT per TRIAL, not per RECORDING
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import numpy as np
import argparse
from tqdm import tqdm

from data.trial_level_loader import create_trial_level_loaders
from models.trial_level_rt_predictor import TrialLevelRTPredictor


def compute_nrmse(predictions, targets, target_std):
    """Compute Normalized RMSE"""
    mse = torch.mean((predictions - targets) ** 2)
    rmse = torch.sqrt(mse)
    nrmse = rmse / target_std
    return nrmse.item()


def train_trial_level(
    challenge='c1',
    epochs=100,
    batch_size=32,
    lr=0.001,
    mini=False,
    save_dir='./checkpoints',
    device='cuda'
):
    """
    Train trial-level RT predictor

    Args:
        challenge: 'c1' only (trial-level RT prediction)
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        mini: Use mini dataset for testing
        save_dir: Where to save model checkpoints
        device: 'cuda' or 'cpu'
    """
    print("="*60)
    print("TRIAL-LEVEL RT PREDICTION")
    print("="*60)

    # Check device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Create data loaders
    print("Loading trial-level data...")
    train_loader, val_loader = create_trial_level_loaders(
        challenge=challenge,
        batch_size=batch_size,
        mini=mini
    )

    # Calculate target std for NRMSE
    print("\nCalculating target std for NRMSE...")
    all_targets = []
    for _, y in train_loader:
        all_targets.append(y)
    all_targets = torch.cat(all_targets)
    target_std = all_targets.std().item()
    print(f"Target std: {target_std:.4f}")

    # Create model
    print("\nCreating model...")
    model = TrialLevelRTPredictor(
        n_channels=129,
        trial_length=200,
        pre_stim_points=50
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    criterion = nn.MSELoss()

    # Training loop
    best_nrmse = float('inf')
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("TRAINING")
    print(f"{'='*60}\n")

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for X_batch, y_batch in pbar:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).squeeze(-1)  # (batch,)

            # Forward
            optimizer.zero_grad()
            predictions = model(X_batch)

            # Loss
            loss = criterion(predictions, y_batch)

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Track
            train_loss += loss.item()
            train_preds.append(predictions.detach().cpu())
            train_targets.append(y_batch.detach().cpu())

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Compute train metrics
        train_loss /= len(train_loader)
        train_preds = torch.cat(train_preds)
        train_targets = torch.cat(train_targets)
        train_nrmse = compute_nrmse(train_preds, train_targets, target_std)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device).squeeze(-1)

                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)

                val_loss += loss.item()
                val_preds.append(predictions.cpu())
                val_targets.append(y_batch.cpu())

        val_loss /= len(val_loader)
        val_preds = torch.cat(val_preds)
        val_targets = torch.cat(val_targets)
        val_nrmse = compute_nrmse(val_preds, val_targets, target_std)

        # Print results
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train NRMSE: {train_nrmse:.4f} | Val NRMSE: {val_nrmse:.4f}")

        # Save best model
        if val_nrmse < best_nrmse:
            best_nrmse = val_nrmse
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_nrmse': best_nrmse,
                'target_std': target_std
            }
            torch.save(checkpoint, save_path / f'trial_level_{challenge}_best.pt')
            print(f"  ✓ Saved best model (NRMSE: {best_nrmse:.4f})")

        # Learning rate scheduling
        scheduler.step(val_nrmse)

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Best Val NRMSE: {best_nrmse:.4f}")
    print(f"Saved to: {save_path / f'trial_level_{challenge}_best.pt'}")

    return model, best_nrmse


def main():
    parser = argparse.ArgumentParser(description='Train Trial-Level RT Predictor')
    parser.add_argument('--challenge', type=str, default='c1', choices=['c1'],
                        help='Challenge (c1 only for now)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--mini', action='store_true', help='Use mini dataset for testing')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Save directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("BREAKTHROUGH APPROACH: TRIAL-LEVEL RT PREDICTION")
    print("="*60)
    print("\nWhy this should work:")
    print("  • Current models: 1 prediction per recording → NRMSE 1.0-1.5")
    print("  • This approach: 30 predictions per recording → Expected NRMSE 0.7-0.9")
    print("\nKey innovations:")
    print("  1. Predict RT per TRIAL, not per RECORDING")
    print("  2. Split pre/post stimulus (attention vs ERP/motor)")
    print("  3. Spatial attention (learn important channels)")
    print("  4. Aggregate trial predictions for final result")
    print("="*60 + "\n")

    train_trial_level(
        challenge=args.challenge,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        mini=args.mini,
        save_dir=args.save_dir,
        device=args.device
    )


if __name__ == '__main__':
    main()
