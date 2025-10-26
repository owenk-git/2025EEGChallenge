"""
Training script for Cross-Task Pre-Training Model
DIRECT DATA LOADING - No preprocessing needed!

Uses official eegdash loader directly (loads from cache/online)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

from models.cross_task_pretrain import (
    create_cross_task_model,
    CrossTaskLoss
)

# Import official loader
try:
    from data.official_eegdash_loader import create_official_eegdash_loaders
    OFFICIAL_LOADER_AVAILABLE = True
except ImportError:
    try:
        from data.official_dataset_example import create_official_dataloaders_with_split
        OFFICIAL_LOADER_AVAILABLE = False
    except ImportError:
        print("❌ No data loader available!")
        exit(1)


def compute_nrmse(predictions, targets):
    """Compute NRMSE"""
    std = np.std(targets)
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    nrmse = rmse / std
    return nrmse


def train_cross_task_direct(challenge='c1', epochs=100, batch_size=64, lr=1e-3,
                           device='cuda', save_dir='checkpoints', pretrain_epochs=50,
                           mini=False):
    """
    Train Cross-Task Pre-Training model with direct data loading

    Args:
        challenge: 'c1' or 'c2'
        epochs: Number of fine-tuning epochs
        batch_size: Batch size
        lr: Learning rate
        device: 'cuda' or 'cpu'
        save_dir: Directory to save checkpoints
        pretrain_epochs: Number of pre-training epochs
        mini: Use mini dataset for testing
    """
    print(f"Training Cross-Task Pre-Training model for {challenge}")
    print(f"Pre-training: {pretrain_epochs} epochs, Fine-tuning: {epochs} epochs")

    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load data DIRECTLY from eegdash
    print("\n📦 Loading data from eegdash (no preprocessing needed)...")

    if OFFICIAL_LOADER_AVAILABLE:
        train_loader, val_loader = create_official_eegdash_loaders(
            challenge=challenge,
            batch_size=batch_size,
            num_workers=4,
            val_split=0.2,
            mini=mini
        )
    else:
        train_loader, val_loader = create_official_dataloaders_with_split(
            challenge=int(challenge[1]),
            batch_size=batch_size,
            val_split=0.2
        )

    print(f"✅ Data loaded: {len(train_loader)} train batches, {len(val_loader)} val batches")

    # Create model
    print("\nCreating model...")
    model = create_cross_task_model(challenge=challenge, device=device)
    target_task = 'contrast_change_detection'

    # Calculate target std for NRMSE
    print("\nCalculating target std for NRMSE...")
    all_val_targets = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            all_val_targets.append(y_batch.cpu().numpy())
    all_val_targets = np.concatenate(all_val_targets)
    target_std = np.std(all_val_targets)
    print(f"Target std: {target_std:.4f}")

    # ===== Stage 1: Pre-training =====
    if pretrain_epochs > 0:
        print(f"\n{'='*60}")
        print(f"Stage 1: Pre-training for {pretrain_epochs} epochs")
        print(f"{'='*60}")

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=pretrain_epochs)

        for epoch in range(pretrain_epochs):
            model.train()
            train_losses = []

            pbar = tqdm(train_loader, desc=f"Pretrain Epoch {epoch+1}/{pretrain_epochs}")
            for X_batch, y_batch in pbar:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                # Forward pass (single task for now)
                predictions = model(X_batch, task_name=target_task)

                # Loss
                loss = F.mse_loss(predictions, y_batch)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_losses.append(loss.item())
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            scheduler.step()

            # Validation
            model.eval()
            val_predictions = []
            val_targets = []

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device)
                    predictions = model(X_batch, task_name=target_task)
                    val_predictions.append(predictions.cpu().numpy())
                    val_targets.append(y_batch.cpu().numpy())

            val_predictions = np.concatenate(val_predictions)
            val_targets = np.concatenate(val_targets)
            val_nrmse = compute_nrmse(val_predictions, val_targets)

            avg_train_loss = np.mean(train_losses)
            print(f"Pretrain Epoch {epoch+1}/{pretrain_epochs} | Loss: {avg_train_loss:.4f} | Val NRMSE: {val_nrmse:.4f}")

        print(f"\nPre-training completed!")

        # Save pre-trained model
        pretrain_path = save_dir / f'cross_task_{challenge}_pretrained.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'challenge': challenge
        }, pretrain_path)
        print(f"Saved pre-trained model to {pretrain_path}")

    # ===== Stage 2: Fine-tuning =====
    print(f"\n{'='*60}")
    print(f"Stage 2: Fine-tuning on {target_task} for {epochs} epochs")
    print(f"{'='*60}")

    fine_tune_lr = lr * 0.1  # Lower learning rate

    optimizer = optim.AdamW(model.parameters(), lr=fine_tune_lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_nrmse = float('inf')

    for epoch in range(epochs):
        model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f"Finetune Epoch {epoch+1}/{epochs}")
        for X_batch, y_batch in pbar:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Forward pass
            predictions = model(X_batch, task_name=target_task)

            # Loss
            loss = F.mse_loss(predictions, y_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        scheduler.step()

        # Validation
        model.eval()
        val_predictions = []
        val_targets = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                predictions = model(X_batch, task_name=target_task)
                val_predictions.append(predictions.cpu().numpy())
                val_targets.append(y_batch.cpu().numpy())

        val_predictions = np.concatenate(val_predictions)
        val_targets = np.concatenate(val_targets)
        val_nrmse = compute_nrmse(val_predictions, val_targets)

        avg_train_loss = np.mean(train_losses)
        print(f"Finetune Epoch {epoch+1}/{epochs} | Loss: {avg_train_loss:.4f} | Val NRMSE: {val_nrmse:.4f}")

        # Save best model
        if val_nrmse < best_val_nrmse:
            best_val_nrmse = val_nrmse
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_nrmse': val_nrmse,
                'y_mean': 0.0,
                'y_std': 1.0,
                'challenge': challenge,
                'target_task': target_task
            }
            torch.save(checkpoint, save_dir / f'cross_task_{challenge}_best.pt')
            print(f"  ✓ Saved best model (NRMSE: {val_nrmse:.4f})")

    print(f"\nFine-tuning completed! Best validation NRMSE: {best_val_nrmse:.4f}")
    return best_val_nrmse


def main():
    parser = argparse.ArgumentParser(description='Train Cross-Task Pre-Training (Direct Loading)')
    parser.add_argument('--challenge', type=str, default='c1', choices=['c1', 'c2'],
                       help='Challenge name')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of fine-tuning epochs')
    parser.add_argument('--pretrain_epochs', type=int, default=50,
                       help='Number of pre-training epochs (0 to skip)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--mini', action='store_true',
                       help='Use mini dataset for testing')

    args = parser.parse_args()

    train_cross_task_direct(
        challenge=args.challenge,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        save_dir=args.save_dir,
        pretrain_epochs=args.pretrain_epochs,
        mini=args.mini
    )


if __name__ == '__main__':
    main()
