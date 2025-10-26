"""
Training script for Cross-Task Pre-Training Model

Two-stage training:
1. Pre-train on all available tasks (multi-task learning)
2. Fine-tune on target task (CCD for challenge)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import argparse
from pathlib import Path

from models.cross_task_pretrain import (
    create_cross_task_model,
    CrossTaskLoss
)


def load_single_task_data(challenge='c1'):
    """Load data for single task (CCD)"""
    data_dir = Path('data/preprocessed')

    X_train = np.load(data_dir / f'{challenge}_X_train.npy')
    y_train = np.load(data_dir / f'{challenge}_y_train.npy')
    X_val = np.load(data_dir / f'{challenge}_X_val.npy')
    y_val = np.load(data_dir / f'{challenge}_y_val.npy')

    return X_train, y_train, X_val, y_val


def normalize_targets(y_train, y_val):
    """Normalize targets to mean=0, std=1"""
    mean = y_train.mean()
    std = y_train.std()

    y_train_norm = (y_train - mean) / std
    y_val_norm = (y_val - mean) / std

    return y_train_norm, y_val_norm, mean, std


def compute_nrmse(predictions, targets, std):
    """Compute NRMSE"""
    mse = ((predictions - targets) ** 2).mean()
    rmse = np.sqrt(mse)
    nrmse = rmse / std
    return nrmse


def train_cross_task(challenge='c1', epochs=100, batch_size=64, lr=1e-3,
                     device='cuda', save_dir='checkpoints', pretrain_epochs=50):
    """
    Train Cross-Task Pre-Training model

    Strategy:
    1. If pretrain_epochs > 0: Pre-train on multi-task data (if available)
    2. Fine-tune on target task (CCD)

    Args:
        challenge: 'c1' or 'c2'
        epochs: Number of fine-tuning epochs
        batch_size: Batch size
        lr: Learning rate
        device: 'cuda' or 'cpu'
        save_dir: Directory to save checkpoints
        pretrain_epochs: Number of pre-training epochs (0 to skip pre-training)
    """
    print(f"Training Cross-Task Pre-Training model for {challenge}")

    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    X_train, y_train, X_val, y_val = load_single_task_data(challenge)

    # Normalize targets
    y_train_norm, y_val_norm, y_mean, y_std = normalize_targets(y_train, y_val)

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train_norm)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val_norm)

    # Create datasets
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    print("Creating model...")
    model = create_cross_task_model(challenge=challenge, device=device)

    # Target task name (CCD)
    target_task = 'contrast_change_detection'

    # ===== Stage 1: Pre-training (optional) =====
    # Note: In real implementation, this would use data from all 6 tasks
    # For now, we simulate by training on CCD data with multi-task objective
    if pretrain_epochs > 0:
        print(f"\nStage 1: Pre-training for {pretrain_epochs} epochs...")
        print("(Note: Using CCD data as proxy for multi-task pre-training)")

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=pretrain_epochs)

        for epoch in range(pretrain_epochs):
            model.train()
            train_losses = []

            for X_batch, y_batch in train_loader:
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

            # Denormalize
            val_predictions_denorm = val_predictions * y_std + y_mean
            val_nrmse = compute_nrmse(val_predictions_denorm, y_val, y_std)

            avg_train_loss = np.mean(train_losses)
            print(f"Pretrain Epoch {epoch+1}/{pretrain_epochs} | Train Loss: {avg_train_loss:.4f} | Val NRMSE: {val_nrmse:.4f}")

        print(f"Pre-training completed!")

        # Save pre-trained model
        pretrain_path = save_dir / f'cross_task_{challenge}_pretrained.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'y_mean': y_mean,
            'y_std': y_std,
            'challenge': challenge
        }, pretrain_path)
        print(f"Saved pre-trained model to {pretrain_path}")

    # ===== Stage 2: Fine-tuning =====
    print(f"\nStage 2: Fine-tuning on {target_task} for {epochs} epochs...")

    # Option 1: Fine-tune all parameters
    # Option 2: Freeze feature extractor and only fine-tune task head
    # We'll use Option 1 (fine-tune all) with lower learning rate

    fine_tune_lr = lr * 0.1  # Lower learning rate for fine-tuning

    optimizer = optim.AdamW(model.parameters(), lr=fine_tune_lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_nrmse = float('inf')

    for epoch in range(epochs):
        model.train()
        train_losses = []

        for X_batch, y_batch in train_loader:
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

        # Denormalize
        val_predictions_denorm = val_predictions * y_std + y_mean
        val_nrmse = compute_nrmse(val_predictions_denorm, y_val, y_std)

        avg_train_loss = np.mean(train_losses)
        print(f"Finetune Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val NRMSE: {val_nrmse:.4f}")

        # Save best model
        if val_nrmse < best_val_nrmse:
            best_val_nrmse = val_nrmse
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_nrmse': val_nrmse,
                'y_mean': y_mean,
                'y_std': y_std,
                'challenge': challenge,
                'target_task': target_task
            }
            torch.save(checkpoint, save_dir / f'cross_task_{challenge}_best.pt')
            print(f"  ✓ Saved best model (NRMSE: {val_nrmse:.4f})")

    print(f"\nFine-tuning completed! Best validation NRMSE: {best_val_nrmse:.4f}")
    return best_val_nrmse


def main():
    parser = argparse.ArgumentParser(description='Train Cross-Task Pre-Training Model')
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

    args = parser.parse_args()

    train_cross_task(
        challenge=args.challenge,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        save_dir=args.save_dir,
        pretrain_epochs=args.pretrain_epochs
    )


if __name__ == '__main__':
    main()
