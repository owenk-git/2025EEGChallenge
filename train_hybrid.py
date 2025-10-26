"""
Training script for Hybrid CNN-Transformer with Domain Adaptation
"""

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import argparse
from pathlib import Path

from models.hybrid_cnn_transformer_da import (
    create_hybrid_cnn_transformer_da,
    HybridLoss
)


def load_data(challenge='c1'):
    """Load preprocessed data"""
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


def train_hybrid(challenge='c1', epochs=100, batch_size=64, lr=1e-3,
                lambda_mmd=0.1, lambda_entropy=0.01,
                device='cuda', save_dir='checkpoints'):
    """
    Train Hybrid CNN-Transformer-DA model

    Args:
        challenge: 'c1' or 'c2'
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        lambda_mmd: Weight for MMD loss
        lambda_entropy: Weight for entropy loss
        device: 'cuda' or 'cpu'
        save_dir: Directory to save checkpoints
    """
    print(f"Training Hybrid CNN-Transformer-DA model for {challenge}")
    print(f"MMD λ={lambda_mmd}, Entropy λ={lambda_entropy}")

    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    X_train, y_train, X_val, y_val = load_data(challenge)

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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    print("Creating model...")
    model = create_hybrid_cnn_transformer_da(challenge=challenge, device=device)

    # Loss function
    criterion = HybridLoss(lambda_mmd=lambda_mmd, lambda_entropy=lambda_entropy)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    best_val_nrmse = float('inf')

    for epoch in range(epochs):
        model.train()

        train_losses = []
        train_loss_details = {
            'task_loss': [],
            'mmd_loss': [],
            'entropy_loss': []
        }

        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Split batch for MMD
            split_idx = X_batch.size(0) // 2

            # Forward pass
            predictions, features = model(X_batch, return_features=True)

            # Split features for MMD
            source_features = features[:split_idx]
            target_features = features[split_idx:]

            # Compute loss
            total_loss, loss_dict = criterion(
                predictions, y_batch,
                source_features=source_features,
                target_features=target_features
            )

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Record losses
            train_losses.append(loss_dict['total_loss'])
            for key in train_loss_details:
                if key in loss_dict:
                    train_loss_details[key].append(loss_dict[key])

        scheduler.step()

        # Validation
        model.eval()
        val_predictions = []
        val_targets = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                predictions = model(X_batch)
                val_predictions.append(predictions.cpu().numpy())
                val_targets.append(y_batch.cpu().numpy())

        val_predictions = np.concatenate(val_predictions)
        val_targets = np.concatenate(val_targets)

        # Denormalize
        val_predictions_denorm = val_predictions * y_std + y_mean
        val_nrmse = compute_nrmse(val_predictions_denorm, y_val, y_std)

        # Print progress
        avg_train_loss = np.mean(train_losses)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val NRMSE: {val_nrmse:.4f}")

        # Print detailed losses
        detail_str = " | ".join([f"{k}: {np.mean(v):.4f}" for k, v in train_loss_details.items() if v])
        print(f"  Details: {detail_str}")

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
                'challenge': challenge
            }
            torch.save(checkpoint, save_dir / f'hybrid_{challenge}_best.pt')
            print(f"  ✓ Saved best model (NRMSE: {val_nrmse:.4f})")

    print(f"\nTraining completed! Best validation NRMSE: {best_val_nrmse:.4f}")
    return best_val_nrmse


def main():
    parser = argparse.ArgumentParser(description='Train Hybrid CNN-Transformer-DA')
    parser.add_argument('--challenge', type=str, default='c1', choices=['c1', 'c2'],
                       help='Challenge name')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--lambda_mmd', type=float, default=0.1,
                       help='Weight for MMD loss')
    parser.add_argument('--lambda_entropy', type=float, default=0.01,
                       help='Weight for entropy loss')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')

    args = parser.parse_args()

    train_hybrid(
        challenge=args.challenge,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_mmd=args.lambda_mmd,
        lambda_entropy=args.lambda_entropy,
        device=args.device,
        save_dir=args.save_dir
    )


if __name__ == '__main__':
    main()
