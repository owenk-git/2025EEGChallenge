"""
Training script for Hybrid CNN-Transformer with Domain Adaptation
DIRECT DATA LOADING - No preprocessing needed!

Uses official eegdash loader directly (loads from cache/online)
"""

import os
import torch
import torch.optim as optim
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

from models.hybrid_cnn_transformer_da import (
    create_hybrid_cnn_transformer_da,
    HybridLoss
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


def train_hybrid_direct(challenge='c1', epochs=100, batch_size=64, lr=1e-3,
                       lambda_mmd=0.1, lambda_entropy=0.01,
                       device='cuda', save_dir='checkpoints', mini=False):
    """
    Train Hybrid CNN-Transformer-DA model with direct data loading

    Args:
        challenge: 'c1' or 'c2'
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        lambda_mmd: Weight for MMD loss
        lambda_entropy: Weight for entropy loss
        device: 'cuda' or 'cpu'
        save_dir: Directory to save checkpoints
        mini: Use mini dataset for testing
    """
    print(f"Training Hybrid CNN-Transformer-DA model for {challenge}")
    print(f"MMD λ={lambda_mmd}, Entropy λ={lambda_entropy}")

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
    model = create_hybrid_cnn_transformer_da(challenge=challenge, device=device)

    # Loss function
    criterion = HybridLoss(lambda_mmd=lambda_mmd, lambda_entropy=lambda_entropy)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Calculate target std for NRMSE
    print("\nCalculating target std for NRMSE...")
    all_val_targets = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            all_val_targets.append(y_batch.cpu().numpy())
    all_val_targets = np.concatenate(all_val_targets)
    target_std = np.std(all_val_targets)
    print(f"Target std: {target_std:.4f}")

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

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (X_batch, y_batch) in enumerate(pbar):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Skip if batch too small
            if X_batch.size(0) < 2:
                continue

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

            pbar.set_postfix({'loss': f"{loss_dict['total_loss']:.4f}"})

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

        # Compute NRMSE
        val_nrmse = compute_nrmse(val_predictions, val_targets)

        # Print progress
        avg_train_loss = np.mean(train_losses)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val NRMSE: {val_nrmse:.4f}")

        # Print detailed losses
        detail_str = " | ".join([f"{k}: {np.mean(v):.4f}" for k, v in train_loss_details.items() if v])
        if detail_str:
            print(f"  Details: {detail_str}")

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
                'challenge': challenge
            }
            torch.save(checkpoint, save_dir / f'hybrid_{challenge}_best.pt')
            print(f"  ✓ Saved best model (NRMSE: {val_nrmse:.4f})")

    print(f"\nTraining completed! Best validation NRMSE: {best_val_nrmse:.4f}")
    return best_val_nrmse


def main():
    parser = argparse.ArgumentParser(description='Train Hybrid CNN-Transformer-DA (Direct Loading)')
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
    parser.add_argument('--mini', action='store_true',
                       help='Use mini dataset for testing')

    args = parser.parse_args()

    train_hybrid_direct(
        challenge=args.challenge,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_mmd=args.lambda_mmd,
        lambda_entropy=args.lambda_entropy,
        device=args.device,
        save_dir=args.save_dir,
        mini=args.mini
    )


if __name__ == '__main__':
    main()
