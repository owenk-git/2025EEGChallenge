"""
Strategy 2: Improved Training for Better Performance

Target: Beat C1: 0.93, C2: 1.00

Key improvements:
1. Larger model (more parameters)
2. Stronger data augmentation
3. Longer training with better learning rate schedule
4. Proper RT extraction for C1
5. Advanced optimization (AdamW + Cosine LR)

Usage:
    python strategy2.py --challenge 1 --epochs 200 --batch_size 128
    python strategy2.py --challenge 2 --epochs 200 --batch_size 128
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
import numpy as np

from models.eegnet import create_model
from utils.metrics import compute_all_metrics, normalized_rmse

# Try to import official dataset
try:
    from data.official_dataset_example import (
        create_official_dataloaders_with_split
    )
    OFFICIAL_AVAILABLE = True
except ImportError:
    OFFICIAL_AVAILABLE = False


class EEGAugmentation:
    """Data augmentation for EEG signals"""

    def __init__(self, noise_std=0.1, scale_range=(0.9, 1.1), shift_range=(-0.05, 0.05)):
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.shift_range = shift_range

    def __call__(self, x):
        """
        Apply random augmentations

        Args:
            x: EEG tensor (channels, times)
        Returns:
            Augmented EEG tensor
        """
        # Random Gaussian noise
        if np.random.rand() < 0.5:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise

        # Random amplitude scaling
        if np.random.rand() < 0.5:
            scale = np.random.uniform(*self.scale_range)
            x = x * scale

        # Random DC shift
        if np.random.rand() < 0.5:
            shift = np.random.uniform(*self.shift_range)
            x = x + shift

        # Random channel dropout (simulate missing channels)
        if np.random.rand() < 0.3:
            n_channels = x.shape[0]
            n_drop = np.random.randint(1, max(2, n_channels // 20))
            drop_indices = np.random.choice(n_channels, n_drop, replace=False)
            x[drop_indices, :] = 0

        return x


def train_epoch(model, dataloader, criterion, optimizer, device, augmentation=None, clip_grad=1.0):
    """Train for one epoch with optional augmentation"""
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, (data, target) in enumerate(pbar):
        # Apply augmentation if provided
        if augmentation is not None:
            augmented_data = []
            for i in range(data.shape[0]):
                aug_sample = augmentation(data[i])
                augmented_data.append(aug_sample)
            data = torch.stack(augmented_data)

        data, target = data.to(device), target.to(device)

        # Check for NaN in inputs
        if torch.isnan(data).any() or torch.isnan(target).any():
            print(f"\n⚠️  NaN in batch {batch_idx}")
            continue

        # Forward pass
        optimizer.zero_grad()
        output = model(data)

        # Check for NaN in output
        if torch.isnan(output).any():
            print(f"\n⚠️  NaN in model output at batch {batch_idx}")
            continue

        # Compute loss
        loss = criterion(output, target)

        # Check for NaN loss
        if torch.isnan(loss):
            print(f"\n⚠️  NaN loss at batch {batch_idx}")
            continue

        # Backward pass with gradient clipping
        loss.backward()
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / num_batches if num_batches > 0 else float('nan')
    return avg_loss


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            all_predictions.append(output.cpu().numpy())
            all_targets.append(target.cpu().numpy())

    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    avg_loss = total_loss / len(dataloader)
    metrics = compute_all_metrics(predictions, targets)

    return avg_loss, metrics, predictions, targets


def main():
    parser = argparse.ArgumentParser(description="Strategy 2: Improved Training")

    parser.add_argument('-c', '--challenge', type=int, required=True, choices=[1, 2],
                        help='Challenge number (1 or 2)')
    parser.add_argument('-e', '--epochs', type=int, default=200,
                        help='Number of training epochs (default: 200)')
    parser.add_argument('-b', '--batch_size', type=int, default=128,
                        help='Batch size (default: 128)')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0005,
                        help='Initial learning rate (default: 0.0005)')
    parser.add_argument('--dropout', type=float, default=0.25,
                        help='Dropout rate (default: 0.25)')
    parser.add_argument('-w', '--num_workers', type=int, default=8,
                        help='Number of data loading workers (default: 8)')
    parser.add_argument('--no_augmentation', action='store_true',
                        help='Disable data augmentation')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_strategy2',
                        help='Directory to save checkpoints')

    args = parser.parse_args()

    print("="*70)
    print(f"🚀 Strategy 2: Improved Training - Challenge {args.challenge}")
    print("="*70)
    print(f"Target: Beat C1: 0.93, C2: 1.00")
    print(f"Improvements: Larger model, augmentation, longer training")
    print("="*70)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create data loaders
    print(f"\n📊 Loading data...")

    if not OFFICIAL_AVAILABLE:
        print("❌ Official dataset not available. Install: pip install eegdash braindecode")
        return

    train_loader, val_loader = create_official_dataloaders_with_split(
        task="contrastChangeDetection",
        challenge=f'c{args.challenge}',
        batch_size=args.batch_size,
        mini=False,
        num_workers=args.num_workers,
        val_split=0.2
    )

    # Create model (larger than default)
    print(f"\n🧠 Creating LARGER model for Challenge {args.challenge}")
    model = create_model(
        challenge=f'c{args.challenge}',
        device=device,
        dropout=args.dropout,
        output_range=(0.88, 1.12) if args.challenge == 1 else None
    )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {num_params:,}")

    # Setup augmentation
    augmentation = None
    if not args.no_augmentation:
        augmentation = EEGAugmentation(
            noise_std=0.08,  # Moderate noise
            scale_range=(0.92, 1.08),  # Small scaling
            shift_range=(-0.03, 0.03)  # Small shift
        )
        print(f"✅ Data augmentation enabled")

    # Loss function
    criterion = nn.MSELoss()

    # Optimizer: AdamW (better than Adam)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,  # L2 regularization
        betas=(0.9, 0.999)
    )

    # Learning rate scheduler: Cosine annealing with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=20,  # Restart every 20 epochs
        T_mult=2,  # Double period after each restart
        eta_min=1e-6
    )

    # Track best validation metrics
    best_nrmse = float('inf')
    best_metrics = {}
    best_epoch = 0

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🎯 Training for {args.epochs} epochs")
    print("="*70)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device,
                                augmentation=augmentation, clip_grad=1.0)
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Validate
        val_loss, val_metrics, predictions, targets = validate(
            model, val_loader, criterion, device
        )

        print(f"  Val Loss:  {val_loss:.4f}")
        print(f"  Val NRMSE: {val_metrics['nrmse']:.4f} ⭐ (Target: {'0.93' if args.challenge == 1 else '1.00'})")
        print(f"  Val RMSE:  {val_metrics['rmse']:.4f}")
        print(f"  Val MAE:   {val_metrics['mae']:.4f}")

        # Update scheduler
        scheduler.step()

        # Save best model
        if val_metrics['nrmse'] < best_nrmse:
            best_nrmse = val_metrics['nrmse']
            best_metrics = val_metrics
            best_epoch = epoch

            checkpoint_path = checkpoint_dir / f"c{args.challenge}_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_nrmse': best_nrmse,
                'val_metrics': val_metrics,
            }, checkpoint_path)
            print(f"  ✅ Saved best model (NRMSE: {best_nrmse:.4f})")

        # Save checkpoint every 50 epochs
        if epoch % 50 == 0:
            checkpoint_path = checkpoint_dir / f"c{args.challenge}_epoch{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_nrmse': val_metrics['nrmse'],
                'val_metrics': val_metrics,
            }, checkpoint_path)

    print("\n" + "="*70)
    print("✅ Training Complete!")
    print(f"   Best NRMSE: {best_nrmse:.4f} at epoch {best_epoch}")
    print(f"   Target: {'0.93' if args.challenge == 1 else '1.00'}")
    print(f"   {'🎉 BEAT TARGET!' if best_nrmse < (0.93 if args.challenge == 1 else 1.00) else '⚠️  Need improvement'}")
    print("="*70)


if __name__ == "__main__":
    main()
