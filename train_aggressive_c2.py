"""
AGGRESSIVE C2 TRAINING - Target: Drop 1.01 → 0.83

Key strategies:
1. Subject-level aggregation (not trial-level)
2. Huber loss for robustness
3. VERY heavy regularization (personality is stable, shouldn't overfit)
4. Lower learning rate
5. More epochs but with early stopping
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import numpy as np

from models.eegnet import create_model
from utils.metrics import compute_all_metrics

try:
    from data.official_eegdash_loader import create_official_eegdash_loaders
    OFFICIAL_LOADER_AVAILABLE = True
except ImportError:
    OFFICIAL_LOADER_AVAILABLE = False
    from data.official_dataset_example import create_official_dataloaders_with_split


def train_epoch(model, dataloader, criterion, optimizer, device, clip_grad=0.5):
    """Train for one epoch with gradient clipping"""
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)

        if torch.isnan(data).any() or torch.isnan(target).any():
            continue

        optimizer.zero_grad()
        output = model(data)

        if torch.isnan(output).any():
            continue

        loss = criterion(output, target)

        if torch.isnan(loss):
            continue

        loss.backward()
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / num_batches if num_batches > 0 else float('nan')


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
    parser = argparse.ArgumentParser(description="Aggressive C2 training")

    parser.add_argument('-e', '--epochs', type=int, default=100)
    parser.add_argument('-b', '--batch_size', type=int, default=32)  # Smaller batch for C2
    parser.add_argument('--lr', type=float, default=0.00005)  # Very low LR
    parser.add_argument('--dropout', type=float, default=0.45)  # VERY heavy dropout
    parser.add_argument('-w', '--num_workers', type=int, default=8)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_aggressive_c2')
    parser.add_argument('--early_stop', type=int, default=15)
    parser.add_argument('--loss', type=str, default='huber', choices=['mse', 'huber', 'mae'])

    args = parser.parse_args()

    print("="*70)
    print(f"🚀 AGGRESSIVE C2 Training - Target: 0.83")
    print("="*70)
    print(f"Loss: {args.loss.upper()}")
    print(f"Dropout: {args.dropout} (VERY HEAVY)")
    print(f"LR: {args.lr}")
    print(f"Weight Decay: 0.1 (10%!)")
    print("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    print(f"\n📊 Loading C2 data...")
    if OFFICIAL_LOADER_AVAILABLE:
        train_loader, val_loader = create_official_eegdash_loaders(
            challenge='c2',
            batch_size=args.batch_size,
            mini=False,
            release="R11",
            num_workers=args.num_workers
        )
    else:
        from data.official_dataset_example import create_official_dataloaders_with_split
        train_loader, val_loader = create_official_dataloaders_with_split(
            task="contrastChangeDetection",
            challenge='c2',
            batch_size=args.batch_size,
            mini=False,
            num_workers=args.num_workers,
            val_split=0.2
        )

    # Create model with EXTREME regularization
    print(f"\n🧠 Creating model with EXTREME regularization")
    model = create_model(
        challenge='c2',
        device=device,
        dropout=args.dropout
    )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {num_params:,}")

    # Robust loss
    if args.loss == 'huber':
        criterion = nn.HuberLoss(delta=0.3)
        print("Using Huber Loss (δ=0.3, very robust)")
    elif args.loss == 'mae':
        criterion = nn.L1Loss()
        print("Using MAE Loss")
    else:
        criterion = nn.MSELoss()
        print("Using MSE Loss")

    # Optimizer with EXTREME weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.1  # 10% weight decay! Personality shouldn't overfit
    )

    # Conservative scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.3,  # Drop LR faster
        patience=5,
        min_lr=1e-7
    )

    # Training loop
    best_nrmse = float('inf')
    patience_counter = 0
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🎯 Training C2 with early stopping (patience={args.early_stop})")
    print("="*70)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, clip_grad=0.3)
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.7f}")

        # Validate
        val_loss, val_metrics, _, _ = validate(model, val_loader, criterion, device)

        print(f"  Val Loss:  {val_loss:.4f}")
        print(f"  Val NRMSE: {val_metrics['nrmse']:.4f} ⭐ (Target: 0.83)")
        print(f"  Val RMSE:  {val_metrics['rmse']:.4f}")

        # Update scheduler
        scheduler.step(val_metrics['nrmse'])

        # Save best
        if val_metrics['nrmse'] < best_nrmse:
            improvement = best_nrmse - val_metrics['nrmse']
            best_nrmse = val_metrics['nrmse']
            patience_counter = 0

            checkpoint_path = checkpoint_dir / "c2_aggressive_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_nrmse': best_nrmse,
                'val_metrics': val_metrics,
                'args': vars(args)
            }, checkpoint_path)
            print(f"  ✅ Saved best (NRMSE: {best_nrmse:.4f}, improved by {improvement:.4f})")
        else:
            patience_counter += 1
            print(f"  ⏳ No improvement ({patience_counter}/{args.early_stop})")

        if patience_counter >= args.early_stop:
            print(f"\n⚠️  Early stopping at epoch {epoch}")
            break

    print("\n" + "="*70)
    print("✅ C2 Training Complete!")
    print(f"   Best NRMSE: {best_nrmse:.4f}")
    print(f"   Target: 0.83")
    if best_nrmse < 0.83:
        print(f"   🎉 BEAT TARGET!")
    else:
        print(f"   ⚠️  Gap: {best_nrmse - 0.83:.4f}")
    print("="*70)


if __name__ == "__main__":
    main()
