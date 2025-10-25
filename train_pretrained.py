"""
Train pretrained CNN model for EEG

Alternative to BENDR (which needs specific library)
Uses standard torchvision pretrained models adapted for EEG

Usage:
    python train_pretrained.py --challenge 1 --backbone resnet18
    python train_pretrained.py --challenge 2 --backbone efficientnet_b0
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import numpy as np

from models.pretrained_eeg import create_pretrained_model
from utils.metrics import compute_all_metrics

try:
    from data.official_eegdash_loader import create_official_eegdash_loaders
    OFFICIAL_LOADER_AVAILABLE = True
except ImportError:
    OFFICIAL_LOADER_AVAILABLE = False
    from data.official_dataset_example import create_official_dataloaders_with_split


def train_epoch(model, dataloader, criterion, optimizer, device, clip_grad=1.0):
    """Train one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc="Training")
    for data, target in pbar:
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
    """Validate"""
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

    return avg_loss, metrics


def main():
    parser = argparse.ArgumentParser(description="Train pretrained CNN")

    parser.add_argument('-c', '--challenge', type=int, required=True, choices=[1, 2])
    parser.add_argument('-e', '--epochs', type=int, default=50)
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--backbone', type=str, default='resnet18',
                       choices=['resnet18', 'resnet34', 'resnet50', 'efficientnet_b0'])
    parser.add_argument('-w', '--num_workers', type=int, default=8)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_pretrained')
    parser.add_argument('--early_stop', type=int, default=10)

    args = parser.parse_args()

    print("="*70)
    print(f"🔄 Transfer Learning - Challenge {args.challenge}")
    print("="*70)
    print(f"Backbone: {args.backbone} (ImageNet pretrained)")
    print(f"Strategy: Fine-tune on EEG data")
    print("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    print(f"\n📊 Loading data...")
    if OFFICIAL_LOADER_AVAILABLE:
        train_loader, val_loader = create_official_eegdash_loaders(
            challenge=f'c{args.challenge}',
            batch_size=args.batch_size,
            mini=False,
            release="R11",
            num_workers=args.num_workers
        )
    else:
        from data.official_dataset_example import create_official_dataloaders_with_split
        train_loader, val_loader = create_official_dataloaders_with_split(
            task="contrastChangeDetection",
            challenge=f'c{args.challenge}',
            batch_size=args.batch_size,
            mini=False,
            num_workers=args.num_workers,
            val_split=0.2
        )

    # Create model
    print(f"\n🔄 Creating pretrained model...")
    model = create_pretrained_model(
        backbone=args.backbone,
        challenge=f'c{args.challenge}',
        device=device,
        output_range=(0.5, 1.5) if args.challenge == 1 else None
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    # Training loop
    best_nrmse = float('inf')
    patience_counter = 0
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🎯 Training with early stopping (patience={args.early_stop})")
    print("="*70)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        val_loss, val_metrics = validate(model, val_loader, criterion, device)

        target_nrmse = 0.93 if args.challenge == 1 else 1.00
        print(f"  Val Loss:  {val_loss:.4f}")
        print(f"  Val NRMSE: {val_metrics['nrmse']:.4f} ⭐ (Target: {target_nrmse:.2f})")

        scheduler.step(val_metrics['nrmse'])

        if val_metrics['nrmse'] < best_nrmse:
            improvement = best_nrmse - val_metrics['nrmse']
            best_nrmse = val_metrics['nrmse']
            patience_counter = 0

            checkpoint_path = checkpoint_dir / f"c{args.challenge}_pretrained_best.pth"
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
    print("✅ Transfer Learning Complete!")
    print(f"   Best NRMSE: {best_nrmse:.4f}")
    print("="*70)


if __name__ == "__main__":
    main()
