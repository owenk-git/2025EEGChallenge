"""
Train with OFFICIAL eegdash RT extraction method

This uses the competition's official annotate_trials_with_target
Should give much better C1 results!

Usage:
    python train_official.py --challenge 1 --epochs 200 --batch_size 128
    python train_official.py --challenge 2 --epochs 200 --batch_size 128
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

# Import official loader
try:
    from data.official_eegdash_loader import create_official_eegdash_loaders
    OFFICIAL_LOADER_AVAILABLE = True
except ImportError:
    OFFICIAL_LOADER_AVAILABLE = False
    print("⚠️ Official loader not available, falling back")
    from data.official_dataset_example import create_official_dataloaders_with_split


def train_epoch(model, dataloader, criterion, optimizer, device, clip_grad=1.0):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)

        # Check for NaN
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
    parser = argparse.ArgumentParser(description="Train with official eegdash method")

    parser.add_argument('-c', '--challenge', type=int, required=True, choices=[1, 2])
    parser.add_argument('-e', '--epochs', type=int, default=200)
    parser.add_argument('-b', '--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--dropout', type=float, default=0.20)
    parser.add_argument('-w', '--num_workers', type=int, default=8)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_official')

    args = parser.parse_args()

    print("="*70)
    print(f"🚀 Training with OFFICIAL Method - Challenge {args.challenge}")
    print("="*70)
    print("Using official eegdash annotate_trials_with_target")
    print("This should match test set RT extraction!")
    print("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create data loaders with official method
    print(f"\n📊 Loading data with OFFICIAL method...")

    if OFFICIAL_LOADER_AVAILABLE and args.challenge == 1:
        print("✅ Using official eegdash loader for C1")
        try:
            train_loader, val_loader = create_official_eegdash_loaders(
                challenge='c1',
                batch_size=args.batch_size,
                mini=False,
                release="R11",
                num_workers=args.num_workers
            )
        except Exception as e:
            print(f"⚠️ Official loader failed: {e}")
            print("   Falling back to standard loader")
            from data.official_dataset_example import create_official_dataloaders_with_split
            train_loader, val_loader = create_official_dataloaders_with_split(
                task="contrastChangeDetection",
                challenge='c1',
                batch_size=args.batch_size,
                mini=False,
                num_workers=args.num_workers,
                val_split=0.2
            )
    else:
        print("Using standard loader")
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
    print(f"\n🧠 Creating model for Challenge {args.challenge}")

    model_kwargs = {
        'challenge': f'c{args.challenge}',
        'device': device,
        'dropout': args.dropout,
    }

    # Output range for C1
    if args.challenge == 1:
        model_kwargs['output_range'] = (0.5, 1.5)  # Match Oct 14 best submission

    model = create_model(**model_kwargs)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {num_params:,}")

    # Setup training
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )

    # Training loop
    best_nrmse = float('inf')
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🎯 Training for {args.epochs} epochs")
    print("="*70)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Validate
        val_loss, val_metrics, _, _ = validate(model, val_loader, criterion, device)

        target_nrmse = 0.93 if args.challenge == 1 else 1.00
        print(f"  Val Loss:  {val_loss:.4f}")
        print(f"  Val NRMSE: {val_metrics['nrmse']:.4f} ⭐ (Target: {target_nrmse:.2f})")
        print(f"  Val RMSE:  {val_metrics['rmse']:.4f}")
        print(f"  Val MAE:   {val_metrics['mae']:.4f}")

        # Update scheduler
        scheduler.step()

        # Save best model
        if val_metrics['nrmse'] < best_nrmse:
            best_nrmse = val_metrics['nrmse']

            checkpoint_path = checkpoint_dir / f"c{args.challenge}_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_nrmse': best_nrmse,
                'val_metrics': val_metrics,
            }, checkpoint_path)
            print(f"  ✅ Saved best model (NRMSE: {best_nrmse:.4f})")

    print("\n" + "="*70)
    print("✅ Training Complete!")
    print(f"   Best NRMSE: {best_nrmse:.4f}")
    print(f"   Target: {target_nrmse:.2f}")
    if best_nrmse < target_nrmse:
        print(f"   🎉 BEAT TARGET!")
    else:
        print(f"   ⚠️  Gap: {best_nrmse - target_nrmse:.4f}")
    print("="*70)


if __name__ == "__main__":
    main()
