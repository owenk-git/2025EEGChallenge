"""
AGGRESSIVE C1 TRAINING - Target: Drop 1.33 → 0.90

Multiple approaches:
1. Different RT extraction methods (median, mode, percentiles)
2. Huber loss for robustness
3. Heavy regularization
4. Output range optimization
5. Early stopping to prevent overfitting
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


def train_epoch(model, dataloader, criterion, optimizer, device, clip_grad=1.0):
    """Train for one epoch"""
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
    parser = argparse.ArgumentParser(description="Aggressive C1 training")

    parser.add_argument('-e', '--epochs', type=int, default=50)
    parser.add_argument('-b', '--batch_size', type=int, default=64)  # Smaller batch
    parser.add_argument('--lr', type=float, default=0.0001)  # Lower LR
    parser.add_argument('--dropout', type=float, default=0.35)  # Heavy dropout
    parser.add_argument('-w', '--num_workers', type=int, default=8)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_aggressive_c1')
    parser.add_argument('--early_stop', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--loss', type=str, default='huber', choices=['mse', 'huber', 'mae'])
    parser.add_argument('--output_min', type=float, default=0.5)
    parser.add_argument('--output_max', type=float, default=1.5)
    parser.add_argument('--rt_method', type=str, default='mean',
                       choices=['mean', 'median', 'mode', 'p25', 'p75', 'trimmed_mean'],
                       help='RT extraction method')

    args = parser.parse_args()

    print("="*70)
    print(f"🚀 AGGRESSIVE C1 Training - Target: 0.90")
    print("="*70)
    print(f"Loss: {args.loss.upper()}")
    print(f"Dropout: {args.dropout}")
    print(f"LR: {args.lr}")
    print(f"Output range: [{args.output_min}, {args.output_max}]")
    print(f"RT method: {args.rt_method}")
    print("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    print(f"\n📊 Loading data...")
    if OFFICIAL_LOADER_AVAILABLE:
        train_loader, val_loader = create_official_eegdash_loaders(
            challenge='c1',
            batch_size=args.batch_size,
            mini=False,
            release="R11",
            num_workers=args.num_workers,
            rt_method=args.rt_method  # Pass RT method!
        )
    else:
        from data.official_dataset_example import create_official_dataloaders_with_split
        train_loader, val_loader = create_official_dataloaders_with_split(
            task="contrastChangeDetection",
            challenge='c1',
            batch_size=args.batch_size,
            mini=False,
            num_workers=args.num_workers,
            val_split=0.2
        )

    # Create model with heavy regularization
    print(f"\n🧠 Creating model with heavy regularization")
    model = create_model(
        challenge='c1',
        device=device,
        dropout=args.dropout,
        output_range=(args.output_min, args.output_max)
    )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {num_params:,}")

    # Setup loss
    if args.loss == 'huber':
        criterion = nn.HuberLoss(delta=0.5)  # More robust to outliers
        print("Using Huber Loss (robust to outliers)")
    elif args.loss == 'mae':
        criterion = nn.L1Loss()
        print("Using MAE Loss")
    else:
        criterion = nn.MSELoss()
        print("Using MSE Loss")

    # Optimizer with HEAVY weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.05  # 5x heavier regularization
    )

    # Aggressive scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        
    )

    # Training loop with early stopping
    best_nrmse = float('inf')
    patience_counter = 0
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🎯 Training with early stopping (patience={args.early_stop})")
    print("="*70)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, clip_grad=0.5)
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Validate
        val_loss, val_metrics, _, _ = validate(model, val_loader, criterion, device)

        print(f"  Val Loss:  {val_loss:.4f}")
        print(f"  Val NRMSE: {val_metrics['nrmse']:.4f} ⭐ (Target: 0.90)")
        print(f"  Val RMSE:  {val_metrics['rmse']:.4f}")

        # Update scheduler
        scheduler.step(val_metrics['nrmse'])

        # Save best model
        if val_metrics['nrmse'] < best_nrmse:
            improvement = best_nrmse - val_metrics['nrmse']
            best_nrmse = val_metrics['nrmse']
            patience_counter = 0

            checkpoint_path = checkpoint_dir / "c1_aggressive_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_nrmse': best_nrmse,
                'val_metrics': val_metrics,
                'args': vars(args)
            }, checkpoint_path)
            print(f"  ✅ Saved best model (NRMSE: {best_nrmse:.4f}, improved by {improvement:.4f})")
        else:
            patience_counter += 1
            print(f"  ⏳ No improvement ({patience_counter}/{args.early_stop})")

        # Early stopping
        if patience_counter >= args.early_stop:
            print(f"\n⚠️  Early stopping triggered at epoch {epoch}")
            break

    print("\n" + "="*70)
    print("✅ Training Complete!")
    print(f"   Best NRMSE: {best_nrmse:.4f}")
    print(f"   Target: 0.90")
    if best_nrmse < 0.90:
        print(f"   🎉 BEAT TARGET!")
    else:
        print(f"   ⚠️  Gap: {best_nrmse - 0.90:.4f}")
    print("="*70)


if __name__ == "__main__":
    main()
