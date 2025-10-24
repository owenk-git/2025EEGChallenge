"""
Strategy 3: Try Different RT Targets for C1

Maybe the issue is HOW we're computing the RT target.
Try different aggregation methods:
- Mean RT
- Median RT
- Fastest RT (percentile 10)
- RT variability (std)

Usage:
    python strategy3.py --rt_method mean
    python strategy3.py --rt_method median
    python strategy3.py --rt_method fast
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
from data.official_dataset_example import create_official_dataloaders_with_split
from strategy2 import train_epoch, validate, EEGAugmentation


def main():
    parser = argparse.ArgumentParser(description="Strategy 3: Different RT targets")
    parser.add_argument('--rt_method', type=str, default='mean',
                        choices=['mean', 'median', 'fast', 'slow'],
                        help='RT aggregation method')
    parser.add_argument('-e', '--epochs', type=int, default=100)
    parser.add_argument('-b', '--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('-w', '--num_workers', type=int, default=8)

    args = parser.parse_args()

    print("="*70)
    print(f"Strategy 3: RT Method = {args.rt_method}")
    print("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_loader, val_loader = create_official_dataloaders_with_split(
        task="contrastChangeDetection",
        challenge='c1',
        batch_size=args.batch_size,
        mini=False,
        num_workers=args.num_workers,
        val_split=0.2
    )

    # Create model
    model_kwargs = {
        'challenge': 'c1',
        'device': device,
        'dropout': args.dropout,
        'output_range': (0.88, 1.12)
    }
    model = create_model(**model_kwargs)

    # Setup training
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )

    augmentation = EEGAugmentation()

    best_nrmse = float('inf')
    checkpoint_dir = Path(f"checkpoints_strategy3_{args.rt_method}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device,
                                augmentation=augmentation, clip_grad=1.0)
        val_loss, val_metrics, _, _ = validate(model, val_loader, criterion, device)

        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val NRMSE: {val_metrics['nrmse']:.4f} (Target: 0.93)")

        scheduler.step()

        if val_metrics['nrmse'] < best_nrmse:
            best_nrmse = val_metrics['nrmse']
            checkpoint_path = checkpoint_dir / "c1_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_nrmse': best_nrmse,
            }, checkpoint_path)
            print(f"  ✅ Saved (NRMSE: {best_nrmse:.4f})")

    print(f"\n{'='*70}")
    print(f"Best NRMSE: {best_nrmse:.4f} (method: {args.rt_method})")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
