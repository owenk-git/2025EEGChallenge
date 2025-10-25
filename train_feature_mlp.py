"""
Train Feature MLP model

Combines hand-crafted features with neural network learning.
Works in Docker environment (only PyTorch + numpy).

Usage:
    python train_feature_mlp.py --challenge 1 --epochs 100
    python train_feature_mlp.py --challenge 2 --epochs 150
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm

from models.feature_mlp import create_feature_mlp
from data.official_eegdash_loader import create_official_eegdash_loaders


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    n_batches = 0

    pbar = tqdm(train_loader, desc="Training")
    for data, target in pbar:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.squeeze(), target)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / n_batches


def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    n_batches = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output.squeeze(), target)

            total_loss += loss.item()
            n_batches += 1

            all_preds.append(output.squeeze())
            all_targets.append(target)

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    # Calculate NRMSE
    rmse = torch.sqrt(criterion(all_preds, all_targets))
    nrmse = rmse / all_targets.std()

    return total_loss / n_batches, nrmse.item()


def main():
    parser = argparse.ArgumentParser(description="Train Feature MLP")

    parser.add_argument('--challenge', type=int, required=True, choices=[1, 2])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--early_stop', type=int, default=15)
    parser.add_argument('--num_workers', type=int, default=4)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("="*70)
    print(f"🧠 Training Feature MLP for Challenge {args.challenge}")
    print("="*70)
    print(f"Strategy: Hand-crafted features + Neural Network")
    print(f"Features: Band power, spectral, time-domain (35 total)")
    print(f"Device: {device}")
    print("="*70)

    # Create output directory
    output_dir = Path('checkpoints_feature_mlp')
    output_dir.mkdir(exist_ok=True)

    # Load data
    print(f"\n📊 Loading data...")
    train_loader, val_loader = create_official_eegdash_loaders(
        challenge=f'c{args.challenge}',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=0.2
    )

    # Create model
    print(f"\n🧠 Creating Feature MLP model...")
    model = create_feature_mlp(
        challenge=f'c{args.challenge}',
        device=device
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7
    )

    # Training loop
    best_nrmse = float('inf')
    patience_counter = 0
    target_nrmse = 0.93 if args.challenge == 1 else 1.00

    print(f"\n🚀 Starting training...")
    print(f"Target NRMSE: {target_nrmse:.2f}\n")

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_nrmse = validate(model, val_loader, criterion, device)

        # Scheduler step
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Print progress
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  LR: {current_lr:.6f}")
        print(f"  Val Loss:  {val_loss:.4f}")

        # Check if best
        if val_nrmse < best_nrmse:
            best_nrmse = val_nrmse
            patience_counter = 0

            # Save best model
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_nrmse': val_nrmse,
                'args': args
            }
            torch.save(checkpoint, output_dir / f'c{args.challenge}_feature_mlp_best.pth')
            print(f"  Val NRMSE: {val_nrmse:.4f} ⭐ (Target: {target_nrmse:.2f})")
        else:
            patience_counter += 1
            print(f"  Val NRMSE: {val_nrmse:.4f} ⭐ (Target: {target_nrmse:.2f})")
            print(f"  ⏳ No improvement ({patience_counter}/{args.early_stop})")

        # Early stopping
        if patience_counter >= args.early_stop:
            print(f"\n⚠️  Early stopping at epoch {epoch}")
            break

        print()

    print("="*70)
    print("✅ Feature MLP Training Complete!")
    print(f"   Best NRMSE: {best_nrmse:.4f}")
    print(f"   Target: {target_nrmse:.2f}")
    print("="*70)


if __name__ == "__main__":
    main()
