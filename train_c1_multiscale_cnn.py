#!/usr/bin/env python3
"""
Train Multi-Scale CNN for C1

Uses parallel Conv1D branches with different kernel sizes
to capture different temporal patterns
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

from models.multiscale_cnn_rt import MultiScaleCNN_RT, InceptionStyleCNN_RT
from data.trial_level_loader import TrialLevelDataset


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []

    pbar = tqdm(train_loader, desc='Training')
    for X_batch, y_batch in pbar:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).squeeze()

        optimizer.zero_grad()

        predictions = model(X_batch).squeeze()
        loss = criterion(predictions, y_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(predictions.detach().cpu().numpy())
        all_targets.extend(y_batch.detach().cpu().numpy())

        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Compute NRMSE
    rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
    nrmse = rmse / np.std(all_targets)

    # Compute correlation
    corr = np.corrcoef(all_preds, all_targets)[0, 1]

    return total_loss / len(train_loader), nrmse, corr, all_preds, all_targets


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in tqdm(val_loader, desc='Validation'):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).squeeze()

            predictions = model(X_batch).squeeze()
            loss = criterion(predictions, y_batch)

            total_loss += loss.item()
            all_preds.extend(predictions.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Compute NRMSE
    rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
    nrmse = rmse / np.std(all_targets)

    # Compute correlation
    corr = np.corrcoef(all_preds, all_targets)[0, 1]

    return total_loss / len(val_loader), nrmse, corr, all_preds, all_targets


def main():
    parser = argparse.ArgumentParser(description='Train Multi-Scale CNN for C1')
    parser.add_argument('--model', type=str, default='multiscale',
                       choices=['multiscale', 'inception'],
                       help='Model architecture')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    print("Loading dataset...")
    full_dataset = TrialLevelDataset(challenge='c1', mini=False)

    # Train/val split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=4, pin_memory=True)

    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    # Create model
    print(f"Creating {args.model} CNN model...")
    if args.model == 'multiscale':
        model = MultiScaleCNN_RT(
            n_channels=129,
            trial_length=200,
            pre_stim_points=50
        ).to(device)
        checkpoint_name = 'c1_multiscale_cnn_best.pt'
    else:  # inception
        model = InceptionStyleCNN_RT(
            n_channels=129,
            trial_length=200
        ).to(device)
        checkpoint_name = 'c1_inception_cnn_best.pt'

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Loss function
    criterion = nn.MSELoss()

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Training loop
    best_val_nrmse = float('inf')
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Training {args.model.upper()} CNN for C1")
    print(f"{'='*60}\n")

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")

        # Train
        train_loss, train_nrmse, train_corr, train_preds, train_targets = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_nrmse, val_corr, val_preds, val_targets = validate(
            model, val_loader, criterion, device
        )

        # Scheduler step
        scheduler.step(val_nrmse)

        # Print statistics
        print(f"\nEpoch {epoch+1} Results:")
        print(f"  Train - Loss: {train_loss:.4f}, NRMSE: {train_nrmse:.4f}, Corr: {train_corr:.4f}")
        print(f"  Train - Pred std: {train_preds.std():.4f}, Target std: {train_targets.std():.4f}")
        print(f"  Train - Pred range: [{train_preds.min():.3f}, {train_preds.max():.3f}]")
        print(f"  Val   - Loss: {val_loss:.4f}, NRMSE: {val_nrmse:.4f}, Corr: {val_corr:.4f}")
        print(f"  Val   - Pred std: {val_preds.std():.4f}, Target std: {val_targets.std():.4f}")
        print(f"  Val   - Pred range: [{val_preds.min():.3f}, {val_preds.max():.3f}]")

        # Save best model
        if val_nrmse < best_val_nrmse:
            best_val_nrmse = val_nrmse
            checkpoint_path = checkpoint_dir / checkpoint_name
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_nrmse': val_nrmse,
                'val_corr': val_corr,
                'model_type': args.model,
            }, checkpoint_path)
            print(f"  ✓ Saved best model (NRMSE: {val_nrmse:.4f})")

        print()

    print(f"Training complete! Best Val NRMSE: {best_val_nrmse:.4f}")


if __name__ == '__main__':
    main()
