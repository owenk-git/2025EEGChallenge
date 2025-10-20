"""
Training script for EEG Challenge 2025

Usage:
    python train.py --challenge 1 --data_path ./data/R1_mini_L100 --epochs 50
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm

from models.eegnet import create_model
from data.dataset import create_dataloader


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc="Training")
    for data, target in pbar:
        data, target = data.to(device), target.to(device)

        # Forward pass
        optimizer.zero_grad()
        output = model(data)

        # Compute loss
        loss = criterion(output, target)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss


def main(args):
    """Main training loop"""
    print("="*70)
    print(f"🚀 Training EEG Challenge {args.challenge}")
    print("="*70)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create data loaders
    print(f"\n📊 Loading data from {args.data_path}")
    train_loader = create_dataloader(
        args.data_path,
        challenge=f'c{args.challenge}',
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    # Create model
    print(f"\n🧠 Creating model for Challenge {args.challenge}")
    model = create_model(
        challenge=f'c{args.challenge}',
        device=device,
        dropout=args.dropout
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    # Loss and optimizer
    criterion = nn.MSELoss()  # For regression
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Training loop
    print(f"\n🎯 Training for {args.epochs} epochs")
    print("="*70)

    best_loss = float('inf')
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}")

        # Update scheduler
        scheduler.step(train_loss)

        # Save best model
        if train_loss < best_loss:
            best_loss = train_loss
            checkpoint_path = checkpoint_dir / f"c{args.challenge}_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, checkpoint_path)
            print(f"✅ Saved best model (loss: {best_loss:.4f})")

        # Save checkpoint every N epochs
        if epoch % args.save_every == 0:
            checkpoint_path = checkpoint_dir / f"c{args.challenge}_epoch{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, checkpoint_path)

    print("\n" + "="*70)
    print(f"✅ Training complete! Best loss: {best_loss:.4f}")
    print(f"📁 Model saved to: {checkpoint_dir}/c{args.challenge}_best.pth")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EEG Challenge model")

    # Data args
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to BIDS dataset')
    parser.add_argument('--challenge', type=int, required=True, choices=[1, 2],
                        help='Challenge number (1 or 2)')

    # Model args
    parser.add_argument('--dropout', type=float, default=0.20,
                        help='Dropout rate')

    # Training args
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')

    # Checkpoint args
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')

    args = parser.parse_args()
    main(args)
