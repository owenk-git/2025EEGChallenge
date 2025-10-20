"""
Training script for EEG Challenge 2025

Usage:
    # Official EEGChallengeDataset (recommended)
    python train.py --challenge 1 --use_official --max_subjects 50 --epochs 50

    # Custom S3 streaming
    python train.py --challenge 1 --data_path s3://nmdatasets/NeurIPS2025/R1_mini_L100_bdf --use_streaming --max_subjects 50 --epochs 50

    # Local mini dataset
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
from data.streaming_dataset import create_streaming_dataloader

# Try to import official dataset (optional)
try:
    from data.official_dataset_example import create_official_dataloader
    OFFICIAL_AVAILABLE = True
except ImportError:
    OFFICIAL_AVAILABLE = False


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
    print(f"\n📊 Loading data...")

    # Choose data loading approach
    if args.use_official:
        # Use official EEGChallengeDataset
        if not OFFICIAL_AVAILABLE:
            print("❌ Official dataset not available. Install: pip install eegdash braindecode")
            print("   Or remove --use_official flag to use custom loader")
            return

        print(f"📦 Using official EEGChallengeDataset")
        print(f"   Task: {args.official_task}")
        print(f"   Mini: {args.official_mini}")
        if args.max_subjects:
            print(f"   Limiting to {args.max_subjects} subjects")

        train_loader = create_official_dataloader(
            task=args.official_task,
            challenge=f'c{args.challenge}',
            batch_size=args.batch_size,
            mini=args.official_mini,
            max_subjects=args.max_subjects,
            num_workers=args.num_workers
        )

    elif args.use_streaming or (args.data_path and args.data_path.startswith('s3://')):
        # Use custom S3 streaming
        if not args.data_path:
            print("❌ --data_path required for streaming mode")
            return

        print(f"☁️  Using custom S3 streaming (no download)")
        print(f"   Path: {args.data_path}")
        if args.max_subjects:
            print(f"   Limiting to {args.max_subjects} subjects")

        train_loader = create_streaming_dataloader(
            args.data_path,
            challenge=f'c{args.challenge}',
            batch_size=args.batch_size,
            max_subjects=args.max_subjects,
            use_cache=True,
            cache_dir='./data_cache'
        )

    else:
        # Use local dataset
        if not args.data_path:
            print("❌ --data_path required for local mode")
            return

        print(f"📁 Using local dataset")
        print(f"   Path: {args.data_path}")

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
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to BIDS dataset (for custom/local mode)')
    parser.add_argument('--challenge', type=int, required=True, choices=[1, 2],
                        help='Challenge number (1 or 2)')

    # Official dataset args
    parser.add_argument('--use_official', action='store_true',
                        help='Use official EEGChallengeDataset (recommended)')
    parser.add_argument('--official_task', type=str, default='contrastChangeDetection',
                        help='Task name for official dataset')
    parser.add_argument('--official_mini', action='store_true',
                        help='Use mini dataset (faster for testing)')

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

    # Streaming args
    parser.add_argument('--use_streaming', action='store_true',
                        help='Use S3 streaming (no download)')
    parser.add_argument('--max_subjects', type=int, default=None,
                        help='Maximum number of subjects to use (for efficiency)')

    args = parser.parse_args()
    main(args)
