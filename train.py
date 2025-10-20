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
import json
from datetime import datetime

from models.eegnet import create_model
from data.dataset import create_dataloader
from data.streaming_dataset import create_streaming_dataloader
from utils.metrics import compute_all_metrics, normalized_rmse

# Try to import official dataset (optional)
try:
    from data.official_dataset_example import (
        create_official_dataloader,
        create_official_dataloaders_with_split,
        create_official_dataloaders_train_val_test
    )
    OFFICIAL_AVAILABLE = True
except ImportError:
    OFFICIAL_AVAILABLE = False


def log_experiment(args, best_metrics, best_epoch):
    """Log experiment configuration and results"""
    if args.exp_num is None:
        return

    exp_dir = Path("experiments")
    exp_dir.mkdir(exist_ok=True)

    # Create experiment entry
    exp_entry = {
        "exp_num": args.exp_num,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "challenge": args.challenge,
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "dropout": args.dropout,
            "max_subjects": args.max_subjects,
            "use_official": args.use_official,
            "official_mini": args.official_mini if args.use_official else None,
        },
        "results": {
            "best_val_nrmse": round(best_metrics.get('nrmse', 0), 4),
            "best_val_rmse": round(best_metrics.get('rmse', 0), 4),
            "best_val_mae": round(best_metrics.get('mae', 0), 4),
            "best_epoch": best_epoch,
        }
    }

    # Append to JSON log
    json_log = exp_dir / "experiments.json"
    if json_log.exists():
        with open(json_log, 'r') as f:
            experiments = json.load(f)
    else:
        experiments = []

    experiments.append(exp_entry)

    with open(json_log, 'w') as f:
        json.dump(experiments, f, indent=2)

    print(f"\n✅ Experiment #{args.exp_num} logged to {json_log}")


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


def validate_with_metrics(model, dataloader, criterion, device):
    """
    Validate model and compute all metrics

    Returns:
        avg_loss: Average MSE loss
        metrics: Dict with NRMSE, RMSE, MAE
        all_predictions: All predictions (for saving)
        all_targets: All targets (for saving)
    """
    model.eval()
    total_loss = 0
    num_batches = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            num_batches += 1

            # Collect predictions and targets
            all_predictions.append(output.cpu())
            all_targets.append(target.cpu())

    avg_loss = total_loss / num_batches

    # Concatenate all predictions and targets
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Compute metrics
    metrics = compute_all_metrics(all_predictions, all_targets)

    return avg_loss, metrics, all_predictions, all_targets


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
    val_loader = None  # Initialize validation loader

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

        # Use train/val split unless disabled
        if not args.no_val:
            print(f"   Validation split: {args.val_split:.1%}")
            train_loader, val_loader = create_official_dataloaders_with_split(
                task=args.official_task,
                challenge=f'c{args.challenge}',
                batch_size=args.batch_size,
                mini=args.official_mini,
                max_subjects=args.max_subjects,
                num_workers=args.num_workers,
                val_split=args.val_split
            )
        else:
            print(f"   ⚠️  No validation split (--no_val)")
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
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Training loop
    print(f"\n🎯 Training for {args.epochs} epochs")
    print("="*70)

    # Track best validation metrics (use NRMSE for model selection)
    best_nrmse = float('inf')
    best_metrics = {}
    best_epoch = 0
    best_predictions = None
    best_targets = None

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"  Train Loss: {train_loss:.4f}")

        # Validate with metrics if validation set exists
        if val_loader is not None:
            val_loss, val_metrics, predictions, targets = validate_with_metrics(
                model, val_loader, criterion, device
            )

            print(f"  Val Loss:  {val_loss:.4f}")
            print(f"  Val NRMSE: {val_metrics['nrmse']:.4f} ⭐ (Competition Metric)")
            print(f"  Val RMSE:  {val_metrics['rmse']:.4f}")
            print(f"  Val MAE:   {val_metrics['mae']:.4f}")

            # Update scheduler based on validation loss
            scheduler.step(val_loss)

            # Save best model based on validation NRMSE
            if val_metrics['nrmse'] < best_nrmse:
                best_nrmse = val_metrics['nrmse']
                best_metrics = val_metrics
                best_epoch = epoch
                best_predictions = predictions
                best_targets = targets

                checkpoint_path = checkpoint_dir / f"c{args.challenge}_best.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_nrmse': best_nrmse,
                    'val_metrics': val_metrics,
                }, checkpoint_path)
                print(f"  ✅ Saved best model (NRMSE: {best_nrmse:.4f})")

        else:
            # No validation - use training loss (old behavior)
            scheduler.step(train_loss)

            if train_loss < best_nrmse:  # Reuse best_nrmse as best_loss
                best_nrmse = train_loss
                best_metrics = {'nrmse': train_loss}  # Dummy metrics
                best_epoch = epoch

                checkpoint_path = checkpoint_dir / f"c{args.challenge}_best.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss,
                }, checkpoint_path)
                print(f"  ✅ Saved best model (loss: {train_loss:.4f})")

        # Save checkpoint every N epochs
        if epoch % args.save_every == 0:
            checkpoint_path = checkpoint_dir / f"c{args.challenge}_epoch{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_nrmse': val_metrics.get('nrmse', train_loss) if val_loader else train_loss,
            }, checkpoint_path)

    print("\n" + "="*70)
    if val_loader is not None:
        print(f"✅ Training complete! Best val NRMSE: {best_nrmse:.4f} (epoch {best_epoch})")
        print(f"   Best val RMSE: {best_metrics['rmse']:.4f}")
        print(f"   Best val MAE:  {best_metrics['mae']:.4f}")
    else:
        print(f"✅ Training complete! Best train loss: {best_nrmse:.4f} (epoch {best_epoch})")

    print(f"📁 Model saved to: {checkpoint_dir}/c{args.challenge}_best.pth")
    print("="*70)

    # Save predictions and results for analysis
    if val_loader is not None and best_predictions is not None:
        results_dir = Path("results")
        if args.exp_num is not None:
            results_dir = results_dir / f"exp_{args.exp_num}"
        results_dir.mkdir(parents=True, exist_ok=True)

        results_path = results_dir / f"c{args.challenge}_results.pt"
        torch.save({
            'predictions': best_predictions,
            'targets': best_targets,
            'metrics': best_metrics,
            'best_epoch': best_epoch,
            'config': vars(args),
        }, results_path)
        print(f"\n💾 Saved predictions and metrics to: {results_path}")

    # Log experiment
    log_experiment(args, best_metrics, best_epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EEG Challenge model")

    # Data args
    parser.add_argument('-d', '--data_path', type=str, default=None,
                        help='Path to BIDS dataset (for custom/local mode)')
    parser.add_argument('-c', '--challenge', type=int, required=True, choices=[1, 2],
                        help='Challenge number (1 or 2)')

    # Official dataset args
    parser.add_argument('-o', '--use_official', action='store_true',
                        help='Use official EEGChallengeDataset (recommended)')
    parser.add_argument('--task', '--official_task', type=str, default='contrastChangeDetection',
                        dest='official_task', help='Task name for official dataset')
    parser.add_argument('-m', '--official_mini', action='store_true',
                        help='Use mini dataset (faster for testing)')

    # Model args
    parser.add_argument('--drop', '--dropout', type=float, default=0.20,
                        dest='dropout', help='Dropout rate')

    # Training args
    parser.add_argument('-e', '--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('-w', '--num_workers', type=int, default=4,
                        help='Number of data loading workers')

    # Checkpoint args
    parser.add_argument('--ckpt', '--checkpoint_dir', type=str, default='./checkpoints',
                        dest='checkpoint_dir', help='Directory to save checkpoints')
    parser.add_argument('--save', '--save_every', type=int, default=10,
                        dest='save_every', help='Save checkpoint every N epochs')

    # Streaming args
    parser.add_argument('-s', '--use_streaming', action='store_true',
                        help='Use S3 streaming (no download)')
    parser.add_argument('--max', '--max_subjects', type=int, default=None,
                        dest='max_subjects', help='Maximum number of subjects to use (for efficiency)')

    # Validation args
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split fraction (default: 0.2)')
    parser.add_argument('--no_val', action='store_true',
                        help='Disable validation split (train on all data)')

    # Experiment tracking
    parser.add_argument('--num', '--exp_num', type=int, default=None,
                        dest='exp_num', help='Experiment number for tracking (see experiments/EXPERIMENT_LOG.md)')

    args = parser.parse_args()
    main(args)
