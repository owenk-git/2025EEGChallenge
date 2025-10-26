"""
Training script for Domain Adaptation EEGNeX model
TRUE S3 STREAMING - NO LOCAL CACHING AT ALL!

Streams data directly from AWS S3 bucket.
Requires: pip install boto3 mne

S3 Bucket: s3://nmdatasets/NeurIPS2025/
"""

import torch
import torch.optim as optim
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

from models.domain_adaptation_eegnex import (
    create_domain_adaptation_eegnex,
    DomainAdaptationLoss
)

# Import S3 streaming loader
try:
    from data.s3_streaming_loader import create_s3_streaming_loaders
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False
    print("❌ S3 streaming not available")
    print("Install: pip install boto3 mne")
    exit(1)


def compute_nrmse(predictions, targets):
    """Compute NRMSE"""
    std = np.std(targets)
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    nrmse = rmse / std
    return nrmse


def train_domain_adaptation_s3(
    challenge='c1',
    epochs=100,
    batch_size=32,
    lr=1e-3,
    lambda_mmd=0.1,
    lambda_entropy=0.01,
    lambda_adv=0.1,
    device='cuda',
    save_dir='checkpoints',
    release='R11',
    mini=False,
    max_files=None
):
    """
    Train Domain Adaptation with S3 streaming

    Args:
        challenge: 'c1' or 'c2'
        epochs: Number of epochs
        batch_size: Batch size
        lr: Learning rate
        lambda_mmd: MMD loss weight
        lambda_entropy: Entropy loss weight
        lambda_adv: Adversarial loss weight
        device: 'cuda' or 'cpu'
        save_dir: Checkpoint directory
        release: Data release (e.g., 'R11')
        mini: Use mini dataset
        max_files: Max files for testing
    """
    print(f"Training Domain Adaptation with S3 Streaming")
    print(f"Challenge: {challenge}")
    print(f"NO LOCAL CACHING - Pure cloud streaming!")

    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create S3 streaming loaders
    print("\n📡 Setting up S3 streaming...")
    train_loader, val_loader = create_s3_streaming_loaders(
        challenge=challenge,
        release=release,
        mini=mini,
        batch_size=batch_size,
        num_workers=2,  # Use fewer workers for S3 streaming
        val_split=0.2,
        max_files=max_files
    )

    # Estimate number of subjects
    num_subjects = min(len(train_loader.dataset) // 10, 100)

    # Create model
    print("\nCreating model...")
    model = create_domain_adaptation_eegnex(
        challenge=challenge,
        num_subjects=num_subjects,
        device=device
    )

    # Loss function
    criterion = DomainAdaptationLoss(
        lambda_mmd=lambda_mmd,
        lambda_entropy=lambda_entropy,
        lambda_adv=lambda_adv
    )

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Calculate target std for NRMSE
    print("\nCalculating target std...")
    all_val_targets = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            all_val_targets.append(y_batch.cpu().numpy())
    all_val_targets = np.concatenate(all_val_targets)
    target_std = np.std(all_val_targets)
    print(f"Target std: {target_std:.4f}")

    # Training loop
    best_val_nrmse = float('inf')

    for epoch in range(epochs):
        model.train()

        # Gradient reversal strength
        p = float(epoch) / epochs
        alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0

        train_losses = []
        train_loss_details = {
            'task_loss': [],
            'mmd_loss': [],
            'entropy_loss': [],
            'adv_loss': []
        }

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [S3 Stream]")
        for batch_idx, (X_batch, y_batch) in enumerate(pbar):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            if X_batch.size(0) < 2:
                continue

            # Split for MMD
            split_idx = X_batch.size(0) // 2

            # Dummy subject labels
            subject_batch = torch.randint(0, num_subjects, (X_batch.size(0),)).to(device)

            # Forward pass
            predictions, features, subject_logits = model(
                X_batch, alpha=alpha, return_features=True
            )

            # Split features
            source_features = features[:split_idx]
            target_features = features[split_idx:]

            # Compute loss
            total_loss, loss_dict = criterion(
                predictions, y_batch,
                source_features=source_features,
                target_features=target_features,
                subject_logits=subject_logits,
                subject_labels=subject_batch
            )

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Record losses
            train_losses.append(loss_dict['total_loss'])
            for key in train_loss_details:
                if key in loss_dict:
                    train_loss_details[key].append(loss_dict[key])

            pbar.set_postfix({'loss': f"{loss_dict['total_loss']:.4f}"})

        scheduler.step()

        # Validation
        model.eval()
        val_predictions = []
        val_targets = []

        with torch.no_grad():
            for X_batch, y_batch in tqdm(val_loader, desc="Validation"):
                X_batch = X_batch.to(device)
                predictions = model(X_batch)
                val_predictions.append(predictions.cpu().numpy())
                val_targets.append(y_batch.cpu().numpy())

        val_predictions = np.concatenate(val_predictions)
        val_targets = np.concatenate(val_targets)

        # Compute NRMSE
        val_nrmse = compute_nrmse(val_predictions, val_targets)

        # Print progress
        avg_train_loss = np.mean(train_losses)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_train_loss:.4f} | Val NRMSE: {val_nrmse:.4f} | α: {alpha:.3f}")

        # Save best model
        if val_nrmse < best_val_nrmse:
            best_val_nrmse = val_nrmse
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_nrmse': val_nrmse,
                'y_mean': 0.0,
                'y_std': 1.0,
                'num_subjects': num_subjects,
                'challenge': challenge
            }
            torch.save(checkpoint, save_dir / f'domain_adaptation_{challenge}_best.pt')
            print(f"  ✓ Saved (NRMSE: {val_nrmse:.4f})")

    print(f"\n✨ Training completed! Best Val NRMSE: {best_val_nrmse:.4f}")
    print(f"📡 All data streamed from S3 - no local caching!")
    return best_val_nrmse


def main():
    parser = argparse.ArgumentParser(description='Train with S3 Streaming')
    parser.add_argument('--challenge', type=str, default='c1', choices=['c1', 'c2'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lambda_mmd', type=float, default=0.1)
    parser.add_argument('--lambda_entropy', type=float, default=0.01)
    parser.add_argument('--lambda_adv', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--release', type=str, default='R11')
    parser.add_argument('--mini', action='store_true', help='Use mini dataset')
    parser.add_argument('--max_files', type=int, default=None, help='Max files for testing')

    args = parser.parse_args()

    if not S3_AVAILABLE:
        print("\n❌ S3 streaming not available!")
        print("Install: pip install boto3 mne")
        exit(1)

    train_domain_adaptation_s3(
        challenge=args.challenge,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_mmd=args.lambda_mmd,
        lambda_entropy=args.lambda_entropy,
        lambda_adv=args.lambda_adv,
        device=args.device,
        save_dir=args.save_dir,
        release=args.release,
        mini=args.mini,
        max_files=args.max_files
    )


if __name__ == '__main__':
    main()
