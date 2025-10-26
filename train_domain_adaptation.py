"""
Training script for Domain Adaptation EEGNeX model

Implements:
- Maximum Mean Discrepancy (MMD) for distribution alignment
- Entropy minimization for confident predictions
- Subject-adversarial training for subject-invariant features
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import argparse
from pathlib import Path

from models.domain_adaptation_eegnex import (
    create_domain_adaptation_eegnex,
    DomainAdaptationLoss,
    compute_mmd_loss
)


def load_data(challenge='c1'):
    """Load preprocessed data"""
    data_dir = Path('data/preprocessed')

    X_train = np.load(data_dir / f'{challenge}_X_train.npy')
    y_train = np.load(data_dir / f'{challenge}_y_train.npy')
    X_val = np.load(data_dir / f'{challenge}_X_val.npy')
    y_val = np.load(data_dir / f'{challenge}_y_val.npy')

    # Load subject IDs if available
    subject_train_path = data_dir / f'{challenge}_subjects_train.npy'
    subject_val_path = data_dir / f'{challenge}_subjects_val.npy'

    if subject_train_path.exists():
        subjects_train = np.load(subject_train_path)
        subjects_val = np.load(subject_val_path)
    else:
        # Create dummy subject IDs
        subjects_train = np.arange(len(y_train)) % 50
        subjects_val = np.arange(len(y_val)) % 50

    return X_train, y_train, subjects_train, X_val, y_val, subjects_val


def get_subject_mapping(subjects_train):
    """Create mapping from subject ID to integer index"""
    unique_subjects = np.unique(subjects_train)
    subject_to_idx = {subj: idx for idx, subj in enumerate(unique_subjects)}
    return subject_to_idx, len(unique_subjects)


def normalize_targets(y_train, y_val):
    """Normalize targets to mean=0, std=1 (for NRMSE)"""
    mean = y_train.mean()
    std = y_train.std()

    y_train_norm = (y_train - mean) / std
    y_val_norm = (y_val - mean) / std

    return y_train_norm, y_val_norm, mean, std


def compute_nrmse(predictions, targets, std):
    """Compute NRMSE (Normalized RMSE)"""
    mse = ((predictions - targets) ** 2).mean()
    rmse = np.sqrt(mse)
    nrmse = rmse / std
    return nrmse


def train_domain_adaptation(challenge='c1', epochs=100, batch_size=64, lr=1e-3,
                           lambda_mmd=0.1, lambda_entropy=0.01, lambda_adv=0.1,
                           device='cuda', save_dir='checkpoints'):
    """
    Train Domain Adaptation EEGNeX model

    Args:
        challenge: 'c1' or 'c2'
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        lambda_mmd: Weight for MMD loss
        lambda_entropy: Weight for entropy minimization
        lambda_adv: Weight for adversarial loss
        device: 'cuda' or 'cpu'
        save_dir: Directory to save checkpoints
    """
    print(f"Training Domain Adaptation model for {challenge}")
    print(f"MMD λ={lambda_mmd}, Entropy λ={lambda_entropy}, Adversarial λ={lambda_adv}")

    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    X_train, y_train, subjects_train, X_val, y_val, subjects_val = load_data(challenge)

    # Get subject mapping
    subject_to_idx, num_subjects = get_subject_mapping(subjects_train)
    print(f"Number of unique subjects: {num_subjects}")

    # Map subjects to indices
    subject_indices_train = np.array([subject_to_idx.get(s, 0) for s in subjects_train])
    subject_indices_val = np.array([subject_to_idx.get(s, 0) for s in subjects_val])

    # Normalize targets
    y_train_norm, y_val_norm, y_mean, y_std = normalize_targets(y_train, y_val)

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train_norm)
    subjects_train_t = torch.LongTensor(subject_indices_train)

    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val_norm)
    subjects_val_t = torch.LongTensor(subject_indices_val)

    # Create datasets
    train_dataset = TensorDataset(X_train_t, y_train_t, subjects_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t, subjects_val_t)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    print("Creating model...")
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

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    best_val_nrmse = float('inf')

    for epoch in range(epochs):
        model.train()

        # Compute gradient reversal strength (increases from 0 to 1)
        p = float(epoch) / epochs
        alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0

        train_losses = []
        train_loss_details = {
            'task_loss': [],
            'mmd_loss': [],
            'entropy_loss': [],
            'adv_loss': []
        }

        for batch_idx, (X_batch, y_batch, subject_batch) in enumerate(train_loader):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            subject_batch = subject_batch.to(device)

            # Split batch into source and target for MMD
            # Use first half as "source" and second half as "target"
            split_idx = X_batch.size(0) // 2

            # Forward pass
            predictions, features, subject_logits = model(
                X_batch, alpha=alpha, return_features=True
            )

            # Split features for MMD
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

        scheduler.step()

        # Validation
        model.eval()
        val_predictions = []
        val_targets = []

        with torch.no_grad():
            for X_batch, y_batch, _ in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                predictions = model(X_batch)

                val_predictions.append(predictions.cpu().numpy())
                val_targets.append(y_batch.cpu().numpy())

        val_predictions = np.concatenate(val_predictions)
        val_targets = np.concatenate(val_targets)

        # Denormalize predictions
        val_predictions_denorm = val_predictions * y_std + y_mean
        val_targets_denorm = y_val

        # Compute NRMSE
        val_nrmse = compute_nrmse(val_predictions_denorm, val_targets_denorm, y_std)

        # Print progress
        avg_train_loss = np.mean(train_losses)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val NRMSE: {val_nrmse:.4f} | α: {alpha:.3f}")

        # Print detailed losses
        detail_str = " | ".join([f"{k}: {np.mean(v):.4f}" for k, v in train_loss_details.items() if v])
        print(f"  Details: {detail_str}")

        # Save best model
        if val_nrmse < best_val_nrmse:
            best_val_nrmse = val_nrmse
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_nrmse': val_nrmse,
                'y_mean': y_mean,
                'y_std': y_std,
                'num_subjects': num_subjects,
                'challenge': challenge
            }
            torch.save(checkpoint, save_dir / f'domain_adaptation_{challenge}_best.pt')
            print(f"  ✓ Saved best model (NRMSE: {val_nrmse:.4f})")

    print(f"\nTraining completed! Best validation NRMSE: {best_val_nrmse:.4f}")
    return best_val_nrmse


def main():
    parser = argparse.ArgumentParser(description='Train Domain Adaptation EEGNeX')
    parser.add_argument('--challenge', type=str, default='c1', choices=['c1', 'c2'],
                       help='Challenge name')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--lambda_mmd', type=float, default=0.1,
                       help='Weight for MMD loss')
    parser.add_argument('--lambda_entropy', type=float, default=0.01,
                       help='Weight for entropy loss')
    parser.add_argument('--lambda_adv', type=float, default=0.1,
                       help='Weight for adversarial loss')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')

    args = parser.parse_args()

    train_domain_adaptation(
        challenge=args.challenge,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_mmd=args.lambda_mmd,
        lambda_entropy=args.lambda_entropy,
        lambda_adv=args.lambda_adv,
        device=args.device,
        save_dir=args.save_dir
    )


if __name__ == '__main__':
    main()
