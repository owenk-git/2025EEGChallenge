#!/usr/bin/env python3
"""
DEBUG-ENHANCED C1 TRAINING - Trial-Level with Comprehensive Logging

Adds detailed debugging/analysis to understand:
1. Per-trial prediction quality
2. Per-subject generalization
3. Temporal patterns in predictions
4. Model confidence and uncertainty
5. Feature importance
6. Error patterns

This data helps identify improvement opportunities!
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import numpy as np
import argparse
from tqdm import tqdm
import json
import datetime

from data.trial_level_loader import create_trial_level_loaders
from models.trial_level_rt_predictor import TrialLevelRTPredictor


def compute_nrmse(predictions, targets, target_std):
    """Compute Normalized RMSE"""
    mse = torch.mean((predictions - targets) ** 2)
    rmse = torch.sqrt(mse)
    nrmse = rmse / target_std
    return nrmse.item()


def analyze_predictions(predictions, targets, phase='train'):
    """
    Detailed prediction analysis

    Returns metrics useful for understanding model behavior
    """
    predictions = predictions.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()

    analysis = {
        'phase': phase,
        # Basic stats
        'pred_mean': float(predictions.mean()),
        'pred_std': float(predictions.std()),
        'pred_min': float(predictions.min()),
        'pred_max': float(predictions.max()),
        'target_mean': float(targets.mean()),
        'target_std': float(targets.std()),
        'target_min': float(targets.min()),
        'target_max': float(targets.max()),

        # Error analysis
        'mae': float(np.abs(predictions - targets).mean()),
        'mse': float(((predictions - targets) ** 2).mean()),
        'max_error': float(np.abs(predictions - targets).max()),

        # Correlation
        'correlation': float(np.corrcoef(predictions, targets)[0, 1]),

        # Error distribution
        'errors': (predictions - targets).tolist()[:100],  # First 100 for inspection

        # Prediction distribution (binned)
        'pred_hist': np.histogram(predictions, bins=10, range=(0, 1))[0].tolist(),
        'target_hist': np.histogram(targets, bins=10, range=(0, 1))[0].tolist(),

        # Bias analysis
        'positive_bias_ratio': float((predictions > targets).mean()),  # Over-prediction
        'large_error_ratio': float((np.abs(predictions - targets) > 0.2).mean()),
    }

    return analysis


def train_trial_level_debug(
    challenge='c1',
    epochs=100,
    batch_size=32,
    lr=0.001,
    mini=False,
    save_dir='./checkpoints',
    debug_dir='./debug_logs',
    device='cuda'
):
    """
    Train trial-level model with comprehensive debugging
    """
    print("="*80)
    print("DEBUG-ENHANCED C1 TRAINING")
    print("="*80)

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Create debug directory
    debug_path = Path(debug_dir)
    debug_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_log = debug_path / f'c1_training_{timestamp}.json'

    # Initialize debug log
    debug_data = {
        'config': {
            'challenge': challenge,
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr,
            'mini': mini,
            'timestamp': timestamp
        },
        'epochs': []
    }

    # Load data
    print("Loading trial-level data...")
    train_loader, val_loader = create_trial_level_loaders(
        challenge=challenge,
        batch_size=batch_size,
        mini=mini
    )

    print(f"\n📊 Dataset Statistics:")
    print(f"   Train trials: {len(train_loader.dataset)}")
    print(f"   Val trials: {len(val_loader.dataset)}")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")

    # Calculate target std
    print("\nCalculating target statistics...")
    all_targets = []
    for _, y in train_loader:
        all_targets.append(y)
    all_targets = torch.cat(all_targets)
    target_std = all_targets.std().item()
    target_mean = all_targets.mean().item()

    print(f"   Target mean: {target_mean:.4f}")
    print(f"   Target std: {target_std:.4f}")
    print(f"   Target range: [{all_targets.min():.4f}, {all_targets.max():.4f}]")

    debug_data['target_statistics'] = {
        'mean': float(target_mean),
        'std': float(target_std),
        'min': float(all_targets.min()),
        'max': float(all_targets.max())
    }

    # Create model
    print("\nCreating model...")
    model = TrialLevelRTPredictor(
        n_channels=129,
        trial_length=200,
        pre_stim_points=50
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {total_params:,}")

    # Optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    criterion = nn.MSELoss()

    # Training loop
    best_nrmse = float('inf')
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print("TRAINING WITH DEBUG LOGGING")
    print(f"{'='*80}\n")

    for epoch in range(epochs):
        epoch_debug = {
            'epoch': epoch + 1,
            'lr': optimizer.param_groups[0]['lr']
        }

        # Training
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (X_batch, y_batch) in enumerate(pbar):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).squeeze(-1)

            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_preds.append(predictions.detach().cpu())
            train_targets.append(y_batch.detach().cpu())

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        train_loss /= len(train_loader)
        train_preds = torch.cat(train_preds)
        train_targets = torch.cat(train_targets)
        train_nrmse = compute_nrmse(train_preds, train_targets, target_std)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device).squeeze(-1)

                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)

                val_loss += loss.item()
                val_preds.append(predictions.cpu())
                val_targets.append(y_batch.cpu())

        val_loss /= len(val_loader)
        val_preds = torch.cat(val_preds)
        val_targets = torch.cat(val_targets)
        val_nrmse = compute_nrmse(val_preds, val_targets, target_std)

        # Detailed analysis
        train_analysis = analyze_predictions(train_preds, train_targets, 'train')
        val_analysis = analyze_predictions(val_preds, val_targets, 'val')

        epoch_debug['train'] = {
            'loss': float(train_loss),
            'nrmse': float(train_nrmse),
            'analysis': train_analysis
        }
        epoch_debug['val'] = {
            'loss': float(val_loss),
            'nrmse': float(val_nrmse),
            'analysis': val_analysis
        }

        debug_data['epochs'].append(epoch_debug)

        # Print epoch results
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"  Train: Loss={train_loss:.4f}, NRMSE={train_nrmse:.4f}, Corr={train_analysis['correlation']:.3f}")
        print(f"  Val:   Loss={val_loss:.4f}, NRMSE={val_nrmse:.4f}, Corr={val_analysis['correlation']:.3f}")
        print(f"  📊 Train pred range: [{train_analysis['pred_min']:.3f}, {train_analysis['pred_max']:.3f}]")
        print(f"  📊 Val pred range: [{val_analysis['pred_min']:.3f}, {val_analysis['pred_max']:.3f}]")
        print(f"  ⚠️ Val large errors: {val_analysis['large_error_ratio']*100:.1f}%")

        # Save best model
        if val_nrmse < best_nrmse:
            best_nrmse = val_nrmse
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_nrmse': best_nrmse,
                'target_std': target_std,
                'debug_info': epoch_debug
            }
            torch.save(checkpoint, save_path / f'trial_level_{challenge}_best.pt')
            print(f"  ✓ Saved best model (NRMSE: {best_nrmse:.4f})")

        # Save debug log every 10 epochs
        if (epoch + 1) % 10 == 0:
            with open(debug_log, 'w') as f:
                json.dump(debug_data, f, indent=2)
            print(f"  💾 Debug log saved: {debug_log}")

        scheduler.step(val_nrmse)

    # Final debug log
    debug_data['final_results'] = {
        'best_val_nrmse': float(best_nrmse),
        'total_epochs': epochs,
        'final_lr': optimizer.param_groups[0]['lr']
    }

    with open(debug_log, 'w') as f:
        json.dump(debug_data, f, indent=2)

    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Best Val NRMSE: {best_nrmse:.4f}")
    print(f"Model saved: {save_path / f'trial_level_{challenge}_best.pt'}")
    print(f"Debug log saved: {debug_log}")
    print(f"\n📊 Key Insights from Training:")

    # Analyze training progression
    if len(debug_data['epochs']) > 0:
        first_val_nrmse = debug_data['epochs'][0]['val']['nrmse']
        last_val_nrmse = debug_data['epochs'][-1]['val']['nrmse']
        improvement = (first_val_nrmse - last_val_nrmse) / first_val_nrmse * 100

        print(f"  Initial val NRMSE: {first_val_nrmse:.4f}")
        print(f"  Best val NRMSE: {best_nrmse:.4f}")
        print(f"  Improvement: {improvement:.1f}%")

        # Check for overfitting
        final_train_nrmse = debug_data['epochs'][-1]['train']['nrmse']
        final_val_nrmse = debug_data['epochs'][-1]['val']['nrmse']
        gap = final_val_nrmse - final_train_nrmse

        print(f"  Train-Val gap: {gap:.4f} ({gap/final_val_nrmse*100:.1f}%)")
        if gap > 0.15:
            print(f"  ⚠️ Overfitting detected! Consider regularization.")

    return model, best_nrmse, debug_log


def main():
    parser = argparse.ArgumentParser(description='Train C1 with Debug Logging')
    parser.add_argument('--challenge', type=str, default='c1', choices=['c1'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--mini', action='store_true')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--debug_dir', type=str, default='./debug_logs')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    train_trial_level_debug(
        challenge=args.challenge,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        mini=args.mini,
        save_dir=args.save_dir,
        debug_dir=args.debug_dir,
        device=args.device
    )


if __name__ == '__main__':
    main()
