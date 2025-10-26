#!/usr/bin/env python3
"""
DEBUG-ENHANCED C2 TRAINING - Recording-Level with Comprehensive Logging

Adds detailed debugging/analysis to understand:
1. Per-subject prediction quality
2. Cross-subject generalization patterns
3. Model predictions vs target distribution
4. Domain adaptation effectiveness
5. Subject-level error patterns
6. Feature distributions

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

from data.official_eegdash_loader import create_official_eegdash_loaders
from models.domain_adaptation_eegnex import DomainAdaptationEEGNeX


def compute_nrmse(predictions, targets):
    """Compute NRMSE"""
    std = np.std(targets)
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    nrmse = rmse / std
    return nrmse


def analyze_predictions_c2(predictions, targets, phase='train'):
    """
    Detailed C2 prediction analysis

    Focus on subject-level patterns and cross-subject generalization
    """
    predictions = predictions if isinstance(predictions, np.ndarray) else predictions.detach().cpu().numpy()
    targets = targets if isinstance(targets, np.ndarray) else targets.detach().cpu().numpy()

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
        'rmse': float(np.sqrt(((predictions - targets) ** 2).mean())),
        'max_error': float(np.abs(predictions - targets).max()),

        # Correlation (key for subject-level prediction)
        'correlation': float(np.corrcoef(predictions, targets)[0, 1]) if len(predictions) > 1 else 0.0,

        # Distribution matching (should match target distribution)
        'pred_hist': np.histogram(predictions, bins=10, range=(-3, 3))[0].tolist(),
        'target_hist': np.histogram(targets, bins=10, range=(-3, 3))[0].tolist(),

        # Subject-level error patterns
        'high_error_ratio': float((np.abs(predictions - targets) > 1.0).mean()),  # Large errors
        'bias': float((predictions - targets).mean()),  # Systematic bias

        # Check if predictions are diverse (not collapsed)
        'pred_unique_ratio': float(len(np.unique(np.round(predictions, 2))) / len(predictions)),
        'pred_range': float(predictions.max() - predictions.min()),
    }

    return analysis


def train_domain_adaptation_c2_debug(
    challenge='c2',
    epochs=100,
    batch_size=64,
    lr=0.001,
    mini=False,
    lambda_mmd=0.1,
    lambda_entropy=0.01,
    lambda_adv=0.1,
    save_dir='./checkpoints',
    debug_dir='./debug_logs',
    device='cuda'
):
    """
    Train C2 domain adaptation with comprehensive debugging
    """
    print("="*80)
    print("DEBUG-ENHANCED C2 TRAINING - Domain Adaptation")
    print("="*80)

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Create debug directory
    debug_path = Path(debug_dir)
    debug_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_log = debug_path / f'c2_domain_adapt_{timestamp}.json'

    # Initialize debug log
    debug_data = {
        'config': {
            'challenge': challenge,
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr,
            'mini': mini,
            'lambda_mmd': lambda_mmd,
            'lambda_entropy': lambda_entropy,
            'lambda_adv': lambda_adv,
            'timestamp': timestamp
        },
        'epochs': []
    }

    # Load data
    print("Loading C2 data...")
    train_loader, val_loader = create_official_eegdash_loaders(
        challenge=challenge,
        batch_size=batch_size,
        val_split=0.2,
        mini=mini
    )

    print(f"\n📊 Dataset Statistics:")
    print(f"   Train recordings: {len(train_loader.dataset)}")
    print(f"   Val recordings: {len(val_loader.dataset)}")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")

    # Analyze target distribution
    print("\nAnalyzing target distribution...")
    all_targets = []
    for _, y in train_loader:
        all_targets.append(y.numpy())
    all_targets = np.concatenate(all_targets)

    target_stats = {
        'mean': float(all_targets.mean()),
        'std': float(all_targets.std()),
        'min': float(all_targets.min()),
        'max': float(all_targets.max()),
        'median': float(np.median(all_targets))
    }

    print(f"   Target mean: {target_stats['mean']:.4f}")
    print(f"   Target std: {target_stats['std']:.4f}")
    print(f"   Target range: [{target_stats['min']:.4f}, {target_stats['max']:.4f}]")

    # Check if standardized
    if abs(target_stats['mean']) < 0.5 and abs(target_stats['std'] - 1.0) < 0.5:
        print(f"   ✅ Targets appear STANDARDIZED (mean~0, std~1)")
        print(f"   ✅ Model output_range=(-3, 3) is appropriate")
    else:
        print(f"   ⚠️ Targets may NOT be standardized!")
        print(f"   ⚠️ Model output_range=(-3, 3) might need adjustment!")

    debug_data['target_statistics'] = target_stats

    # Create model
    print("\nCreating model...")
    model = DomainAdaptationEEGNeX(
        n_channels=129,
        n_times=200,
        challenge=challenge
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {total_params:,}")
    print(f"   Output range: {model.output_range}")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

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
            'lr': optimizer.param_groups[0]['lr'],
            'alpha': min(epoch / 50, 1.0)  # Gradually increase domain adaptation
        }

        alpha = epoch_debug['alpha']

        # Training
        model.train()
        train_losses = {'total': 0.0, 'task': 0.0, 'mmd': 0.0, 'entropy': 0.0, 'adv': 0.0}
        train_preds = []
        train_targets = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for X_batch, y_batch in pbar:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).squeeze(-1)

            optimizer.zero_grad()

            # Split batch for domain adaptation
            split = len(X_batch) // 2
            source_X = X_batch[:split]
            target_X = X_batch[split:]
            source_y = y_batch[:split]

            # Forward pass with domain adaptation
            predictions, source_features = model(source_X, alpha=alpha, return_features=True)
            _, target_features = model(target_X, alpha=alpha, return_features=True)

            # Task loss
            task_loss = nn.MSELoss()(predictions, source_y)

            # MMD loss (align distributions)
            from models.domain_adaptation_eegnex import compute_mmd_loss
            mmd_loss = compute_mmd_loss(source_features, target_features)

            # Entropy loss (confident predictions)
            target_preds, _ = model(target_X, alpha=alpha, return_features=True)
            target_probs = torch.sigmoid(target_preds)
            entropy_loss = -(target_probs * torch.log(target_probs + 1e-8) +
                            (1 - target_probs) * torch.log(1 - target_probs + 1e-8)).mean()

            # Total loss
            total_loss = (task_loss +
                         lambda_mmd * mmd_loss +
                         lambda_entropy * entropy_loss)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses['total'] += total_loss.item()
            train_losses['task'] += task_loss.item()
            train_losses['mmd'] += mmd_loss.item()
            train_losses['entropy'] += entropy_loss.item()

            train_preds.append(predictions.detach().cpu())
            train_targets.append(source_y.detach().cpu())

            pbar.set_postfix({'loss': f'{total_loss.item():.4f}'})

        for key in train_losses:
            train_losses[key] /= len(train_loader)

        train_preds = torch.cat(train_preds).numpy()
        train_targets = torch.cat(train_targets).numpy()
        train_nrmse = compute_nrmse(train_preds, train_targets)

        # Validation
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device).squeeze(-1)

                predictions = model(X_batch, alpha=0.0)  # No domain adaptation in val
                val_preds.append(predictions.cpu())
                val_targets.append(y_batch.cpu())

        val_preds = torch.cat(val_preds).numpy()
        val_targets = torch.cat(val_targets).numpy()
        val_nrmse = compute_nrmse(val_preds, val_targets)

        # Detailed analysis
        train_analysis = analyze_predictions_c2(train_preds, train_targets, 'train')
        val_analysis = analyze_predictions_c2(val_preds, val_targets, 'val')

        epoch_debug['train'] = {
            'losses': train_losses,
            'nrmse': float(train_nrmse),
            'analysis': train_analysis
        }
        epoch_debug['val'] = {
            'nrmse': float(val_nrmse),
            'analysis': val_analysis
        }

        debug_data['epochs'].append(epoch_debug)

        # Print epoch results
        print(f"\nEpoch {epoch+1}/{epochs} | α={alpha:.3f}")
        print(f"  Train: NRMSE={train_nrmse:.4f}, Corr={train_analysis['correlation']:.3f}")
        print(f"    Losses: Task={train_losses['task']:.4f}, MMD={train_losses['mmd']:.4f}, Ent={train_losses['entropy']:.4f}")
        print(f"  Val:   NRMSE={val_nrmse:.4f}, Corr={val_analysis['correlation']:.3f}")
        print(f"  📊 Val pred range: [{val_analysis['pred_min']:.2f}, {val_analysis['pred_max']:.2f}]")
        print(f"  📊 Val pred diversity: {val_analysis['pred_unique_ratio']*100:.1f}%")
        print(f"  ⚠️ Val high errors: {val_analysis['high_error_ratio']*100:.1f}%")

        # Check for prediction collapse
        if val_analysis['pred_range'] < 0.5:
            print(f"  ⚠️ WARNING: Predictions collapsed! Range only {val_analysis['pred_range']:.3f}")

        # Save best model
        if val_nrmse < best_nrmse:
            best_nrmse = val_nrmse
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_nrmse': best_nrmse,
                'debug_info': epoch_debug
            }
            torch.save(checkpoint, save_path / f'domain_adaptation_{challenge}_best.pt')
            print(f"  ✓ Saved best model (NRMSE: {best_nrmse:.4f})")

        # Save debug log every 10 epochs
        if (epoch + 1) % 10 == 0:
            with open(debug_log, 'w') as f:
                json.dump(debug_data, f, indent=2)
            print(f"  💾 Debug log saved")

        scheduler.step(val_nrmse)

    # Final debug log
    debug_data['final_results'] = {
        'best_val_nrmse': float(best_nrmse),
        'total_epochs': epochs
    }

    with open(debug_log, 'w') as f:
        json.dump(debug_data, f, indent=2)

    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Best Val NRMSE: {best_nrmse:.4f}")
    print(f"Model saved: {save_path / f'domain_adaptation_{challenge}_best.pt'}")
    print(f"Debug log saved: {debug_log}")

    return model, best_nrmse, debug_log


def main():
    parser = argparse.ArgumentParser(description='Train C2 with Debug Logging')
    parser.add_argument('--challenge', type=str, default='c2', choices=['c2'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--mini', action='store_true')
    parser.add_argument('--lambda_mmd', type=float, default=0.1)
    parser.add_argument('--lambda_entropy', type=float, default=0.01)
    parser.add_argument('--lambda_adv', type=float, default=0.1)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--debug_dir', type=str, default='./debug_logs')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    train_domain_adaptation_c2_debug(
        challenge=args.challenge,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        mini=args.mini,
        lambda_mmd=args.lambda_mmd,
        lambda_entropy=args.lambda_entropy,
        lambda_adv=args.lambda_adv,
        save_dir=args.save_dir,
        debug_dir=args.debug_dir,
        device=args.device
    )


if __name__ == '__main__':
    main()
