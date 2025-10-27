"""
Train ALL Model Architectures with K-Fold Cross-Validation

Trains multiple model architectures (ERP MLP, CNN Ensemble, EEGNeX, etc.)
on each fold and selects the best performing model per fold.

Strategy:
- For each fold (1-5):
  - Train ERP MLP
  - Train CNN Ensemble
  - Train EEGNeX Improved
  - Select best model based on validation NRMSE
- Save best model per fold
- Create ensemble of best models (may be different architectures)

Usage:
    # Train all models with 5-fold CV for C1
    python3 train_all_models_kfold.py --c 1 --f 5 --e 30

    # Train all models with 3-fold CV for C2
    python3 train_all_models_kfold.py --c 2 --f 3 --e 30
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
from tqdm import tqdm
import json
from datetime import datetime

# Import models
from models.erp_features_mlp import ERPMLP
from models.cnn_ensemble import CNNEnsemble
from models.eegnex_augmented import EEGNeXImproved

# Import dataset
from data.official_eegdash_loader import OfficialEEGDashDataset


# Model registry
MODEL_REGISTRY = {
    'erp_mlp': {
        'class': ERPMLP,
        'params': lambda challenge: {
            'n_channels': 129,
            'sfreq': 100,
            'challenge_name': challenge,
            'output_range': (0.5, 1.5) if challenge == 'c1' else (-3, 3)
        }
    },
    'cnn_ensemble': {
        'class': CNNEnsemble,
        'params': lambda challenge: {
            'n_channels': 129,
            'n_times': 200,
            'challenge_name': challenge
        }
    },
    'eegnex_improved': {
        'class': EEGNeXImproved,
        'params': lambda challenge: {
            'n_channels': 129,
            'n_times': 200,
            'challenge_name': challenge
        }
    }
}


class RecordingLevelKFold:
    """K-Fold cross-validation at recording level"""

    def __init__(self, dataset, n_splits=5, seed=42):
        self.dataset = dataset
        self.n_splits = n_splits
        self.seed = seed
        self.recording_indices = list(range(len(dataset)))
        np.random.seed(seed)
        np.random.shuffle(self.recording_indices)
        self.kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    def split(self):
        """Generate K-fold splits"""
        for fold_idx, (train_idx, val_idx) in enumerate(self.kfold.split(self.recording_indices)):
            train_recordings = [self.recording_indices[i] for i in train_idx]
            val_recordings = [self.recording_indices[i] for i in val_idx]
            yield fold_idx, train_recordings, val_recordings


def create_model(model_name, challenge, device):
    """Create model instance"""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")

    model_info = MODEL_REGISTRY[model_name]
    model_class = model_info['class']
    model_params = model_info['params'](challenge)

    model = model_class(**model_params)
    return model.to(device)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for data, target in tqdm(loader, desc='Training', leave=False):
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.size(0)
        all_preds.append(output.detach().cpu())
        all_targets.append(target.detach().cpu())

    avg_loss = total_loss / len(loader.dataset)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    mse = ((all_preds - all_targets) ** 2).mean().item()
    target_std = all_targets.std().item()
    rmse = np.sqrt(mse)
    nrmse = rmse / (target_std + 1e-8)

    pred_centered = all_preds - all_preds.mean()
    target_centered = all_targets - all_targets.mean()
    numerator = (pred_centered * target_centered).sum()
    denominator = torch.sqrt((pred_centered ** 2).sum() * (target_centered ** 2).sum())
    correlation = (numerator / (denominator + 1e-8)).item()

    return {
        'loss': avg_loss,
        'mse': mse,
        'rmse': rmse,
        'nrmse': nrmse,
        'correlation': correlation,
        'pred_std': all_preds.std().item(),
        'target_std': target_std
    }


def validate(model, loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in tqdm(loader, desc='Validating', leave=False):
            data = data.to(device)
            target = target.to(device)

            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item() * data.size(0)
            all_preds.append(output.cpu())
            all_targets.append(target.cpu())

    avg_loss = total_loss / len(loader.dataset)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    mse = ((all_preds - all_targets) ** 2).mean().item()
    target_std = all_targets.std().item()
    rmse = np.sqrt(mse)
    nrmse = rmse / (target_std + 1e-8)

    pred_centered = all_preds - all_preds.mean()
    target_centered = all_targets - all_targets.mean()
    numerator = (pred_centered * target_centered).sum()
    denominator = torch.sqrt((pred_centered ** 2).sum() * (target_centered ** 2).sum())
    correlation = (numerator / (denominator + 1e-8)).item()

    return {
        'loss': avg_loss,
        'mse': mse,
        'rmse': rmse,
        'nrmse': nrmse,
        'correlation': correlation,
        'pred_std': all_preds.std().item(),
        'target_std': target_std
    }


def train_single_model(
    model_name,
    fold_idx,
    train_loader,
    val_loader,
    challenge,
    epochs,
    lr,
    device,
    checkpoint_dir
):
    """Train a single model architecture for one fold"""
    print(f"\n  Training {model_name}...")

    model = create_model(model_name, challenge, device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    best_val_nrmse = float('inf')
    best_epoch = 0
    patience_counter = 0
    max_patience = 10

    for epoch in range(epochs):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step(val_metrics['nrmse'])

        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"    Epoch {epoch+1}/{epochs} - Val NRMSE: {val_metrics['nrmse']:.4f}, "
                  f"Corr: {val_metrics['correlation']:.4f}")

        # Save best model
        if val_metrics['nrmse'] < best_val_nrmse:
            best_val_nrmse = val_metrics['nrmse']
            best_epoch = epoch + 1
            patience_counter = 0

            checkpoint_path = checkpoint_dir / f"fold_{fold_idx}_{model_name}_best.pth"
            torch.save({
                'fold': fold_idx,
                'model_name': model_name,
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_nrmse': val_metrics['nrmse'],
                'val_correlation': val_metrics['correlation'],
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            }, checkpoint_path)
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= max_patience:
            print(f"    Early stopping at epoch {epoch + 1}")
            break

    print(f"  ✅ {model_name}: Best Val NRMSE = {best_val_nrmse:.4f} (epoch {best_epoch})")

    return {
        'model_name': model_name,
        'best_val_nrmse': best_val_nrmse,
        'best_epoch': best_epoch
    }


def train_fold_all_models(
    fold_idx,
    train_indices,
    val_indices,
    full_dataset,
    challenge,
    epochs,
    batch_size,
    lr,
    device,
    checkpoint_dir
):
    """Train all model architectures for one fold and select best"""
    print(f"\n{'='*60}")
    print(f"FOLD {fold_idx + 1}")
    print(f"{'='*60}")
    print(f"Train recordings: {len(train_indices)}")
    print(f"Val recordings: {len(val_indices)}")

    # Create data loaders
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Train all model architectures
    model_results = []

    for model_name in MODEL_REGISTRY.keys():
        try:
            result = train_single_model(
                model_name=model_name,
                fold_idx=fold_idx,
                train_loader=train_loader,
                val_loader=val_loader,
                challenge=challenge,
                epochs=epochs,
                lr=lr,
                device=device,
                checkpoint_dir=checkpoint_dir
            )
            model_results.append(result)
        except Exception as e:
            print(f"  ❌ {model_name} failed: {e}")
            continue

    # Select best model
    best_result = min(model_results, key=lambda x: x['best_val_nrmse'])
    best_model_name = best_result['model_name']
    best_nrmse = best_result['best_val_nrmse']

    print(f"\n🏆 FOLD {fold_idx + 1} BEST: {best_model_name} (NRMSE: {best_nrmse:.4f})")

    # Copy best model to standard location
    src = checkpoint_dir / f"fold_{fold_idx}_{best_model_name}_best.pth"
    dst = checkpoint_dir / f"fold_{fold_idx}_best.pth"
    if src.exists():
        import shutil
        shutil.copy(src, dst)

    return {
        'fold': fold_idx,
        'best_model': best_model_name,
        'best_val_nrmse': best_nrmse,
        'all_models': model_results
    }


def main():
    parser = argparse.ArgumentParser(description='Train All Models with K-Fold')
    parser.add_argument('--challenge', '--c', type=str, default='c1',
                       choices=['c1', 'c2', '1', '2'],
                       help='Challenge: 1 or c1, 2 or c2')
    parser.add_argument('--n_folds', '--f', type=int, default=5,
                       help='Number of folds')
    parser.add_argument('--epochs', '--e', type=int, default=30,
                       help='Max epochs per model')
    parser.add_argument('--batch_size', '--b', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--mini', action='store_true',
                       help='Use mini dataset')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')

    args = parser.parse_args()

    # Map challenge shortcuts
    if args.challenge in ['1', '2']:
        args.challenge = f'c{args.challenge}'

    print("="*60)
    print("TRAIN ALL MODELS WITH K-FOLD")
    print("="*60)
    print(f"Challenge: {args.challenge}")
    print(f"Models: {', '.join(MODEL_REGISTRY.keys())}")
    print(f"N-Folds: {args.n_folds}")
    print(f"Epochs per model: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print(f"Device: {args.device}")

    # Create checkpoint directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    checkpoint_dir = Path(f"checkpoints_all_models_{args.challenge}_{timestamp}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📁 Checkpoints: {checkpoint_dir}")

    # Load dataset
    print(f"\n📦 Loading dataset...")
    full_dataset = OfficialEEGDashDataset(
        task="contrastChangeDetection",
        challenge=args.challenge,
        release="R11",
        mini=args.mini,
        rt_method='mean'
    )

    print(f"✅ Loaded {len(full_dataset)} recordings")

    # Create K-fold splitter
    kfold = RecordingLevelKFold(full_dataset, n_splits=args.n_folds, seed=42)

    # Train each fold with all models
    fold_results = []

    for fold_idx, train_indices, val_indices in kfold.split():
        fold_result = train_fold_all_models(
            fold_idx=fold_idx,
            train_indices=train_indices,
            val_indices=val_indices,
            full_dataset=full_dataset,
            challenge=args.challenge,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            checkpoint_dir=checkpoint_dir
        )
        fold_results.append(fold_result)

    # Summary
    print("\n" + "="*60)
    print("BEST MODEL PER FOLD")
    print("="*60)

    for result in fold_results:
        fold_idx = result['fold']
        best_model = result['best_model']
        best_nrmse = result['best_val_nrmse']
        print(f"Fold {fold_idx + 1}: {best_model:20s} (NRMSE: {best_nrmse:.4f})")

    # Overall statistics
    nrmses = [r['best_val_nrmse'] for r in fold_results]
    avg_nrmse = np.mean(nrmses)
    std_nrmse = np.std(nrmses)

    print(f"\nAverage Val NRMSE: {avg_nrmse:.4f} ± {std_nrmse:.4f}")

    # Save summary
    summary = {
        'challenge': args.challenge,
        'n_folds': args.n_folds,
        'models_trained': list(MODEL_REGISTRY.keys()),
        'fold_results': fold_results,
        'avg_nrmse': avg_nrmse,
        'std_nrmse': std_nrmse,
        'timestamp': timestamp
    }

    summary_path = checkpoint_dir / 'best_per_fold_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Summary saved to {summary_path}")
    print(f"\n📝 Next step: Create ensemble submission with:")
    print(f"   python3 create_mixed_ensemble_submission.py \\")
    for fold_idx in range(args.n_folds):
        print(f"     --c{args.challenge[-1]} {checkpoint_dir}/fold_{fold_idx}_best.pth \\")
    print(f"     --name best_per_fold_ensemble")


if __name__ == '__main__':
    main()
