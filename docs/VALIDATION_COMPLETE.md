# ✅ Validation System - COMPLETE

All validation features have been implemented based on the ultrathink analysis!

## 🎯 What Was Fixed

### Before (Problems):
- ❌ No validation split - training on 100% of data
- ❌ Using MSE loss only - not the competition metric (NRMSE)
- ❌ Saving based on training loss - can overfit!
- ❌ No predictions saved - can't analyze errors
- ❌ Don't know performance until submission

### After (Solutions):
- ✅ **80/20 train/val split** - proper held-out evaluation
- ✅ **Validation NRMSE computed** - competition metric tracked!
- ✅ **Save best model by val NRMSE** - prevents overfitting
- ✅ **Predictions saved** - enable error analysis
- ✅ **Know performance before submission** - validation metrics

---

## 📋 Implementation Details

### 1. Metrics Module ([utils/metrics.py](../utils/metrics.py))

```python
from utils.metrics import compute_all_metrics, normalized_rmse

# Competition metric
nrmse = normalized_rmse(predictions, targets)
# nrmse = sqrt(mean((pred - target)²)) / std(target)

# All metrics at once
metrics = compute_all_metrics(predictions, targets)
# Returns: {'nrmse': X, 'rmse': Y, 'mae': Z}
```

**Functions:**
- `normalized_rmse()` - **Competition metric!**
- `rmse()` - Root Mean Squared Error
- `mae()` - Mean Absolute Error
- `compute_all_metrics()` - All at once
- `combined_challenge_score()` - Final: 0.3×C1 + 0.7×C2

### 2. Train/Val Split ([data/official_dataset_example.py](../data/official_dataset_example.py))

```python
train_loader, val_loader = create_official_dataloaders_with_split(
    task="contrastChangeDetection",
    challenge='c1',
    batch_size=32,
    mini=True,
    max_subjects=100,
    num_workers=4,
    val_split=0.2,      # 20% validation
    random_seed=42       # Reproducible
)
```

**Features:**
- Default 80/20 split
- Reproducible with seed
- Returns both loaders
- No data leakage

### 3. Validation with Metrics ([train.py](../train.py))

```python
val_loss, val_metrics, predictions, targets = validate_with_metrics(
    model, val_loader, criterion, device
)

print(f"Val NRMSE: {val_metrics['nrmse']:.4f} ⭐")
print(f"Val RMSE:  {val_metrics['rmse']:.4f}")
print(f"Val MAE:   {val_metrics['mae']:.4f}")
```

**Returns:**
- `val_loss` - MSE loss
- `val_metrics` - Dict with NRMSE, RMSE, MAE
- `predictions` - All predictions (for saving)
- `targets` - All targets (for analysis)

### 4. Updated Training Loop

**Key Changes:**
```python
# 1. Load with split
if not args.no_val:
    train_loader, val_loader = create_official_dataloaders_with_split(...)
else:
    train_loader = create_official_dataloader(...)
    val_loader = None

# 2. Track best val NRMSE (not train loss!)
best_nrmse = float('inf')
best_metrics = {}

# 3. Validate each epoch
for epoch in range(1, args.epochs + 1):
    train_loss = train_epoch(...)

    if val_loader is not None:
        val_loss, val_metrics, predictions, targets = validate_with_metrics(...)

        # Save if NRMSE improved
        if val_metrics['nrmse'] < best_nrmse:
            best_nrmse = val_metrics['nrmse']
            best_metrics = val_metrics
            best_predictions = predictions
            best_targets = targets
            # Save checkpoint...

# 4. Save predictions
if val_loader and best_predictions:
    torch.save({
        'predictions': best_predictions,
        'targets': best_targets,
        'metrics': best_metrics,
        'best_epoch': best_epoch,
        'config': vars(args),
    }, f'results/exp_{exp_num}/c{challenge}_results.pt')

# 5. Log with metrics
log_experiment(args, best_metrics, best_epoch)
```

### 5. New Arguments

```bash
# Validation split (default: 20%)
--val_split 0.2

# Disable validation (train on all data)
--no_val
```

### 6. Experiment Logging

**JSON Output** (`experiments/experiments.json`):
```json
{
  "exp_num": 1,
  "timestamp": "2025-10-20 14:30:00",
  "challenge": 1,
  "config": {
    "epochs": 100,
    "batch_size": 32,
    "lr": 0.001,
    "dropout": 0.2,
    "max_subjects": 100,
    "use_official": true,
    "official_mini": false
  },
  "results": {
    "best_val_nrmse": 1.2345,
    "best_val_rmse": 0.0456,
    "best_val_mae": 0.0321,
    "best_epoch": 67
  }
}
```

---

## 🚀 Usage

### Basic Training (with validation)
```bash
CUDA_VISIBLE_DEVICES=1 python train.py \
  -c 1 \
  -d dummy \
  -o \
  --max 100 \
  -e 100 \
  --num 1
```

**Output during training:**
```
Epoch 1/100
  Train Loss: 0.1234
  Val Loss:  0.1456
  Val NRMSE: 1.2345 ⭐ (Competition Metric)
  Val RMSE:  0.0456
  Val MAE:   0.0321
  ✅ Saved best model (NRMSE: 1.2345)

Epoch 2/100
  Train Loss: 0.1100
  Val Loss:  0.1350
  Val NRMSE: 1.1987 ⭐ (Competition Metric)
  Val RMSE:  0.0421
  Val MAE:   0.0298
  ✅ Saved best model (NRMSE: 1.1987)
...
```

**Final summary:**
```
======================================================================
✅ Training complete! Best val NRMSE: 1.0543 (epoch 67)
   Best val RMSE: 0.0389
   Best val MAE:  0.0267
📁 Model saved to: checkpoints/c1_best.pth
======================================================================

💾 Saved predictions and metrics to: results/exp_1/c1_results.pt

✅ Experiment #1 logged to experiments/experiments.json
```

### Quick Test (5 subjects, 3 epochs)
```bash
python train.py -c 1 -d dummy -o -m --max 5 -e 3 --num 999
```

### Custom Validation Split
```bash
# 90/10 split (more training data)
python train.py -c 1 -d dummy -o --max 100 -e 100 --val_split 0.1 --num 1

# 70/30 split (more validation data)
python train.py -c 1 -d dummy -o --max 100 -e 100 --val_split 0.3 --num 1
```

### No Validation (train on all data)
```bash
# For final submission training
python train.py -c 1 -d dummy -o --max 100 -e 100 --no_val --num 1
```

---

## 📊 Analyzing Results

### 1. View Experiments
```bash
python experiments/analyze_experiments.py
```

**Output:**
```
====================================================================================================
EXPERIMENT SUMMARY
====================================================================================================
Exp   Challenge  Subjects   Epochs   LR         Dropout    Val NRMSE    Best Epoch
----------------------------------------------------------------------------------------------------
1     C1         100        100      1.0e-03    0.20       1.0543       67
2     C2         100        100      1.0e-03    0.20       0.9876       82
====================================================================================================
```

### 2. Load Predictions for Analysis
```python
import torch

# Load results
results = torch.load('results/exp_1/c1_results.pt')

predictions = results['predictions']  # Tensor: (N, 1)
targets = results['targets']          # Tensor: (N, 1)
metrics = results['metrics']          # Dict: {'nrmse': X, 'rmse': Y, 'mae': Z}
config = results['config']            # Dict: all training arguments

# Analyze errors
errors = predictions - targets
print(f"Mean error: {errors.mean():.4f}")
print(f"Std error: {errors.std():.4f}")

# Find worst predictions
abs_errors = torch.abs(errors)
worst_idx = abs_errors.argmax()
print(f"Worst prediction: pred={predictions[worst_idx]:.4f}, target={targets[worst_idx]:.4f}")
```

### 3. Plot Results
```python
import matplotlib.pyplot as plt

# Load results
results = torch.load('results/exp_1/c1_results.pt')
predictions = results['predictions'].numpy()
targets = results['targets'].numpy()

# Scatter plot
plt.figure(figsize=(8, 8))
plt.scatter(targets, predictions, alpha=0.5)
plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', lw=2)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title(f"Challenge 1 - NRMSE: {results['metrics']['nrmse']:.4f}")
plt.tight_layout()
plt.savefig('results/exp_1/scatter.png')

# Error distribution
errors = predictions - targets
plt.figure(figsize=(10, 5))
plt.hist(errors, bins=50, edgecolor='black')
plt.xlabel('Prediction Error')
plt.ylabel('Count')
plt.title('Error Distribution')
plt.tight_layout()
plt.savefig('results/exp_1/error_dist.png')
```

---

## 🎯 Expected Impact

### Model Selection
- **Before:** Save based on training loss → overfitting
- **After:** Save based on validation NRMSE → generalization

### Performance Visibility
- **Before:** Blind until submission (25 submissions left!)
- **After:** Know performance immediately from validation

### Error Analysis
- **Before:** No predictions saved, can't debug
- **After:** All predictions saved, can find patterns

### Iterative Improvement
- **Before:** Submit → wait → see score → guess what to change
- **After:** Train → analyze val predictions → fix issues → train again

---

## 🔍 What to Look For

### Good Signs
- Val NRMSE decreasing over epochs
- Val NRMSE < 1.2 (better than current best 1.14)
- Train and val loss moving together (not diverging)
- Predictions centered around targets

### Warning Signs
- Val NRMSE increasing while train loss decreasing → overfitting
- Val NRMSE > 1.5 → model not learning
- Large gap between train and val → need regularization
- Predictions all similar → model collapsed

---

## 📈 Next Steps

1. **Quick Test:**
   ```bash
   python train.py -c 1 -d dummy -o -m --max 5 -e 3 --num 999
   ```

2. **Full Training C1:**
   ```bash
   CUDA_VISIBLE_DEVICES=1 python train.py -c 1 -d dummy -o --max 100 -e 100 --num 1
   ```

3. **Analyze Results:**
   ```bash
   python experiments/analyze_experiments.py
   ```

4. **Check Predictions:**
   ```python
   results = torch.load('results/exp_1/c1_results.pt')
   print(results['metrics'])
   ```

5. **Train C2:**
   ```bash
   CUDA_VISIBLE_DEVICES=1 python train.py -c 2 -d dummy -o --max 100 -e 100 --num 2
   ```

6. **Create Submission:**
   ```bash
   python create_submission.py \
     --model_c1 checkpoints/c1_best.pth \
     --model_c2 checkpoints/c2_best.pth
   ```

---

## 🎓 Competition Metric Reference

**Normalized RMSE:**
```
NRMSE = RMSE / std(y_true)
      = sqrt(mean((y_pred - y_true)²)) / std(y_true)
```

**Final Score:**
```
Score = 0.3 × C1_NRMSE + 0.7 × C2_NRMSE
```

**Current Leaderboard:**
- Your best: 1.14 (C1: 1.45, C2: 1.01)
- SOTA: 0.978
- **Target: < 1.0** ← First goal!

---

## ✅ Checklist

- [x] Implemented normalized RMSE metric
- [x] Added train/val split (80/20)
- [x] Validate with metrics each epoch
- [x] Save based on best val NRMSE
- [x] Save predictions for analysis
- [x] Updated experiment logging
- [x] Added --val_split and --no_val args
- [ ] Test with quick run
- [ ] Full training C1
- [ ] Full training C2
- [ ] Analyze results
- [ ] Create submission

**Status: Ready to train! 🚀**
