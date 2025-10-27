# Recording-Level N-Fold Training

## Why Recording-Level Instead of Trial-Level?

Your best competition scores (1.09-1.10) came from **recording-level models**, not trial-level models:

- ✅ `cnn_ensemble_v2_submission.zip` (1.09) - Recording-level CNN
- ✅ `transformer_v2` (1.09) - Recording-level Transformer
- ✅ `eegnex_improved_submission.zip` (1.10) - Recording-level EEGNeX
- ✅ `erp_mlp_submission.zip` (1.10) - Recording-level ERP features

**Trial-level submissions got worse scores:**
- ❌ `trial_level_fixed_v4.zip` (1.32)
- ❌ `c1_combined_final.zip` (1.32)

**Root cause:** Trial extraction + aggregation adds complexity and loses information. Recording-level prediction is simpler and better!

## Available Models

### 1. ERP MLP (Best: 1.10) ⭐ RECOMMENDED
Uses neuroscience-based features:
- P300 latency (correlates 0.6-0.8 with RT)
- N2 amplitude (response inhibition)
- Pre-stimulus alpha (attention)
- Motor beta (preparation)

**Why it's best:** Based on 50+ years of cognitive neuroscience research showing P300 latency predicts RT.

### 2. CNN Ensemble (Best: 1.09)
Combines 3 different CNN architectures:
- Temporal CNN (time patterns)
- Spatial CNN (channel patterns)
- Hybrid CNN (both)

### 3. EEGNeX Improved (Best: 1.10)
Modern depthwise-separable CNN architecture with:
- Residual connections
- Channel attention
- Efficient computation

## Training with N-Fold Cross-Validation

### Challenge 1 (RT Prediction)

```bash
# ERP MLP (RECOMMENDED - scored 1.10)
python3 train_recording_kfold.py \
  --model erp_mlp \
  --challenge c1 \
  --n_folds 5 \
  --epochs 50 \
  --batch_size 32 \
  --lr 1e-3

# CNN Ensemble (scored 1.09)
python3 train_recording_kfold.py \
  --model cnn_ensemble \
  --challenge c1 \
  --n_folds 5 \
  --epochs 50 \
  --batch_size 32 \
  --lr 1e-3

# EEGNeX Improved (scored 1.10)
python3 train_recording_kfold.py \
  --model eegnex_improved \
  --challenge c1 \
  --n_folds 5 \
  --epochs 50 \
  --batch_size 32 \
  --lr 1e-3
```

### Challenge 2 (Externalizing Factor)

```bash
python3 train_recording_kfold.py \
  --model erp_mlp \
  --challenge c2 \
  --n_folds 5 \
  --epochs 50 \
  --batch_size 32 \
  --lr 1e-3
```

### Quick Test (Mini Dataset)

```bash
python3 train_recording_kfold.py \
  --model erp_mlp \
  --challenge c1 \
  --n_folds 3 \
  --epochs 10 \
  --mini
```

## What Gets Saved

After training completes:

```
checkpoints_kfold/
├── erp_mlp_c1/
│   ├── fold_0_best.pth      # Best checkpoint for fold 0
│   ├── fold_1_best.pth      # Best checkpoint for fold 1
│   ├── fold_2_best.pth      # Best checkpoint for fold 2
│   ├── fold_3_best.pth      # Best checkpoint for fold 3
│   ├── fold_4_best.pth      # Best checkpoint for fold 4
│   └── kfold_summary.json   # Summary with best fold info
```

## Understanding the Output

```
Fold 1: Val NRMSE = 0.9823
Fold 2: Val NRMSE = 0.9654  ⭐ Best
Fold 3: Val NRMSE = 1.0123
Fold 4: Val NRMSE = 0.9987
Fold 5: Val NRMSE = 1.0045

Average Val NRMSE: 0.9926 ± 0.0189
Best Fold: 2 (NRMSE: 0.9654)
```

**What to use for submission:** The best fold checkpoint!
```
checkpoints_kfold/erp_mlp_c1/fold_1_best.pth  (fold 2 = index 1)
```

## Creating Submission

After training completes, create submission with the best fold:

```bash
# Check kfold_summary.json to find best fold
cat checkpoints_kfold/erp_mlp_c1/kfold_summary.json

# Create submission with best fold checkpoint
python3 create_universal_submission.py \
  --model erp_mlp \
  --c1 checkpoints_kfold/erp_mlp_c1/fold_1_best.pth \
  --c2 checkpoints_kfold/erp_mlp_c2/fold_0_best.pth
```

## Why N-Fold Validation Matters

**Previous problem:** Your models trained without proper validation showed:
- Training NRMSE: 0.99
- Competition NRMSE: 1.32

**Overfitting!** The model memorized training data.

**N-Fold solution:**
- Trains 5 different models on different data splits
- Each fold validates on held-out recordings
- Prevents overfitting by testing generalization
- Gives realistic performance estimate

**Expected improvement:**
- More robust models
- Better generalization to test set
- True validation NRMSE ≈ Competition NRMSE

## Monitoring Training

Each epoch shows:
```
Train - NRMSE: 0.9823, Corr: 0.6234, Pred Std: 0.1823
Val   - NRMSE: 1.0156, Corr: 0.5987, Pred Std: 0.1654, Target Std: 0.1823
```

**What to watch:**
- ✅ Val NRMSE decreasing → Model learning
- ✅ Val Corr increasing → Better predictions
- ✅ Pred Std ≈ Target Std → No variance collapse
- ❌ Train NRMSE << Val NRMSE → Overfitting (early stopping will handle this)

## Hyperparameter Tuning

If results aren't good enough:

```bash
# Try higher learning rate
python3 train_recording_kfold.py --model erp_mlp --challenge c1 --lr 3e-3

# Try larger batch size
python3 train_recording_kfold.py --model erp_mlp --challenge c1 --batch_size 64

# Try more epochs
python3 train_recording_kfold.py --model erp_mlp --challenge c1 --epochs 100
```

## Expected Results

Based on leaderboard history:
- **Target:** NRMSE < 1.09 (beat your previous best)
- **Realistic:** NRMSE ≈ 0.95-1.05
- **Excellent:** NRMSE < 0.95

## Troubleshooting

### "ModuleNotFoundError: No module named 'eegdash'"
```bash
pip install eegdash braindecode
```

### "CUDA out of memory"
```bash
# Reduce batch size
python3 train_recording_kfold.py --model erp_mlp --challenge c1 --batch_size 16
```

### "All targets are the same"
This means RT extraction failed. Check that:
- Dataset is properly downloaded
- Annotations exist in the recordings
- RT extractor is working

### Training is too slow
```bash
# Use mini dataset for testing
python3 train_recording_kfold.py --model erp_mlp --challenge c1 --mini

# Or reduce number of folds
python3 train_recording_kfold.py --model erp_mlp --challenge c1 --n_folds 3
```

## Next Steps

1. **Train ERP MLP with 5-fold CV** (most likely to beat 1.09)
2. **Check kfold_summary.json** to find best fold
3. **Create submission** with best fold checkpoint
4. **Submit to competition** and compare with 1.09 baseline
5. **If better:** Try other models (CNN Ensemble, EEGNeX)
6. **If worse:** Debug RT extraction or try different hyperparameters
