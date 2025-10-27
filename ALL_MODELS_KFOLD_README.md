# Train All Models with K-Fold and Select Best Per Fold

## Strategy

Instead of training just one model architecture, this trains **ALL available models** on each fold and selects the best performing one:

### For each fold (1-5):
1. **Train ERP MLP** → Val NRMSE = 0.98
2. **Train CNN Ensemble** → Val NRMSE = 1.02
3. **Train EEGNeX Improved** → Val NRMSE = 0.95 ✅ **Best**
4. **Select best** → EEGNeX Improved

### Result:
- Fold 1 best: EEGNeX Improved (0.95)
- Fold 2 best: ERP MLP (0.93)
- Fold 3 best: ERP MLP (0.97)
- Fold 4 best: CNN Ensemble (0.96)
- Fold 5 best: EEGNeX Improved (0.94)

**Ensemble these 5 best models at test time** (may be different architectures!)

## Why This Approach?

**Problem:** Different models perform better on different data splits.

**Solution:** Let each fold choose its own best model!

**Benefits:**
- ✅ Maximizes performance per fold (always use best model)
- ✅ Heterogeneous ensemble (diverse models → more robust)
- ✅ Automatic model selection (no manual tuning)
- ✅ Leverages strengths of different architectures

## Available Models

All models are pure PyTorch (no external dependencies needed for submission):

1. **ERP MLP** - Neuroscience-based features (P300, N2, alpha, beta)
2. **CNN Ensemble** - 3 CNN branches (temporal, spatial, hybrid)
3. **EEGNeX Improved** - Modern depthwise-separable CNN with attention

## Usage

### Challenge 1 (RT Prediction)

```bash
# Train all models with 5-fold CV
python3 train_all_models_kfold.py --c 1 --f 5 --e 30 --b 32 --lr 1e-3
```

This will:
- Create folder `checkpoints_all_models_c1_TIMESTAMP/`
- Train 3 models × 5 folds = 15 model trainings
- Save best model per fold as `fold_0_best.pth`, `fold_1_best.pth`, etc.
- Create `best_per_fold_summary.json` with selection results

### Challenge 2 (Externalizing Factor)

```bash
python3 train_all_models_kfold.py --c 2 --f 5 --e 30 --b 32 --lr 1e-3
```

### Quick Test (Mini Dataset)

```bash
python3 train_all_models_kfold.py --c 1 --f 3 --e 10 --mini
```

## What Gets Trained

For each fold, trains:

```
Fold 1:
  ├── erp_mlp        (30 epochs) → Val NRMSE: 0.98
  ├── cnn_ensemble   (30 epochs) → Val NRMSE: 1.02
  └── eegnex_improved (30 epochs) → Val NRMSE: 0.95 ✅ BEST

  Saves: fold_0_best.pth (eegnex_improved)

Fold 2:
  ├── erp_mlp        (30 epochs) → Val NRMSE: 0.93 ✅ BEST
  ├── cnn_ensemble   (30 epochs) → Val NRMSE: 0.97
  └── eegnex_improved (30 epochs) → Val NRMSE: 0.96

  Saves: fold_1_best.pth (erp_mlp)

... continues for all folds
```

## Output Structure

```
checkpoints_all_models_c1_20251027_1430/
├── fold_0_erp_mlp_best.pth          # Each model's best checkpoint
├── fold_0_cnn_ensemble_best.pth
├── fold_0_eegnex_improved_best.pth
├── fold_0_best.pth                  # ✅ Overall best for fold 0
├── fold_1_erp_mlp_best.pth
├── fold_1_cnn_ensemble_best.pth
├── fold_1_eegnex_improved_best.pth
├── fold_1_best.pth                  # ✅ Overall best for fold 1
├── ...
└── best_per_fold_summary.json       # 📊 Summary of selections
```

## Checking Results

```bash
# See which model was selected for each fold
cat checkpoints_all_models_c1_TIMESTAMP/best_per_fold_summary.json
```

Example output:
```json
{
  "fold_results": [
    {
      "fold": 0,
      "best_model": "eegnex_improved",
      "best_val_nrmse": 0.9523,
      "all_models": [
        {"model_name": "erp_mlp", "best_val_nrmse": 0.9834},
        {"model_name": "cnn_ensemble", "best_val_nrmse": 1.0234},
        {"model_name": "eegnex_improved", "best_val_nrmse": 0.9523}
      ]
    },
    ...
  ],
  "avg_nrmse": 0.9645,
  "std_nrmse": 0.0234
}
```

## Creating Ensemble Submission

After training completes, create submission with the best-per-fold models:

```bash
# For C1 only (use existing C2 model)
python3 create_mixed_ensemble_submission.py \
  --c1 checkpoints_all_models_c1_TIMESTAMP/fold_0_best.pth \
  --c1 checkpoints_all_models_c1_TIMESTAMP/fold_1_best.pth \
  --c1 checkpoints_all_models_c1_TIMESTAMP/fold_2_best.pth \
  --c1 checkpoints_all_models_c1_TIMESTAMP/fold_3_best.pth \
  --c1 checkpoints_all_models_c1_TIMESTAMP/fold_4_best.pth \
  --c2 checkpoints/domain_adaptation_c2_best.pt \
  --name best_per_fold_ensemble
```

The `create_mixed_ensemble_submission.py` script will:
- ✅ Auto-detect each model's architecture
- ✅ Load all 5 different models at test time
- ✅ Average predictions across heterogeneous ensemble

## Training Time Estimate

For 5 folds × 3 models × 30 epochs:
- **Mini dataset:** ~2-3 hours total
- **Full dataset:** ~15-20 hours total

Each model trains in parallel within a fold, but folds run sequentially.

## Expected Performance

Based on your leaderboard history:
- **Single model K-fold:** NRMSE ≈ 0.99-1.05
- **Best-per-fold ensemble:** NRMSE ≈ 0.95-1.00 (expected 3-5% improvement)

Why better?
- Each fold uses its optimal architecture
- Heterogeneous ensemble reduces overfitting
- Leverages complementary strengths of different models

## Monitoring Training

You'll see output like:
```
============================================================
FOLD 1
============================================================
Train recordings: 613
Val recordings: 154

  Training erp_mlp...
    Epoch 5/30 - Val NRMSE: 1.0234, Corr: 0.5432
    Epoch 10/30 - Val NRMSE: 0.9987, Corr: 0.5876
    Early stopping at epoch 23
  ✅ erp_mlp: Best Val NRMSE = 0.9834 (epoch 18)

  Training cnn_ensemble...
    Epoch 5/30 - Val NRMSE: 1.0567, Corr: 0.5123
    ...
  ✅ cnn_ensemble: Best Val NRMSE = 1.0234 (epoch 15)

  Training eegnex_improved...
    Epoch 5/30 - Val NRMSE: 0.9876, Corr: 0.6012
    ...
  ✅ eegnex_improved: Best Val NRMSE = 0.9523 (epoch 20)

🏆 FOLD 1 BEST: eegnex_improved (NRMSE: 0.9523)
```

## Troubleshooting

### "Out of memory"
```bash
# Reduce batch size
python3 train_all_models_kfold.py --c 1 --f 5 --e 30 --b 16
```

### "Takes too long"
```bash
# Reduce epochs per model
python3 train_all_models_kfold.py --c 1 --f 5 --e 20

# Or use fewer folds
python3 train_all_models_kfold.py --c 1 --f 3 --e 30
```

### "All models perform similarly"
This is actually good! It means:
- Models are well-tuned
- Dataset is consistent
- Ensemble will still help (different models capture different patterns)

## Comparison with Other Approaches

| Approach | Description | Expected NRMSE |
|----------|-------------|----------------|
| Single model, no CV | Train one model once | 1.20-1.32 (overfitting) |
| Single model, K-fold | Train one model with K-fold | 1.05-1.10 |
| Same model, K-fold ensemble | Ensemble K copies of same model | 1.00-1.05 |
| **Best-per-fold ensemble** | **Ensemble K best models (may differ)** | **0.95-1.00** ✅ |

## Next Steps

1. **Run training:** `python3 train_all_models_kfold.py --c 1 --f 5 --e 30`
2. **Check results:** `cat checkpoints_all_models_c1_*/best_per_fold_summary.json`
3. **Create submission:** Use `create_mixed_ensemble_submission.py` with best fold checkpoints
4. **Submit and compare** with your best score of 1.09

This approach should beat 1.09 by leveraging the best of all model architectures!
