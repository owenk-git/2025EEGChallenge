# C1 Performance Improvement Approaches

This document describes all implemented approaches to improve C1 (reaction time prediction) performance beyond the current best of 1.09 NRMSE.

## Problem: Prediction Collapse

**Observed issue:**
- Training data: std=0.184, range [0, 1]
- Test predictions: std=0.012, range [0.70, 0.82]
- Model only uses 12% of learned range → severe prediction collapse

## Implemented Solutions

### 1. Distribution Matching Loss ⭐ **RECOMMENDED**

**File:** `train_c1_distribution_matching.py`

**What it does:**
- Forces predictions to match target distribution (mean AND variance)
- Prevents collapse by penalizing narrow prediction distributions

**Usage:**
```bash
python3 train_c1_distribution_matching.py \
  --epochs 100 \
  --batch_size 32 \
  --lr 1e-3 \
  --lambda_mean 0.1 \
  --lambda_std 0.1 \
  --device cuda
```

**Hyperparameters:**
- `--lambda_mean`: Weight for mean matching (default: 0.1)
- `--lambda_std`: Weight for std matching (default: 0.1) - **KEY PARAMETER**
- Higher lambda_std = stronger variance preservation

**Expected improvement:** Should maintain prediction variance close to training data

**Output:** `checkpoints/c1_distribution_matching_best.pt`

---

### 2. Mixup Augmentation ⭐ **RECOMMENDED**

**File:** `train_c1_mixup.py`

**What it does:**
- Mixes pairs of training samples to create intermediate examples
- Forces model to predict intermediate RT values
- Prevents model from only learning to predict mean

**Usage:**
```bash
python3 train_c1_mixup.py \
  --epochs 100 \
  --batch_size 32 \
  --lr 1e-3 \
  --mixup_alpha 0.4 \
  --device cuda
```

**Hyperparameters:**
- `--mixup_alpha`: Controls mixing strength
  - 0.0 = no mixup
  - 0.3-0.4 = moderate (recommended)
  - 1.0 = strong mixing

**Expected improvement:** More diverse predictions, better coverage of RT range

**Output:** `checkpoints/c1_mixup_best.pt`

---

### 3. Confidence-Weighted Loss

**File:** `train_c1_confidence_weighted.py`

**What it does:**
- Gives higher weight to extreme RTs (very fast/slow)
- Forces model to learn full RT range, not just middle values

**Usage:**
```bash
python3 train_c1_confidence_weighted.py \
  --epochs 100 \
  --batch_size 32 \
  --lr 1e-3 \
  --weight_strength 1.0 \
  --device cuda
```

**Hyperparameters:**
- `--weight_strength`: How much to emphasize extremes
  - 0.0 = uniform weighting
  - 1.0 = moderate emphasis (recommended)
  - 2.0 = strong emphasis

**Expected improvement:** Better prediction of extreme RTs

**Output:** `checkpoints/c1_confidence_weighted_best.pt`

---

### 4. Multi-Component RT Model ⭐ **NOVEL APPROACH**

**Files:**
- Model: `models/rt_component_model.py`
- Training: `train_c1_component_model.py`

**What it does:**
- Models RT as sum of interpretable components:
  - **Attention state** (pre-stimulus alpha power)
  - **Decision time** (P300 amplitude/latency)
  - **Motor execution** (beta suppression)
- Learns component weights adaptively

**Usage:**
```bash
python3 train_c1_component_model.py \
  --epochs 100 \
  --batch_size 32 \
  --lr 1e-3 \
  --device cuda
```

**Expected improvement:**
- More interpretable predictions
- Better generalization through explicit RT modeling

**Output:** `checkpoints/c1_component_model_best.pt`

---

### 5. Combined Best Approaches ⭐⭐ **MOST RECOMMENDED**

**File:** `train_c1_combined_best.py`

**What it does:**
- Combines ALL successful techniques:
  - Distribution matching loss
  - Mixup augmentation (50% probability)
  - Confidence weighting

**Usage:**
```bash
python3 train_c1_combined_best.py \
  --epochs 150 \
  --batch_size 32 \
  --lr 1e-3 \
  --lambda_mean 0.05 \
  --lambda_std 0.1 \
  --weight_strength 0.5 \
  --mixup_alpha 0.3 \
  --device cuda
```

**Hyperparameters (tuned defaults):**
- `--lambda_mean 0.05`: Gentle mean matching
- `--lambda_std 0.1`: Strong variance preservation
- `--weight_strength 0.5`: Moderate extreme emphasis
- `--mixup_alpha 0.3`: Conservative mixing

**Expected improvement:** Best overall performance, combines all benefits

**Output:** `checkpoints/c1_combined_best.pt`

---

## Recommended Training Strategy

### Quick Test (1-2 hours each)
```bash
# 1. Distribution matching (fastest, high impact)
python3 train_c1_distribution_matching.py --epochs 50 --device cuda

# 2. Mixup (fast, high impact)
python3 train_c1_mixup.py --epochs 50 --device cuda

# 3. Combined (if above work)
python3 train_c1_combined_best.py --epochs 100 --device cuda
```

### Full Training (overnight)
```bash
# Run combined approach with full epochs
python3 train_c1_combined_best.py \
  --epochs 150 \
  --batch_size 32 \
  --device cuda
```

### Hyperparameter Search
Try different lambda_std values (most important):
```bash
# Conservative (gentle variance preservation)
python3 train_c1_distribution_matching.py --lambda_std 0.05 --epochs 100

# Moderate (recommended)
python3 train_c1_distribution_matching.py --lambda_std 0.1 --epochs 100

# Aggressive (strong variance preservation)
python3 train_c1_distribution_matching.py --lambda_std 0.2 --epochs 100
```

---

## Creating Submissions

After training, create submission with best checkpoint:

```bash
# Pull latest code
git pull

# Create submission
python3 create_advanced_submission.py \
  --checkpoint_c1 checkpoints/c1_combined_best.pt \
  --checkpoint_c2 checkpoints/domain_adaptation_c2_best.pt \
  --name c1_combined_v1
```

This creates: `submissions/c1_combined_v1.zip`

---

## Monitoring Training

**Key metrics to watch:**

1. **Prediction std** - Should match target std (0.18-0.20)
   - If too low (<0.05) → prediction collapse
   - If matches target → good variance preservation

2. **Prediction range** - Should cover [0, 1]
   - Current bad: [0.70, 0.82]
   - Good: [0.3, 0.9] or wider

3. **NRMSE** - Main metric
   - Current best: 1.09
   - Target: <1.05

4. **Correlation** - Should be high (>0.5)
   - Measures if predictions follow true RT patterns

**Example good output:**
```
Epoch 50 Results:
  Train - Pred std: 0.185, Target std: 0.184  ← GOOD
  Train - Pred range: [0.25, 0.95]            ← GOOD
  Val   - NRMSE: 0.98, Corr: 0.62             ← EXCELLENT
```

**Example bad output (collapse):**
```
Epoch 50 Results:
  Train - Pred std: 0.012, Target std: 0.184  ← BAD (collapse)
  Train - Pred range: [0.70, 0.82]            ← BAD (narrow)
  Val   - NRMSE: 1.50, Corr: 0.15             ← POOR
```

---

## Troubleshooting

**Q: Training loss decreases but validation NRMSE increases**
- A: Overfitting. Increase dropout, reduce model size, or add regularization

**Q: Predictions still collapse (low std)**
- A: Increase `--lambda_std` to 0.2 or 0.3
- A: Try combined approach with all techniques

**Q: NRMSE is good but predictions are narrow**
- A: This is the main problem! Use distribution matching loss

**Q: Model predicts only mean value**
- A: Use mixup or confidence weighting to force diverse predictions

**Q: Out of memory**
- A: Reduce `--batch_size` to 16 or 8

---

## Expected Timeline

| Approach | Training Time | Expected NRMSE | Expected Std |
|----------|--------------|----------------|--------------|
| Baseline (current) | 2 hours | 1.09 | 0.012 (collapsed) |
| Distribution matching | 2-3 hours | 1.05-1.08 | 0.15-0.18 ✓ |
| Mixup | 2-3 hours | 1.06-1.09 | 0.10-0.15 |
| Confidence weighted | 2-3 hours | 1.07-1.10 | 0.05-0.10 |
| Component model | 3-4 hours | 1.04-1.07 | 0.12-0.16 |
| **Combined best** | **4-5 hours** | **1.02-1.05** ✓ | **0.16-0.19** ✓ |

---

## Next Steps After This

If combined approach achieves ~1.05 NRMSE with good variance:

1. **Ensemble multiple approaches**
   - Average predictions from 3-4 best models
   - Could reach 1.00-1.03 NRMSE

2. **Subject-aware training**
   - Add subject embeddings
   - Per-subject calibration

3. **Event-based trial extraction**
   - Use actual stimulus/response markers
   - Better quality trials

4. **Curriculum learning**
   - Train in stages (easy → hard)
   - Better convergence

---

## Questions?

Check training logs for:
- Prediction variance (`pred_std`)
- Prediction range (`pred range`)
- Correlation with targets (`corr`)

If still having issues, the problem is likely:
1. Lambda values too small/large
2. Model architecture mismatch
3. Data quality issues
