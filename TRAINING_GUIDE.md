# Training Guide for Advanced Models

This guide explains how to train the three advanced models designed to beat the 0.976 score.

## Overview

We've implemented 3 state-of-the-art approaches from the literature review:

1. **Domain Adaptation EEGNeX**: MMD + Entropy Minimization + Subject-Adversarial Training
2. **Cross-Task Pre-Training**: Multi-task learning with pre-training and fine-tuning
3. **Hybrid CNN-Transformer-DA**: CNN + Transformer + ERP Features + Domain Adaptation

## Quick Start: Train All Models

```bash
chmod +x train_all_advanced_models.sh
./train_all_advanced_models.sh
```

This will:
- Train all 3 models for both C1 and C2
- Create submission zip files
- Take approximately 6-8 hours total on GPU

## Individual Model Training

### 1. Domain Adaptation EEGNeX

**Expected Improvement**: 10-15% over baseline (Target: C1 ~ 1.15-1.20)

**Train C1:**
```bash
python3 train_domain_adaptation.py \
    --challenge c1 \
    --epochs 100 \
    --batch_size 64 \
    --lr 1e-3 \
    --lambda_mmd 0.1 \
    --lambda_entropy 0.01 \
    --lambda_adv 0.1 \
    --device cuda \
    --save_dir checkpoints
```

**Train C2:**
```bash
python3 train_domain_adaptation.py \
    --challenge c2 \
    --epochs 100 \
    --batch_size 64 \
    --lr 1e-3 \
    --lambda_mmd 0.1 \
    --lambda_entropy 0.01 \
    --lambda_adv 0.1 \
    --device cuda \
    --save_dir checkpoints
```

**Create Submission:**
```bash
python3 create_advanced_submission.py \
    --model domain_adaptation \
    --name domain_adaptation_v1 \
    --checkpoint_c1 checkpoints/domain_adaptation_c1_best.pt \
    --checkpoint_c2 checkpoints/domain_adaptation_c2_best.pt \
    --output_dir submissions
```

---

### 2. Cross-Task Pre-Training

**Expected Improvement**: 10-20% over baseline (Competition recommended approach)

**Train C1:**
```bash
python3 train_cross_task.py \
    --challenge c1 \
    --epochs 100 \
    --pretrain_epochs 50 \
    --batch_size 64 \
    --lr 1e-3 \
    --device cuda \
    --save_dir checkpoints
```

**Train C2:**
```bash
python3 train_cross_task.py \
    --challenge c2 \
    --epochs 100 \
    --pretrain_epochs 50 \
    --batch_size 64 \
    --lr 1e-3 \
    --device cuda \
    --save_dir checkpoints
```

**Create Submission:**
```bash
python3 create_advanced_submission.py \
    --model cross_task \
    --name cross_task_pretrain_v1 \
    --checkpoint_c1 checkpoints/cross_task_c1_best.pt \
    --checkpoint_c2 checkpoints/cross_task_c2_best.pt \
    --output_dir submissions
```

---

### 3. Hybrid CNN-Transformer-DA

**Expected Improvement**: Best of all approaches (Target: C1 ~ 1.10-1.15)

**Train C1:**
```bash
python3 train_hybrid.py \
    --challenge c1 \
    --epochs 100 \
    --batch_size 64 \
    --lr 1e-3 \
    --lambda_mmd 0.1 \
    --lambda_entropy 0.01 \
    --device cuda \
    --save_dir checkpoints
```

**Train C2:**
```bash
python3 train_hybrid.py \
    --challenge c2 \
    --epochs 100 \
    --batch_size 64 \
    --lr 1e-3 \
    --lambda_mmd 0.1 \
    --lambda_entropy 0.01 \
    --device cuda \
    --save_dir checkpoints
```

**Create Submission:**
```bash
python3 create_advanced_submission.py \
    --model hybrid \
    --name hybrid_cnn_transformer_v1 \
    --checkpoint_c1 checkpoints/hybrid_c1_best.pt \
    --checkpoint_c2 checkpoints/hybrid_c2_best.pt \
    --output_dir submissions
```

---

## Hyperparameter Tuning

### Domain Adaptation

Key hyperparameters:
- `lambda_mmd`: Weight for MMD loss (default: 0.1)
  - Higher = stronger distribution alignment
  - Try: [0.05, 0.1, 0.2]
- `lambda_entropy`: Weight for entropy minimization (default: 0.01)
  - Higher = more confident predictions
  - Try: [0.005, 0.01, 0.02]
- `lambda_adv`: Weight for adversarial loss (default: 0.1)
  - Higher = more subject-invariant features
  - Try: [0.05, 0.1, 0.2]

### Cross-Task Pre-Training

Key hyperparameters:
- `pretrain_epochs`: Number of pre-training epochs (default: 50)
  - More = better general representations
  - Try: [30, 50, 100]
- `lr`: Fine-tuning uses 0.1x pre-training LR
  - Pre-training LR: 1e-3
  - Fine-tuning LR: 1e-4

### Hybrid CNN-Transformer

Key hyperparameters:
- `lambda_mmd`: MMD weight (default: 0.1)
- `lambda_entropy`: Entropy weight (default: 0.01)
- `batch_size`: May need to reduce if OOM (try 32)

---

## Expected Performance

Based on literature review:

### Domain Adaptation
- **Validation**: C1 ~ 0.95-1.05, C2 ~ 0.98-1.02
- **Test (Expected)**: C1 ~ 1.15-1.20, C2 ~ 1.00-1.05
- **Overall**: ~ 1.05-1.10

### Cross-Task Pre-Training
- **Validation**: C1 ~ 0.90-1.00, C2 ~ 0.95-1.00
- **Test (Expected)**: C1 ~ 1.10-1.20, C2 ~ 0.98-1.03
- **Overall**: ~ 1.02-1.08

### Hybrid CNN-Transformer-DA
- **Validation**: C1 ~ 0.90-1.00, C2 ~ 0.95-1.00
- **Test (Expected)**: C1 ~ 1.10-1.15, C2 ~ 0.98-1.02
- **Overall**: ~ 1.01-1.07 (Best expected)

### Target Performance
- **MBZUAI Team**: C1 = 0.92, C2 = 1.00, Overall = 0.976
- **Gap to Close**: Need C1 ~ 0.92 (vs. our expected 1.10-1.15)

---

## Model Architectures

### 1. Domain Adaptation EEGNeX

```
Input (129 channels × 900 time points)
    ↓
[Spatial Conv] → Filter across channels
    ↓
[Temporal Conv] → Filter across time
    ↓
[Separable Convs] → 3 blocks with residual connections
    ↓
[Feature Extractor] → (batch, feature_dim)
    ↓
┌──────────────┬──────────────────────┐
│              │                      │
[Task Head]    [Subject Discriminator] (adversarial)
│              │
Prediction     Subject ID (reversed gradient)

Losses:
- Task Loss (MSE)
- MMD Loss (distribution alignment)
- Entropy Loss (confident predictions)
- Adversarial Loss (subject invariance)
```

### 2. Cross-Task Pre-Training

```
Stage 1: Pre-training
Input (all 6 tasks)
    ↓
[Shared Feature Extractor]
    ↓
┌────┬────┬────┬────┬────┬────┐
Task1 Task2 Task3 Task4 Task5 Task6
(Multi-task learning)

Stage 2: Fine-tuning
Input (CCD task only)
    ↓
[Shared Feature Extractor] (pre-trained)
    ↓
[Task-Specific Head] (fine-tuned)
    ↓
Prediction
```

### 3. Hybrid CNN-Transformer-DA

```
Input (129 channels × 900 time points)
    ↓
┌─────────────────────┬──────────────────────┐
│                     │                      │
[CNN Feature         [ERP Feature            │
 Extractor]           Extractor]             │
│                     │                      │
(batch, 128, time)   (batch, 13)            │
│                     │                      │
[Transformer         │                      │
 Encoder]             │                      │
│                     │                      │
(batch, 128, time)   │                      │
│                     │                      │
[Flatten]            │                      │
│                     │                      │
└─────────────────────┴──────────────────────┘
                      │
            [Concatenate Features]
                      │
            (batch, learned + ERP)
                      │
              [Fusion Network]
                      │
                 Prediction

Losses:
- Task Loss (MSE)
- MMD Loss (distribution alignment)
- Entropy Loss (confident predictions)
```

---

## Troubleshooting

### Out of Memory (OOM)

**Reduce batch size:**
```bash
--batch_size 32  # or even 16
```

**Reduce model size (Hybrid only):**
- Reduce transformer layers: `--num_transformer_layers 2`
- Reduce d_model: Use 64 instead of 128

### Training Too Slow

**Reduce epochs:**
```bash
--epochs 50  # Quick test
```

**Skip pre-training (Cross-Task only):**
```bash
--pretrain_epochs 0
```

### Model Not Converging

**Increase learning rate:**
```bash
--lr 2e-3
```

**Check data:**
- Ensure data is properly preprocessed
- Check for NaN values
- Verify normalization

---

## Submission Checklist

Before submitting:

1. ✓ Trained both C1 and C2 models
2. ✓ Best checkpoints saved in `checkpoints/`
3. ✓ Submission zip created in `submissions/`
4. ✓ Validated submission file structure:
   ```
   submission.zip
   ├── submission.py
   ├── model_c1.pt
   ├── model_c2.pt
   └── [model_file].py
   ```
5. ✓ Test submission locally (optional):
   ```bash
   cd submissions
   unzip [submission].zip -d test/
   cd test
   python3 submission.py
   ```

---

## Next Steps

1. **Train all 3 models** using `train_all_advanced_models.sh`
2. **Submit all 3** to competition (you have 3 submissions left today)
3. **Analyze results** to see which approach works best
4. **Iterate** based on what works:
   - If Domain Adaptation works → tune λ values
   - If Cross-Task works → get real multi-task data
   - If Hybrid works → tune transformer architecture

---

## Expected Timeline

- **Training**: 2-3 hours per model (6-8 hours total)
- **Submission creation**: 5 minutes per model
- **Upload to CodaBench**: 5 minutes per submission
- **Evaluation**: 1-2 hours per submission

**Total time**: ~10-12 hours from start to results

Good luck! 🚀
