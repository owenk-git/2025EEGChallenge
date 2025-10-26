# Final Commands - All 3 Models with Direct Loading

## ✅ All 3 Models Now Support Direct Loading!

**No preprocessing needed for any model!**

---

## 🚀 Quick Start - Train All 3 Models

```bash
# Make script executable
chmod +x train_all_models_direct.sh

# Train all 3 models + create submissions (6-8 hours)
./train_all_models_direct.sh
```

This trains all 3 models and creates 3 submissions automatically.

---

## 📋 Individual Model Commands

### Model 1: Domain Adaptation EEGNeX

**Expected**: 1.05-1.10 overall (10-15% improvement)

```bash
# Train C1
python3 train_domain_adaptation_direct.py \
    --challenge c1 \
    --epochs 100 \
    --batch_size 64 \
    --lr 1e-3 \
    --lambda_mmd 0.1 \
    --lambda_entropy 0.01 \
    --lambda_adv 0.1 \
    --device cuda

# Train C2
python3 train_domain_adaptation_direct.py \
    --challenge c2 \
    --epochs 100 \
    --batch_size 64 \
    --lr 1e-3 \
    --lambda_mmd 0.1 \
    --lambda_entropy 0.01 \
    --lambda_adv 0.1 \
    --device cuda

# Create submission
python3 create_advanced_submission.py \
    --model domain_adaptation \
    --name domain_adaptation_v1 \
    --checkpoint_c1 checkpoints/domain_adaptation_c1_best.pt \
    --checkpoint_c2 checkpoints/domain_adaptation_c2_best.pt
```

---

### Model 2: Cross-Task Pre-Training

**Expected**: 1.02-1.08 overall (15-20% improvement)

```bash
# Train C1
python3 train_cross_task_direct.py \
    --challenge c1 \
    --epochs 100 \
    --pretrain_epochs 50 \
    --batch_size 64 \
    --lr 1e-3 \
    --device cuda

# Train C2
python3 train_cross_task_direct.py \
    --challenge c2 \
    --epochs 100 \
    --pretrain_epochs 50 \
    --batch_size 64 \
    --lr 1e-3 \
    --device cuda

# Create submission
python3 create_advanced_submission.py \
    --model cross_task \
    --name cross_task_pretrain_v1 \
    --checkpoint_c1 checkpoints/cross_task_c1_best.pt \
    --checkpoint_c2 checkpoints/cross_task_c2_best.pt
```

---

### Model 3: Hybrid CNN-Transformer-DA ⭐ BEST

**Expected**: 1.01-1.07 overall (20-25% improvement)

```bash
# Train C1
python3 train_hybrid_direct.py \
    --challenge c1 \
    --epochs 100 \
    --batch_size 64 \
    --lr 1e-3 \
    --lambda_mmd 0.1 \
    --lambda_entropy 0.01 \
    --device cuda

# Train C2
python3 train_hybrid_direct.py \
    --challenge c2 \
    --epochs 100 \
    --batch_size 64 \
    --lr 1e-3 \
    --lambda_mmd 0.1 \
    --lambda_entropy 0.01 \
    --device cuda

# Create submission
python3 create_advanced_submission.py \
    --model hybrid \
    --name hybrid_cnn_transformer_v1 \
    --checkpoint_c1 checkpoints/hybrid_c1_best.pt \
    --checkpoint_c2 checkpoints/hybrid_c2_best.pt
```

---

## 🧪 Quick Test Commands (Mini Dataset)

Test each model quickly before full training:

```bash
# Model 1: Domain Adaptation (5-10 min)
python3 train_domain_adaptation_direct.py --challenge c1 --epochs 10 --batch_size 32 --mini

# Model 2: Cross-Task (5-10 min)
python3 train_cross_task_direct.py --challenge c1 --epochs 10 --pretrain_epochs 5 --batch_size 32 --mini

# Model 3: Hybrid (5-10 min)
python3 train_hybrid_direct.py --challenge c1 --epochs 10 --batch_size 32 --mini
```

---

## 📊 Complete Command Summary

| Model | Script | C1 Command | C2 Command |
|-------|--------|------------|------------|
| **Domain Adaptation** | `train_domain_adaptation_direct.py` | `--challenge c1 --epochs 100 --batch_size 64` | `--challenge c2 --epochs 100 --batch_size 64` |
| **Cross-Task** | `train_cross_task_direct.py` | `--challenge c1 --epochs 100 --pretrain_epochs 50` | `--challenge c2 --epochs 100 --pretrain_epochs 50` |
| **Hybrid** | `train_hybrid_direct.py` | `--challenge c1 --epochs 100 --batch_size 64` | `--challenge c2 --epochs 100 --batch_size 64` |

---

## ⚡ One-Line Commands

### Train Single Model

```bash
# Domain Adaptation only (fastest to results)
python3 train_domain_adaptation_direct.py --challenge c1 --epochs 100 --batch_size 64 && \
python3 train_domain_adaptation_direct.py --challenge c2 --epochs 100 --batch_size 64

# Cross-Task
python3 train_cross_task_direct.py --challenge c1 --epochs 100 --pretrain_epochs 50 && \
python3 train_cross_task_direct.py --challenge c2 --epochs 100 --pretrain_epochs 50

# Hybrid (best expected performance)
python3 train_hybrid_direct.py --challenge c1 --epochs 100 --batch_size 64 && \
python3 train_hybrid_direct.py --challenge c2 --epochs 100 --batch_size 64
```

### Train All 3 Models

```bash
./train_all_models_direct.sh
```

---

## 🎯 Recommended Workflow

### Option A: Quick Start (Train Best Model Only)

```bash
# Train Hybrid model (best expected performance)
python3 train_hybrid_direct.py --challenge c1 --epochs 100 --batch_size 64
python3 train_hybrid_direct.py --challenge c2 --epochs 100 --batch_size 64

# Create submission
python3 create_advanced_submission.py \
    --model hybrid \
    --name hybrid_cnn_transformer_v1 \
    --checkpoint_c1 checkpoints/hybrid_c1_best.pt \
    --checkpoint_c2 checkpoints/hybrid_c2_best.pt
```

**Time**: ~2-3 hours
**Expected**: 1.01-1.07 overall

---

### Option B: Train All 3 Models (Best Coverage)

```bash
# Train all 3 models
./train_all_models_direct.sh
```

**Time**: ~6-8 hours
**Result**: 3 submissions to test which works best

---

### Option C: Test First, Then Train

```bash
# Quick test (5-10 min each)
python3 train_domain_adaptation_direct.py --challenge c1 --epochs 10 --mini
python3 train_cross_task_direct.py --challenge c1 --epochs 10 --pretrain_epochs 5 --mini
python3 train_hybrid_direct.py --challenge c1 --epochs 10 --mini

# If all work, train full models
./train_all_models_direct.sh
```

---

## 📦 Data Loading Methods

All models now support **3 data loading methods**:

### Method 1: Direct Loading (RECOMMENDED) ✅

```bash
# Uses eegdash cache (~500 MB)
python3 train_domain_adaptation_direct.py --challenge c1 --epochs 100
python3 train_cross_task_direct.py --challenge c1 --epochs 100
python3 train_hybrid_direct.py --challenge c1 --epochs 100
```

**Pros**: No preprocessing, reasonable cache
**Cons**: None

---

### Method 2: S3 Streaming (Zero Cache)

```bash
# Install: pip install boto3 mne
# Only available for Domain Adaptation
python3 train_domain_adaptation_s3.py --challenge c1 --epochs 100 --batch_size 32
```

**Pros**: Zero local disk usage
**Cons**: 2-3x slower, only Domain Adaptation

---

### Method 3: Preprocessed (Fastest Training)

```bash
# Preprocess once
python3 preprocess_data_for_advanced_models.py --challenges c1,c2

# Train (slightly faster)
python3 train_domain_adaptation.py --challenge c1 --epochs 100
python3 train_cross_task.py --challenge c1 --epochs 100
python3 train_hybrid.py --challenge c1 --epochs 100
```

**Pros**: 5-10% faster training
**Cons**: Preprocessing step, more disk space

---

## 📈 Expected Results

| Model | Val NRMSE | Test NRMSE (Expected) | Overall Score | Improvement |
|-------|-----------|---------------------|---------------|-------------|
| **Current Best** | 1.00 | 1.31 (C1), 1.00 (C2) | 1.09 | Baseline |
| **Domain Adaptation** | 0.95-1.05 | 1.15-1.20 (C1), 1.00-1.05 (C2) | **1.05-1.10** | +4-9% |
| **Cross-Task** | 0.90-1.00 | 1.10-1.20 (C1), 0.98-1.03 (C2) | **1.02-1.08** | +7-13% |
| **Hybrid** ⭐ | 0.90-1.00 | 1.10-1.15 (C1), 0.98-1.02 (C2) | **1.01-1.07** | +9-15% |
| **Target (MBZUAI)** | ? | 0.92 (C1), 1.00 (C2) | **0.976** | +30% |

---

## 🎬 Final Recommendation

**Just run this:**

```bash
git pull
chmod +x train_all_models_direct.sh
./train_all_models_direct.sh
```

This will:
1. ✅ Train all 3 models (no preprocessing needed)
2. ✅ Create 3 submissions automatically
3. ✅ Take 6-8 hours
4. ✅ Give you 3 submissions to test

Then submit all 3 and see which performs best! 🚀

---

## 💡 Pro Tips

1. **Start with Hybrid** if you only have time for one model
2. **Train all 3** if you want best coverage
3. **Test with `--mini` first** to verify everything works
4. **Monitor with `nvidia-smi`** to check GPU usage
5. **Use `tmux` or `screen`** for long training sessions

---

**Ready to beat 0.976!** 🏆
