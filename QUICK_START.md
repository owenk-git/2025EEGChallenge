# Quick Start Guide - No Preprocessing Needed!

You can train the advanced models **directly** without preprocessing data to disk.

## Two Methods Available

### Method 1: Direct Loading (RECOMMENDED - No Preprocessing!)

Loads data directly from eegdash cache/online during training:

```bash
# Train Domain Adaptation (uses data directly)
python3 train_domain_adaptation_direct.py --challenge c1 --epochs 100 --batch_size 64
python3 train_domain_adaptation_direct.py --challenge c2 --epochs 100 --batch_size 64

# Test with mini dataset first (much faster)
python3 train_domain_adaptation_direct.py --challenge c1 --epochs 10 --batch_size 32 --mini
```

**Advantages**:
- ✅ No preprocessing step needed
- ✅ Uses less disk space
- ✅ Works with eegdash cache (downloads once, caches locally)
- ✅ Same as your existing `train_official.py` approach

**Disadvantages**:
- ❌ Slightly slower per epoch (loads data on-the-fly)
- ❌ Harder to implement subject-based splitting for domain adaptation

---

### Method 2: Preprocessed Loading (For Faster Training)

Preprocesses data to numpy arrays first, then trains:

```bash
# Step 1: Preprocess (10-30 minutes, only once)
python3 preprocess_data_for_advanced_models.py --challenges c1,c2

# Step 2: Train (uses preprocessed data)
python3 train_domain_adaptation.py --challenge c1 --epochs 100 --batch_size 64
python3 train_domain_adaptation.py --challenge c2 --epochs 100 --batch_size 64
```

**Advantages**:
- ✅ Faster training (5-10% speedup per epoch)
- ✅ Easier to implement subject-based domain adaptation
- ✅ Can inspect/validate data before training

**Disadvantages**:
- ❌ Requires preprocessing step
- ❌ Uses more disk space (~100-500 MB per challenge)

---

## Which Method Should You Use?

### Use Method 1 (Direct) if:
- You want to start training immediately
- Disk space is limited
- You're testing/debugging
- You're familiar with eegdash workflow

### Use Method 2 (Preprocessed) if:
- You want slightly faster training
- You'll train multiple models on same data
- You want better domain adaptation (subject-based splitting)
- You have sufficient disk space

---

## Full Training Commands

### Method 1: Direct Loading

```bash
# Train all 3 models (no preprocessing needed)
python3 train_domain_adaptation_direct.py --challenge c1 --epochs 100
python3 train_domain_adaptation_direct.py --challenge c2 --epochs 100

# Note: Cross-Task and Hybrid are similar but not yet implemented for direct loading
# For now, use Method 2 for those models
```

### Method 2: Preprocessed

```bash
# Preprocess once
python3 preprocess_data_for_advanced_models.py --challenges c1,c2

# Train all 3 models
./train_all_advanced_models.sh
```

---

## Recommendation

**Start with Method 1 (Direct) for Domain Adaptation**:
```bash
# Quick test
python3 train_domain_adaptation_direct.py --challenge c1 --epochs 10 --mini

# Full training
python3 train_domain_adaptation_direct.py --challenge c1 --epochs 100
python3 train_domain_adaptation_direct.py --challenge c2 --epochs 100
```

This way you can start training immediately and see results faster!

---

## Master Training Script

The `train_all_advanced_models.sh` script runs:

```bash
# Domain Adaptation (C1 and C2)
python3 train_domain_adaptation.py --challenge c1 --epochs 100 --batch_size 64 \
    --lr 1e-3 --lambda_mmd 0.1 --lambda_entropy 0.01 --lambda_adv 0.1
python3 train_domain_adaptation.py --challenge c2 --epochs 100 --batch_size 64 \
    --lr 1e-3 --lambda_mmd 0.1 --lambda_entropy 0.01 --lambda_adv 0.1

# Cross-Task (C1 and C2)
python3 train_cross_task.py --challenge c1 --epochs 100 --pretrain_epochs 50
python3 train_cross_task.py --challenge c2 --epochs 100 --pretrain_epochs 50

# Hybrid (C1 and C2)
python3 train_hybrid.py --challenge c1 --epochs 100 --lambda_mmd 0.1 --lambda_entropy 0.01
python3 train_hybrid.py --challenge c2 --epochs 100 --lambda_mmd 0.1 --lambda_entropy 0.01

# Plus creates submissions automatically!
```

So yes, it's exactly the same as running those commands individually, but with automatic submission creation after each model completes.
