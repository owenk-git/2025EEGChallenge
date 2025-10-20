# 🚀 START HERE - Master Guide

## Welcome to NeurIPS 2025 EEG Challenge

This is your complete setup for the competition. Everything is ready to go!

---

## ⚡ Quick Start (5 Minutes)

### 1. Test the pipeline works:
```bash
python train.py -c 1 -o -m --max 5 -e 3
```
**Expected**: Training starts, loads ~10 subjects from mini dataset

### 2. If test passes, run full training:
```bash
python train.py -c 1 -o -e 100
python train.py -c 2 -o -e 100
```
**Expected**: Trains on ALL 3,387 subjects (12-24 hours)

### 3. Create submission:
```bash
python create_submission.py \
  --model_c1 checkpoints/c1_best.pth \
  --model_c2 checkpoints/c2_best.pth
```
**Output**: `YYYYMMDD_HHMM_trained_submission.zip`

### 4. Submit to Codabench:
Upload ZIP to: https://www.codabench.org/competitions/9975/

**That's it!** You're competing! 🎯

---

## 📚 Documentation Guide

### 🔰 Getting Started (Read These First)

| Document | Purpose | Time |
|----------|---------|------|
| [docs/guides/TRAIN_NOW.md](docs/guides/TRAIN_NOW.md) | Quick reference commands | 2 min |
| [docs/guides/DATA_SETUP.md](docs/guides/DATA_SETUP.md) | How data loading works | 5 min |
| [README.md](README.md) | Project overview | 5 min |

### 💡 Understanding the Changes

| Document | Purpose | Time |
|----------|---------|------|
| [docs/reference/ANSWERS_TO_YOUR_QUESTIONS.md](docs/reference/ANSWERS_TO_YOUR_QUESTIONS.md) | Why we changed defaults | 3 min |
| [docs/reference/BEFORE_VS_AFTER.md](docs/reference/BEFORE_VS_AFTER.md) | What changed visually | 3 min |
| [docs/reference/ALL_DATA_STREAMING_SUMMARY.md](docs/reference/ALL_DATA_STREAMING_SUMMARY.md) | Technical details | 10 min |

### 🧠 Deep Strategy (Read When Ready)

| Document | Purpose | Time |
|----------|---------|------|
| [docs/strategy/ULTRATHINK_DATA_STRATEGY.md](docs/strategy/ULTRATHINK_DATA_STRATEGY.md) | Data & validation strategy | 30 min |
| [docs/strategy/FUTURE_STRATEGY_ROADMAP.md](docs/strategy/FUTURE_STRATEGY_ROADMAP.md) | Week-by-week plan to win | 30 min |
| [docs/strategies/EXPLORATION_STRATEGY.md](docs/strategies/EXPLORATION_STRATEGY.md) | 10 experiments to run | 15 min |

### 🏗️ Project Structure

| Document | Purpose | Time |
|----------|---------|------|
| [docs/reference/PROJECT_ORGANIZATION.md](docs/reference/PROJECT_ORGANIZATION.md) | File organization | 10 min |

---

## 🎯 Current Status

### ✅ What's Ready:
- [x] ALL data streaming (3,387 subjects from R1-R11 + NC)
- [x] Subject-wise validation (prevents data leakage)
- [x] Comprehensive metrics (7 metrics including competition NRMSE)
- [x] Training pipeline with validation
- [x] Best model checkpointing
- [x] Submission creation
- [x] Exploration framework (10 experiments)
- [x] K-Fold CV script
- [x] Ensemble submission script

### 📊 Competition Status:
- **Current Best**: 1.14 NRMSE (previous submission with random weights)
- **SOTA Target**: 0.978 NRMSE
- **Gap to Close**: -14.2%
- **Submissions Remaining**: 25
- **Days Remaining**: ~12 (until Nov 2, 2025)

### 🎯 Next Goals:
1. Run baseline training on ALL data → Target: 1.05 NRMSE
2. Run exploration grid → Find best hyperparameters
3. Train optimized model → Target: 1.00 NRMSE
4. Create ensemble → Target: 0.95 NRMSE
5. Beat SOTA → Target: <0.978 NRMSE

---

## 🗺️ Strategy Overview

### Week 1: Foundation ✅ (DONE)
- Setup ALL data streaming
- Implement validation
- Create exploration framework

### Week 2: Exploration (Now → Next Week)
- Run baseline: `python train.py -c 1 -o -e 100`
- Run 10 experiments: `bash scripts/run_exploration_streaming.sh`
- Find best hyperparameters
- **Target**: 1.05 NRMSE

### Week 3: Optimization
- Train on full dataset with best params
- Implement data augmentation
- Try architecture variants
- **Target**: 1.00 NRMSE

### Week 4: Ensemble
- Train multiple models (5+ models)
- K-Fold cross-validation
- Create ensemble submission
- **Target**: 0.95 NRMSE (beat SOTA!)

### Week 5+: Advanced
- Test-Time Augmentation (TTA)
- Pseudo-labeling
- Multi-task learning
- **Target**: <0.90 NRMSE (top of leaderboard!)

---

## 🏃 What to Do Right Now

### Option 1: Quick Test (5 minutes)
```bash
python train.py -c 1 -o -m --max 5 -e 3
```
**Purpose**: Verify everything works
**Time**: 5 minutes
**Next**: Run full training if test passes

---

### Option 2: Baseline Training (Today/Tonight)
```bash
# Both challenges in parallel (use 2 GPUs)
CUDA_VISIBLE_DEVICES=0 python train.py -c 1 -o -e 100 &
CUDA_VISIBLE_DEVICES=1 python train.py -c 2 -o -e 100 &
wait
```
**Purpose**: Get first competitive submission
**Time**: 12-24 hours
**Expected**: Beat 1.14, reach ~1.05

---

### Option 3: Start Exploration (This Week)
```bash
# Run all 10 exploration experiments
bash scripts/run_exploration_streaming.sh

# Analyze results
python scripts/compare_exploration.py
```
**Purpose**: Find best hyperparameters
**Time**: 2-3 days
**Expected**: Identify best direction

---

## 📖 How to Use This Codebase

### For Training:

**Main script**: `train.py`

```bash
# Basic usage
python train.py -c <challenge> -o -e <epochs>

# With options
python train.py -c 1 -o -e 100 \
  --max 100 \        # Limit to 100 subjects (for testing)
  --dropout 0.3 \    # Higher regularization
  --lr 5e-4 \        # Lower learning rate
  --batch_size 64    # Larger batches
```

**Flags**:
- `-c 1` or `-c 2`: Challenge number
- `-o`: Use official dataset (streams ALL data)
- `-m`: Mini mode (small subset for testing)
- `-e N`: Number of epochs
- `--max N`: Limit to N subjects
- `--num X`: Experiment number (for tracking)

---

### For K-Fold CV:

**Script**: `train_kfold.py`

```bash
python train_kfold.py -c 1 -o -e 150 --n_folds 5
```

**Output**:
- 5 models: `checkpoints_kfold/c1_fold0_best.pth` ... `c1_fold4_best.pth`
- Results: `results_kfold/c1_kfold_results.json`

---

### For Creating Submissions:

**Single model**:
```bash
python create_submission.py \
  --model_c1 checkpoints/c1_best.pth \
  --model_c2 checkpoints/c2_best.pth
```

**Ensemble**:
```bash
python create_ensemble_submission.py \
  --models checkpoints_kfold/c1_*.pth checkpoints_kfold/c2_*.pth \
  --method weighted \
  --weights 0.25 0.20 0.30 0.15 0.10
```

---

### For Analysis:

**Compare experiments**:
```bash
python scripts/compare_exploration.py
```

**View logs**:
```bash
cat experiments/experiments.json
```

**Check results**:
```bash
ls results/exp_*/
```

---

## 🔍 Verification Checklist

When you run training, verify these:

### ✅ Data Loading:
```
📦 Loading EEGChallengeDataset
   Task: contrastChangeDetection
   Release: all (ALL RELEASES - 3,387 subjects)  ← ✅ Must say "all"
   Mini: False 🌐 (FULL dataset)                 ← ✅ Must say "False"
✅ Loaded 67231 recordings
   Unique subjects: 3387                         ← ✅ Must be ~3387
```

### ✅ Training Progress:
```
Epoch 1/100
  Train loss: 0.1234
  Val NRMSE: 1.0234 ⭐ (Competition Metric)      ← ✅ Should decrease
  Val Pearson: 0.4567
  Val R²: 0.3456
  ✅ Best model saved!
```

### ✅ Submission Created:
```
✅ Submission created: YYYYMMDD_HHMM_trained_submission.zip
   Size: 15.3 MB

📋 Contents:
   - submission.py (123 KB)
   - c1_weights.pth (7.5 MB)
   - c2_weights.pth (7.5 MB)
```

---

## ❓ Common Questions

### Q: How many subjects are we using?
**A**: ALL 3,387 subjects from 12 releases (R1-R11 + NC)

### Q: Is it downloading or streaming?
**A**: Streaming from S3 (small metadata cache, no raw data download)

### Q: How long does training take?
**A**:
- Quick test (5 subjects, 3 epochs): 5 minutes
- Subset (100 subjects, 50 epochs): 2 hours
- Full (3,387 subjects, 100 epochs): 12-24 hours

### Q: How many models should I train?
**A**:
- Baseline: 2 models (C1 + C2)
- Ensemble: 10 models (5 per challenge)
- Advanced: 20+ models with different strategies

### Q: What's the expected performance?
**A**:
- Baseline (100 subjects): ~1.1-1.2 NRMSE
- Full training (all data): ~1.0-1.05 NRMSE
- Ensemble (5 models): ~0.95-1.0 NRMSE
- Advanced (TTA + ensemble): ~0.90-0.95 NRMSE

---

## 🚨 Red Flags (Something's Wrong)

### ❌ Only loading 10 subjects
**Problem**: Using mini dataset
**Fix**: Remove `-m` flag

### ❌ Validation NRMSE increasing
**Problem**: Overfitting
**Fix**: Increase dropout, add regularization

### ❌ Train loss not decreasing
**Problem**: Learning rate too high/low
**Fix**: Try different learning rates

### ❌ Submission > 100MB
**Problem**: Too large
**Fix**: Check if accidentally including data

---

## 🎁 Key Features

### Data:
- ✅ Streams ALL 3,387 subjects from S3
- ✅ Subject-wise splitting (prevents data leakage)
- ✅ Multiple split strategies (train/val, train/val/test, K-fold)

### Training:
- ✅ Validation during training
- ✅ Best model checkpointing (based on val NRMSE)
- ✅ Comprehensive metrics (7 metrics tracked)
- ✅ Experiment tracking (all configs logged)

### Submission:
- ✅ Single model submission
- ✅ Ensemble submission (multiple models)
- ✅ Verified format (matches competition requirements)

### Strategy:
- ✅ Exploration framework (10 experiments)
- ✅ K-Fold CV support
- ✅ Ensemble methods
- ✅ Complete roadmap to beat SOTA

---

## 🏆 Success Metrics

### Minimum Success:
- ✅ Beat current best (1.14)
- ✅ Reach ~1.05 NRMSE

### Good Success:
- ✅ Reach ~1.00 NRMSE
- ✅ Ensemble working

### Great Success:
- ✅ Beat SOTA (0.978)
- ✅ Reach ~0.95 NRMSE

### Exceptional Success:
- ✅ Top 3 on leaderboard
- ✅ Reach <0.90 NRMSE

---

## 📞 Quick Reference

### Essential Commands:
```bash
# Quick test
python train.py -c 1 -o -m --max 5 -e 3

# Full training
python train.py -c 1 -o -e 100

# Create submission
python create_submission.py \
  --model_c1 checkpoints/c1_best.pth \
  --model_c2 checkpoints/c2_best.pth
```

### Essential Documents:
- [docs/guides/TRAIN_NOW.md](docs/guides/TRAIN_NOW.md) - Commands
- [docs/guides/DATA_SETUP.md](docs/guides/DATA_SETUP.md) - Data setup
- [docs/strategy/FUTURE_STRATEGY_ROADMAP.md](docs/strategy/FUTURE_STRATEGY_ROADMAP.md) - Strategy

### Essential URLs:
- Competition: https://www.codabench.org/competitions/9975/
- Website: https://eeg2025.github.io/
- Leaderboard: https://www.codabench.org/competitions/9975/#/results

---

## 🚀 Ready to Compete!

Everything is set up. All you need to do is:

### Today:
1. Quick test: `python train.py -c 1 -o -m --max 5 -e 3` (5 min)
2. Verify it works ✅

### This Week:
1. Full training: `python train.py -c 1 -o -e 100` (12-24 hrs)
2. Create submission
3. Submit to Codabench
4. Beat 1.14! 🎯

### Next 2-3 Weeks:
1. Run exploration
2. Optimize hyperparameters
3. Create ensemble
4. Beat SOTA (0.978)! 🏆

**You have all the tools. Now execute and win!** 🚀

---

**Last Updated**: 2024-11-15
**Next Step**: Run the quick test!
