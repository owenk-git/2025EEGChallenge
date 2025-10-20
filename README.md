# NeurIPS 2025 EEG Challenge

Training pipeline for NeurIPS 2025 EEG Foundation Challenge.

**Current Best:** 1.14 (Sub 3)
**SOTA to Beat:** 0.978
**Submissions Remaining:** 25 of 35
**Days Left:** ~12 (ends Nov 2, 2025)

---

## 🚀 Quick Start

```bash
# 1. Install packages (on remote server)
pip install eegdash braindecode s3fs boto3 mne pandas torch

# 2. Test data loading
python data/official_dataset_example.py

# 3. Train models
python train.py --challenge 1 --use_official --max_subjects 100 --epochs 100
python train.py --challenge 2 --use_official --max_subjects 100 --epochs 100

# 4. Create submission
python create_submission.py \
  --model_c1 checkpoints/c1_best.pth \
  --model_c2 checkpoints/c2_best.pth

# 5. Submit to Codabench!
```

**Expected:** Beat current best (1.14), aim for 0.95-1.00!

---

## 📁 Project Structure

```
BCI/
├── README.md                    # This file
├── train.py                     # Main training script
├── create_submission.py         # Create submission ZIP
│
├── models/
│   └── eegnet.py               # EEGNeX model (proven 1.14 score)
│
├── data/
│   ├── official_dataset_example.py   # Official EEGChallengeDataset wrapper
│   ├── streaming_dataset.py          # Custom S3 streaming
│   └── behavioral_streaming.py       # BIDS behavioral data loader
│
├── scripts/
│   └── test_s3_training.py     # Test S3 streaming pipeline
│
├── docs/
│   ├── README.md               # Documentation guide
│   ├── QUICKSTART.md           # 5-minute getting started
│   ├── READY_TO_TRAIN.md       # Complete setup guide
│   ├── ULTRATHINK_SUMMARY.md   # Pipeline verification
│   ├── INTEGRATION_GUIDE.md    # Data loader integration
│   │
│   └── strategies/             # Strategy guides
│       ├── STRATEGY_SUMMARY.md      # Overview
│       ├── TRAINING_STRATEGIES.md   # Training improvements
│       ├── INFERENCE_STRATEGIES.md  # Test-time improvements
│       └── ENSEMBLE_STRATEGY.md     # Multi-model approaches
│
└── archive/                    # Old submissions & experiments
```

---

## 📚 Documentation

### Start Here:
1. **[docs/QUICKSTART.md](docs/QUICKSTART.md)** - Get started in 5 minutes
2. **[docs/READY_TO_TRAIN.md](docs/READY_TO_TRAIN.md)** - Complete setup guide

### Improve Your Score:
1. **[docs/strategies/STRATEGY_SUMMARY.md](docs/strategies/STRATEGY_SUMMARY.md)** - All strategies overview
2. **[docs/strategies/TRAINING_STRATEGIES.md](docs/strategies/TRAINING_STRATEGIES.md)** - Better training
3. **[docs/strategies/INFERENCE_STRATEGIES.md](docs/strategies/INFERENCE_STRATEGIES.md)** - Test-time tricks
4. **[docs/strategies/ENSEMBLE_STRATEGY.md](docs/strategies/ENSEMBLE_STRATEGY.md)** - Multi-model approach

---

## 🎯 Training Commands

### Quick Test (10 subjects, 20 epochs):
```bash
python train.py --challenge 1 --use_official --official_mini \
  --max_subjects 10 --epochs 20
```

### Full Training (100 subjects, 100 epochs):
```bash
python train.py --challenge 1 --use_official \
  --max_subjects 100 --epochs 100 --batch_size 32 --lr 0.001
```

### With Custom S3 Streaming:
```bash
python train.py --challenge 1 \
  --data_path s3://nmdatasets/NeurIPS2025/R1_mini_L100_bdf \
  --use_streaming --max_subjects 100 --epochs 100
```

---

## 📊 Model Architecture

**EEGNeX** - Proven architecture (Sub 3: 1.14 score)

```
Input: (batch, 129 channels, 200 timepoints)
  ↓
Temporal Conv (129 → 64)
  ↓
Spatial Conv (64 → 32)
  ↓
Feature Conv (32 → 16)
  ↓
Global Pooling
  ↓
Classifier (sigmoid-inside for C1)
  ↓
Output: (batch, 1)
```

**Key innovation:** Sigmoid INSIDE classifier architecture (not in forward)

---

## 🔍 Competition Details

- **Challenge 1 (30%):** Predict response time (regression)
- **Challenge 2 (70%):** Predict externalizing factor (regression)
- **Metric:** Normalized RMSE (lower is better)
- **Overall Score:** 0.3 × C1_NRMSE + 0.7 × C2_NRMSE

**Current Scores:**
- Your best: 1.14 (C1: 1.45, C2: 1.01)
- SOTA: 0.978 (C1: 0.928, C2: 1.0)
- Top 3: ~0.988

---

## ✅ What's Working

- ✅ EEGNeX architecture (Sub 3 = 1.14)
- ✅ Sigmoid-inside-classifier
- ✅ Output scaling [0.88, 1.12] for C1
- ✅ S3 streaming (no download needed)
- ✅ Both official & custom data loaders ready
- ✅ Bandpass filter (0.5-50 Hz) implemented

---

## 🚨 Critical Fixes Applied

1. ✅ **Bandpass filter** added to custom loader (0.5-50 Hz)
2. ✅ **S3 paths** corrected (`s3://nmdatasets/NeurIPS2025/...`)
3. ✅ **Behavioral targets** loaded from real BIDS format
4. ✅ **Training pipeline** verified end-to-end

---

## 🎯 Next Steps

### Today:
1. Transfer code to remote server
2. Install packages
3. Test data loading
4. Quick training test (5-10 subjects, 3-5 epochs)

### This Week:
1. Train baseline (50-100 subjects, 50-100 epochs)
2. Submit → Beat 1.14
3. Add inference improvements (TTA, clipping)
4. Submit → Aim for <1.0

### Next Week:
1. Hyperparameter tuning
2. 3-model ensemble
3. Submit → Beat SOTA (0.978)

---

## 📝 Useful Commands

### Test Pipeline:
```bash
python data/official_dataset_example.py
python scripts/test_s3_training.py
```

### Train Both Challenges:
```bash
python train.py --challenge 1 --use_official --max_subjects 100 --epochs 100
python train.py --challenge 2 --use_official --max_subjects 100 --epochs 100
```

### Create Submission:
```bash
python create_submission.py \
  --model_c1 checkpoints/c1_best.pth \
  --model_c2 checkpoints/c2_best.pth
```

---

## 🏆 Goal

**Beat SOTA (0.978) in 1-2 weeks, aim for top 3!**

Path:
- 1.14 (current) → Train properly
- → 0.98 (more data/epochs)
- → 0.95 (+ inference strategies)
- → 0.90 (+ ensemble)
- → **Top 3!** 🥇

---

## 📞 Resources

- **Competition:** https://eeg2025.github.io
- **Leaderboard:** https://www.codabench.org/competitions/9975/
- **Docs:** [docs/README.md](docs/README.md)
- **Strategies:** [docs/strategies/](docs/strategies/)

---

**Let's beat SOTA!** 🚀
