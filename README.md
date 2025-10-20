# NeurIPS 2025 EEG Challenge

Training pipeline for NeurIPS 2025 EEG Foundation Challenge.

**Current Best:** 1.14 (Sub 3)
**SOTA to Beat:** 0.978
**Submissions Remaining:** 25 of 35
**Days Left:** ~12 (ends Nov 2, 2025)

---

## 🚀 NEW USERS: Start Here!

**👉 [START_HERE_MASTER.md](START_HERE_MASTER.md)** - Complete guide with everything you need!

**Quick Commands**:
```bash
# Test (5 min)
python train.py -c 1 -o -m --max 5 -e 3

# Train (12-24 hrs)
python train.py -c 1 -o -e 100
```

**New setup streams ALL 3,387 subjects automatically!** ✅

---

## 📚 Documentation

### Quick Start
- **[START_HERE_MASTER.md](START_HERE_MASTER.md)** - Complete guide (start here!)
- **[docs/guides/TRAIN_NOW.md](docs/guides/TRAIN_NOW.md)** - Quick command reference
- **[docs/guides/DATA_SETUP.md](docs/guides/DATA_SETUP.md)** - Data loading explained
- **[docs/INDEX.md](docs/INDEX.md)** - Complete documentation index

### Strategy
- **[docs/strategy/FUTURE_STRATEGY_ROADMAP.md](docs/strategy/FUTURE_STRATEGY_ROADMAP.md)** - Week-by-week plan
- **[docs/strategy/ULTRATHINK_DATA_STRATEGY.md](docs/strategy/ULTRATHINK_DATA_STRATEGY.md)** - Data & validation strategy
- **[docs/strategies/EXPLORATION_STRATEGY.md](docs/strategies/EXPLORATION_STRATEGY.md)** - 10 experiments to run

### Reference
- **[docs/reference/PROJECT_ORGANIZATION.md](docs/reference/PROJECT_ORGANIZATION.md)** - File structure
- **[docs/reference/ANSWERS_TO_YOUR_QUESTIONS.md](docs/reference/ANSWERS_TO_YOUR_QUESTIONS.md)** - Recent changes Q&A
- **[docs/reference/ULTRATHINK_COMPLETE_SUMMARY.md](docs/reference/ULTRATHINK_COMPLETE_SUMMARY.md)** - Complete analysis

---

## ⚡ Quick Start

### 1. Install Dependencies
```bash
pip install eegdash braindecode s3fs boto3 mne pandas torch
```

### 2. Test the Pipeline (5 minutes)
```bash
python train.py -c 1 -o -m --max 5 -e 3
```
**Verify**: Should load ~10 subjects and start training

### 3. Train on ALL Data (12-24 hours)
```bash
# Challenge 1
python train.py -c 1 -o -e 100

# Challenge 2
python train.py -c 2 -o -e 100
```
**Note**: Streams ALL 3,387 subjects from R1-R11 + NC automatically

### 4. Create Submission
```bash
python create_submission.py \
  --model_c1 checkpoints/c1_best.pth \
  --model_c2 checkpoints/c2_best.pth
```

### 5. Submit to Codabench
Upload ZIP to: https://www.codabench.org/competitions/9975/

---

## 🎯 Training Commands

### Quick Test (Mini dataset)
```bash
python train.py -c 1 -o -m --max 5 -e 3
```
**Uses**: Small mini subset (~5 minutes)

### Medium Training (100 subjects)
```bash
python train.py -c 1 -o --max 100 -e 50
```
**Uses**: 100 subjects from full dataset (~2 hours)

### Full Training (ALL 3,387 subjects)
```bash
python train.py -c 1 -o -e 100
```
**Uses**: Complete competition dataset (~12-24 hours)

**Key flags**:
- `-c 1` or `-c 2`: Challenge number
- `-o`: Use official dataset (streams from S3)
- `-m`: Mini mode (small subset for testing)
- `--max N`: Limit to N subjects
- `-e N`: Number of epochs
- NO `-m` and NO `--max` = uses ALL 3,387 subjects!

---

## 📁 Project Structure

```
BCI/
├── 📄 Core Scripts
│   ├── train.py                           # Main training
│   ├── train_kfold.py                     # K-Fold CV
│   ├── create_submission.py               # Single model submission
│   └── create_ensemble_submission.py      # Ensemble submission
│
├── 📚 Documentation
│   ├── START_HERE_MASTER.md               # Master guide
│   ├── README.md                          # This file
│   └── docs/
│       ├── INDEX.md                       # Complete doc index
│       ├── guides/                        # How-to guides
│       ├── strategy/                      # Strategic planning
│       ├── strategies/                    # Implementation strategies
│       └── reference/                     # Reference docs
│
├── 📂 Implementation
│   ├── models/                            # Model architectures
│   │   └── eegnet.py                      # EEGNeX model
│   ├── data/                              # Data loaders
│   │   └── official_dataset_example.py    # Official dataset loader
│   ├── utils/                             # Utilities
│   │   └── metrics.py                     # 7 comprehensive metrics
│   └── scripts/                           # Helper scripts
│       ├── run_exploration_streaming.sh   # Run experiments
│       └── compare_exploration.py         # Analyze results
│
└── 📂 Auto-Generated (during training)
    ├── checkpoints/                       # Trained models
    ├── checkpoints_kfold/                 # K-Fold models
    ├── results/                           # Training results
    ├── experiments/                       # Experiment logs
    ├── submissions/                       # Generated ZIPs
    └── data_cache/                        # Cached metadata
```

---

## 📊 Model Architecture

**EEGNeX** (Temporal → Spatial → Feature convolutions):
- **Input**: 129 channels × 200 timepoints (2 seconds @ 100 Hz)
- **Temporal Conv**: Extract temporal patterns (64 kernel)
- **Spatial Conv**: Depthwise spatial filtering (129 channels → 16 features)
- **Separable Conv**: Feature refinement
- **Classifier**: Sigmoid-inside design for [0,1] output

**Parameters**: ~157k
**Training Time**: ~1-2 sec/batch (on GPU)

---

## 🎯 Competition Strategy

### Week 1: Foundation ✅ (DONE)
- [x] Setup ALL data streaming (3,387 subjects)
- [x] Implement subject-wise validation
- [x] Add comprehensive metrics
- [x] Create frameworks

### Week 2: Exploration (This Week)
- [ ] Run baseline training
- [ ] Test 10 exploration experiments
- [ ] Find best hyperparameters
- **Target**: Beat 1.14, reach ~1.05 NRMSE

### Week 3: Optimization
- [ ] Train on full dataset with best params
- [ ] Implement data augmentation
- [ ] Try architecture variants
- **Target**: Reach ~1.00 NRMSE

### Week 4: Ensemble
- [ ] Train 5 models (different seeds)
- [ ] K-Fold cross-validation
- [ ] Create ensemble submission
- **Target**: Beat SOTA (0.978), reach ~0.95 NRMSE

### Week 5+: Advanced
- [ ] Test-Time Augmentation
- [ ] Pseudo-labeling
- [ ] Multi-task learning
- **Target**: Top 3 on leaderboard (<0.90 NRMSE)

See [docs/strategy/FUTURE_STRATEGY_ROADMAP.md](docs/strategy/FUTURE_STRATEGY_ROADMAP.md) for complete roadmap.

---

## ✨ Key Features

### Data:
- ✅ Streams ALL 3,387 subjects from S3 (R1-R11 + NC)
- ✅ Subject-wise splitting (prevents data leakage)
- ✅ Multiple split strategies (train/val, train/val/test, K-fold)
- ✅ No manual S3 configuration needed

### Training:
- ✅ Validation during training
- ✅ Best model checkpointing (by validation NRMSE)
- ✅ 7 comprehensive metrics tracked
- ✅ All experiments logged automatically

### Submission:
- ✅ Single model submission
- ✅ Ensemble submission (multiple models)
- ✅ Verified format (matches competition requirements)

### Strategy:
- ✅ 10 exploration experiments ready
- ✅ K-Fold CV support
- ✅ Ensemble methods implemented
- ✅ Complete roadmap to beat SOTA

---

## 🔍 Verification

When training, verify these outputs:

### ✅ Data Loading:
```
📦 Loading EEGChallengeDataset
   Release: all (ALL RELEASES - 3,387 subjects)  ← Must say "all"
   Mini: False 🌐 (FULL dataset)                 ← Must say "False"
   Unique subjects: 3387                         ← Must be ~3387
```

### ✅ Training Progress:
```
Epoch 1/100
  Train loss: 0.1234
  Val NRMSE: 1.0234 ⭐ (Competition Metric)      ← Should decrease
  Val Pearson: 0.4567
  Val R²: 0.3456
  ✅ Best model saved!
```

### ✅ Submission Created:
```
✅ Submission created: YYYYMMDD_HHMM_trained_submission.zip
   Size: 15.3 MB
   Contents: submission.py, c1_weights.pth, c2_weights.pth
```

---

## 🆘 Troubleshooting

### Issue: Only loading 10 subjects
**Fix**: Remove `-m` flag (that's mini mode)

### Issue: "eegdash not installed"
**Fix**: `pip install eegdash braindecode s3fs boto3 mne`

### Issue: Validation NRMSE increasing
**Fix**: Increase dropout, add regularization

### Issue: Training too slow
**Fix**: Reduce `--max` subjects for testing, or use GPU

See [docs/guides/DATA_SETUP.md](docs/guides/DATA_SETUP.md) for complete troubleshooting.

---

## 📊 Expected Performance

| Stage | NRMSE | Time | vs SOTA |
|-------|-------|------|---------|
| Baseline (100 subj) | ~1.10 | 2 hrs | +12.5% |
| Full training | ~1.05 | 24 hrs | +7.4% |
| Optimized | ~1.00 | 24 hrs | +2.2% |
| Ensemble (5 models) | ~0.95 | 5 days | **-2.9% (Beat!)** |
| Advanced | <0.90 | 2 weeks | **-8.0%** 🏆 |

---

## 📝 Citation

```bibtex
@misc{eeg2025challenge,
  title={NeurIPS 2025 EEG Foundation Challenge},
  author={EEG Challenge Organizers},
  year={2025},
  url={https://eeg2025.github.io/}
}
```

---

## 🔗 Links

- **Competition**: https://www.codabench.org/competitions/9975/
- **Website**: https://eeg2025.github.io/
- **Leaderboard**: https://www.codabench.org/competitions/9975/#/results

---

## 📞 Quick Reference

```bash
# Quick test (5 min)
python train.py -c 1 -o -m --max 5 -e 3

# Full training (12-24 hrs)
python train.py -c 1 -o -e 100
python train.py -c 2 -o -e 100

# Create submission
python create_submission.py \
  --model_c1 checkpoints/c1_best.pth \
  --model_c2 checkpoints/c2_best.pth

# K-Fold CV (for ensemble)
python train_kfold.py -c 1 -o -e 150 --n_folds 5

# Ensemble submission
python create_ensemble_submission.py \
  --models checkpoints_kfold/c1_*.pth \
  --method weighted
```

---

**Ready to compete!** Start with [START_HERE_MASTER.md](START_HERE_MASTER.md) 🚀

**Last Updated**: 2024-11-15
