# NeurIPS 2025 EEG Challenge

Training pipeline for NeurIPS 2025 EEG Foundation Challenge.

**Current Best:** 1.12
**SOTA to Beat:** 0.978 (C1: 0.928, C2: 1.00)
**Submissions Used:** 12 of 35
**Deadline:** Nov 3, 2025

---

## 🚀 Quick Start

**👉 [RUN_THIS.md](RUN_THIS.md)** - Instructions for training with official method!

**Train with official RT extraction:**
```bash
# Challenge 1 - Uses official eegdash method
python train_official.py -c 1 -e 200 -b 128 -w 8

# Challenge 2 - Optimized hyperparameters
python train_official.py -c 2 -e 200 -b 128 -w 8 --dropout 0.17
```

**Expected improvement:** C1: 1.36 → 0.95, C2: 1.01 → 0.99, Overall: **~0.97** (beats SOTA!)

---

## 📚 Documentation

### Current Instructions
- **[RUN_THIS.md](RUN_THIS.md)** - **START HERE!** Training with official method
- **[START_HERE_MASTER.md](START_HERE_MASTER.md)** - Legacy guide

### Analysis & Strategy
- **[docs/analysis/ULTRATHINK_BREAKTHROUGH.md](docs/analysis/ULTRATHINK_BREAKTHROUGH.md)** - Discovery of official RT method
- **[docs/analysis/ULTRATHINK_FINDINGS.md](docs/analysis/ULTRATHINK_FINDINGS.md)** - Investigation results
- **[docs/instructions/STRATEGY2_GUIDE.md](docs/instructions/STRATEGY2_GUIDE.md)** - Strategy 2 approach
- **[docs/strategy/](docs/strategy/)** - Additional strategy documents

### Reference
- **[docs/guides/](docs/guides/)** - Training guides and data setup
- **[docs/reference/](docs/reference/)** - Project organization and Q&A
- **[docs/INDEX.md](docs/INDEX.md)** - Complete documentation index

---

## ⚡ Installation

```bash
# Install required packages
pip install eegdash braindecode s3fs boto3 mne pandas torch

# Clone repository
git clone https://github.com/owenk-git/2025EEGChallenge.git
cd 2025EEGChallenge

# Start training (see RUN_THIS.md for details)
python train_official.py -c 1 -e 200 -b 128 -w 8
```

---

## 🏆 Competition Details

- **Competition:** NeurIPS 2025 EEG Foundation Challenge
- **Link:** https://www.codabench.org/competitions/9975/
- **Website:** https://eeg2025.github.io/
- **Deadline:** November 3, 2025

### Challenges
1. **Challenge 1:** Predict response time (regression) - Target NRMSE: < 0.928
2. **Challenge 2:** Predict externalizing factor - Target NRMSE: < 1.00

### Current Progress
- C1: 1.36 NRMSE (training with official method should → 0.95)
- C2: 1.01 NRMSE (optimization should → 0.99)
- Overall: 1.12 (target: < 0.978 to beat SOTA)

---

## 💡 Key Discovery

We found the **official competition baseline** uses a different RT extraction method!

**Problem:** Our manual RT extraction → wrong targets → C1: 1.36
**Solution:** Official `annotate_trials_with_target` → correct targets → Expected C1: ~0.95

See [docs/analysis/ULTRATHINK_BREAKTHROUGH.md](docs/analysis/ULTRATHINK_BREAKTHROUGH.md) for details.

