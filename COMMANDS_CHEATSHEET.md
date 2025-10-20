# Quick Reference - Command Cheatsheet

## 🚀 Complete Workflow (Copy & Paste)

### 1. Setup (First Time Only)
```bash
git clone https://github.com/owenk-git/2025EEGChallenge.git
cd 2025EEGChallenge
conda env create -f environment.yml
conda activate eeg2025
```

### 2. Download Data (First Time Only)
```bash
# Manual: Download from https://nemar.org
# Search: "HBN-EEG R1_mini_L100"
# Extract to: ./data/R1_mini_L100/
```

### 3. Train Models
```bash
# Challenge 1
python train.py --challenge 1 --data_path ./data/R1_mini_L100 --epochs 50

# Challenge 2
python train.py --challenge 2 --data_path ./data/R1_mini_L100 --epochs 50
```

### 4. Create Submission
```bash
python create_submission.py \
  --model_c1 checkpoints/c1_best.pth \
  --model_c2 checkpoints/c2_best.pth
```

### 5. Submit
Upload `YYYYMMDD_HHMM_trained_submission.zip` to Codabench

---

## 📋 Common Commands

### Training Variations
```bash
# Longer training
python train.py --challenge 1 --data_path ./data/R1_mini_L100 --epochs 100

# Smaller batch (if out of memory)
python train.py --challenge 1 --data_path ./data/R1_mini_L100 --batch_size 16

# Different learning rate
python train.py --challenge 1 --data_path ./data/R1_mini_L100 --lr 0.0005

# CPU only
CUDA_VISIBLE_DEVICES="" python train.py --challenge 1 --data_path ./data/R1_mini_L100
```

### Check Progress
```bash
# List checkpoints
ls -lh checkpoints/

# Check if training completed
tail checkpoints/training.log

# Test model loading
python -c "from models.eegnet import create_model; m = create_model('c1'); print('OK')"
```

### Environment
```bash
# Activate environment
conda activate eeg2025

# Deactivate
conda deactivate

# Update dependencies
pip install -r requirements.txt --upgrade
```

---

## ⚡ Quick Start (Absolute Minimal)

```bash
# Everything in one go:
git clone https://github.com/owenk-git/2025EEGChallenge.git && \
cd 2025EEGChallenge && \
conda env create -f environment.yml && \
conda activate eeg2025

# Then download data to ./data/R1_mini_L100/ and run:
python train.py --challenge 1 --data_path ./data/R1_mini_L100 --epochs 50 && \
python train.py --challenge 2 --data_path ./data/R1_mini_L100 --epochs 50 && \
python create_submission.py --model_c1 checkpoints/c1_best.pth --model_c2 checkpoints/c2_best.pth
```

---

## 🎯 Expected Files After Training

```
checkpoints/
├── c1_best.pth         ← Challenge 1 trained weights
├── c2_best.pth         ← Challenge 2 trained weights
├── c1_epoch10.pth      ← Checkpoint at epoch 10
├── c1_epoch20.pth      ← Checkpoint at epoch 20
└── ...

./
└── 20251014_1700_trained_submission.zip  ← Ready to upload!
```

---

## 🐛 Troubleshooting One-Liners

```bash
# Module not found
pip install -r requirements.txt

# Check conda env
conda env list

# Check GPU
python -c "import torch; print(torch.cuda.is_available())"

# Test dataset path
ls ./data/R1_mini_L100/sub-*/eeg/*.set | head

# Check checkpoint
python -c "import torch; c = torch.load('checkpoints/c1_best.pth'); print(c.keys())"
```

---

## 📊 Track Your Progress

```bash
# After each submission, note:
Submission: YYYYMMDD_HHMM_trained_submission.zip
Training: R1_mini_L100, 50 epochs
C1 Score: ?
C2 Score: ?
Overall: ?
Notes: First trained model
```

---

## 🔗 Quick Links

- **Repo:** https://github.com/owenk-git/2025EEGChallenge
- **Competition:** https://www.codabench.org/competitions/9975/
- **Dataset:** https://nemar.org (search "HBN-EEG R1_mini_L100")
