# 2025 EEG Challenge - Training & Submission Pipeline

Complete training and inference pipeline for the NeurIPS 2025 EEG Foundation Challenge.

## 🎯 Quick Start

```bash
# 1. Setup environment
bash setup_env.sh
conda activate eeg2025

# 2. Download mini dataset (for testing)
python scripts/download_mini_data.py --release 1

# 3. Train models
python train.py --challenge 1 --data_path ./data/R1_mini_L100

# 4. Create submission
python create_submission.py --model_c1 checkpoints/c1_best.pth --model_c2 checkpoints/c2_best.pth
```

## 📊 Current Status

- **Best Score:** 1.14 (C1: 1.45, C2: 1.01)
- **Target:** 0.958 (Top: C1: 0.944, C2: 0.999)
- **Gap:** 35% improvement needed in C1

## 🧠 Model: EEGNeX with Sigmoid-Inside-Classifier

Proven architecture achieving 1.14 overall score.

## 📁 Structure

```
.
├── README.md              # This file
├── environment.yml        # Conda dependencies
├── setup_env.sh          # Setup script
├── train.py              # Training pipeline
├── create_submission.py  # Package for submission
├── models/               # Model architectures
├── data/                 # Data loaders
├── scripts/              # Utility scripts
└── checkpoints/          # Saved weights
```

## 🔗 Links

- **Competition:** https://www.codabench.org/competitions/9975/
- **Paper:** https://arxiv.org/abs/2506.19141
- **Dataset:** HBN-EEG (100 Hz, 129 channels)

## 📜 License

MIT License
