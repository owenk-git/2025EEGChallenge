# Quick Start Guide

## 🚀 Complete Workflow

### 1. Setup (5 minutes)

```bash
# Clone repo
git clone https://github.com/owenk-git/2025EEGChallenge.git
cd 2025EEGChallenge

# Setup environment
bash setup_env.sh
conda activate eeg2025
```

### 2. Get Data (10-60 minutes)

**Option A: Mini Dataset (Recommended for testing)**
```bash
# Manual download from https://nemar.org
# Search: "HBN-EEG R1_mini_L100"
# Extract to: ./data/R1_mini_L100/
```

**Option B: Full Dataset from S3 (100-250 GB per release)**
```bash
pip install awscli
python scripts/download_mini_data.py --release 1 --from_s3
```

### 3. Train Models (Hours to days depending on data)

```bash
# Train Challenge 1
python train.py \
  --challenge 1 \
  --data_path ./data/R1_mini_L100 \
  --epochs 50 \
  --batch_size 32

# Train Challenge 2
python train.py \
  --challenge 2 \
  --data_path ./data/R1_mini_L100 \
  --epochs 50 \
  --batch_size 32
```

### 4. Create Submission (1 minute)

```bash
python create_submission.py \
  --model_c1 checkpoints/c1_best.pth \
  --model_c2 checkpoints/c2_best.pth

# Output: YYYYMMDD_HHMM_trained_submission.zip
```

### 5. Submit to Codabench

1. Go to https://www.codabench.org/competitions/9975/
2. Upload the generated ZIP file
3. Wait for evaluation (1-2 hours)

## 📊 Expected Results

With mini dataset training:
- **First submission:** ~1.3-1.5 overall
- **After tuning:** ~1.1-1.2 overall

With full dataset training (multiple releases):
- **Target:** < 1.0 overall
- **Goal:** < 0.96 (top 3)

## 💡 Tips

1. **Start with mini dataset** (R1_mini_L100) for fast experimentation
2. **Train on multiple releases** (R1-R11) for better generalization
3. **Increase epochs** (100-200) for full training
4. **Monitor training loss** - should decrease steadily
5. **Save checkpoints** regularly to avoid losing progress

## 🐛 Troubleshooting

**Dataset not loading?**
- Check file format (.set or .bdf)
- Verify BIDS structure
- Try with mini dataset first

**Out of memory?**
- Reduce batch_size (e.g., 16 or 8)
- Use single release for training
- Check GPU memory: `nvidia-smi`

**Training loss not decreasing?**
- Check learning rate (try 1e-4 or 5e-4)
- Verify data loading correctly
- Check for NaN values

## 📧 Need Help?

- Open an issue on GitHub
- Contact: neurips2025-eeg-competition@googlegroups.com
