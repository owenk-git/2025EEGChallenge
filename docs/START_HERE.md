# 🚀 START HERE - Quick Training Guide

Everything is ready! Just run the scripts in order.

---

## ✅ Setup Complete Checklist

- [x] Python 3.10 environment created
- [x] PyTorch with CUDA installed
- [x] All packages installed (eegdash, braindecode, etc.)
- [x] Run scripts created and ready
- [x] Project structure organized

---

## 🎯 What You Need to Do

### Step 1: Quick Test (5 minutes)

```bash
bash run_test.sh
```

**Expected output:**
```
🧪 Quick Test: Verifying setup and data loading...
📊 Loading data...
📦 Using official EEGChallengeDataset
✅ Loaded 5 recordings

Epoch 1/3
Training: 100%|████| 2/2 [00:10<00:00]
Train Loss: 0.0234  ← Should be ~0.02

Epoch 2/3
Train Loss: 0.0189  ← Should DECREASE!

Epoch 3/3
Train Loss: 0.0145  ← Should keep decreasing!

✅ Training complete!
```

**If this works → You're ready for full training!**

---

### Step 2: Full Training (overnight)

**Option A: Sequential (Safer)**
```bash
# Tonight: Train Challenge 1 (~8-12 hours)
bash run_challenge_1.sh

# Tomorrow night: Train Challenge 2 (~8-12 hours)
bash run_challenge_2.sh
```

**Option B: Parallel (If you have 2 GPUs)**
```bash
# Train both at once (~8-12 hours total)
bash run_both_parallel.sh
```

**Option C: Fully Automated**
```bash
# Runs test + train C1 + train C2 + create submission
bash run_full_pipeline.sh
```

---

### Step 3: Create Submission

```bash
bash create_submission.sh
```

**Output:**
```
📦 Creating Submission Package
✅ Created submission.py
✅ Copied C1 weights: checkpoints/c1_best.pth
✅ Copied C2 weights: checkpoints/c2_best.pth
✅ Submission created: 20251020_1234_trained_submission.zip
   Size: 2.3 MB

📤 Upload to Codabench!
```

---

### Step 4: Upload to Competition

1. Go to: https://www.codabench.org/competitions/9975/
2. Click "Participate" → "Submit"
3. Upload your `*_submission.zip` file
4. Wait for results!

**Expected score: 0.95-1.00 (beats your current 1.14!)** ✅

---

## 📋 All Available Scripts

| Script | What It Does | Time |
|--------|--------------|------|
| **run_test.sh** | Quick test (START HERE!) | 5 min |
| **run_challenge_1.sh** | Train Challenge 1 | 8-12 hrs |
| **run_challenge_2.sh** | Train Challenge 2 | 8-12 hrs |
| **run_both_parallel.sh** | Train both (2 GPUs) | 8-12 hrs |
| **create_submission.sh** | Create ZIP | <1 min |
| **run_full_pipeline.sh** | Everything automated | ~24 hrs |
| **run_large_scale.sh** | Max performance (200 subjects) | 48+ hrs |

See [SCRIPTS_GUIDE.md](SCRIPTS_GUIDE.md) for details.

---

## 🔍 Monitoring Training

### Check if running:
```bash
ps aux | grep python
nvidia-smi  # GPU usage
```

### Watch progress:
```bash
# If using run_both_parallel.sh
tail -f logs/challenge_1.log
tail -f logs/challenge_2.log

# Otherwise, just watch the terminal
```

### Check checkpoints:
```bash
ls -lh checkpoints/
```

You should see files appear like:
```
c1_best.pth       ← Best Challenge 1 model
c2_best.pth       ← Best Challenge 2 model
c1_epoch10.pth    ← Checkpoint every 10 epochs
c1_epoch20.pth
...
```

---

## 🎯 Quick Commands Reference

```bash
# 1. Test (5 min)
bash run_test.sh

# 2. Train C1 (overnight)
bash run_challenge_1.sh

# 3. Train C2 (next night)
bash run_challenge_2.sh

# 4. Create submission
bash create_submission.sh

# 5. Upload ZIP to Codabench!
```

---

## 🆘 Troubleshooting

### "CUDA out of memory"
Edit script and change:
```bash
--batch_size 32  →  --batch_size 16  # or 8
```

### "Loss not decreasing"
- Verify data loads (should show "Loaded X recordings")
- Try lower LR: `--lr 0.0005`

### "Can't find checkpoints"
Check if training actually completed:
```bash
ls checkpoints/
```

### Want to run in background:
```bash
nohup bash run_challenge_1.sh > train.log 2>&1 &
tail -f train.log  # Monitor
```

---

## 📊 What to Expect

### Training Loss:
```
Epoch 1:   Loss = 0.025  (starting)
Epoch 10:  Loss = 0.012  (learning)
Epoch 50:  Loss = 0.006  (converging)
Epoch 100: Loss = 0.004  (converged)
```

**Key: Loss should DECREASE!** If it stays flat, something's wrong.

### Competition Score:
- Current best: 1.14 (C1: 1.45, C2: 1.01)
- Expected with training: **0.95-1.00** ✅
- SOTA to beat: 0.978

**You should beat your current best!**

---

## 🏆 Next Steps After First Submission

Once you get your first score:

1. **If score > 1.14:** Something wrong, check training logs
2. **If score 0.95-1.00:** Great! Now try:
   - More subjects: Edit script `--max_subjects 200`
   - More epochs: `--epochs 150`
   - Hyperparameter tuning
   - Ensemble (train 3 models)

3. **If score < 0.95:** Excellent! Keep improving:
   - Scale up to 200+ subjects
   - Try ensemble approach
   - Add inference strategies
   - Aim for top 3!

---

## 📁 Project Structure

```
BCI/
├── START_HERE.md              ← You are here!
├── SCRIPTS_GUIDE.md           ← Detailed script documentation
├── README.md                  ← Project overview
│
├── run_test.sh               ⭐ START with this
├── run_challenge_1.sh
├── run_challenge_2.sh
├── run_both_parallel.sh
├── create_submission.sh
├── run_full_pipeline.sh
├── run_large_scale.sh
│
├── train.py                   ← Main training script
├── create_submission.py       ← Creates ZIP
├── models/eegnet.py          ← Model architecture
├── data/                     ← Data loaders
├── checkpoints/              ← Saved models (created during training)
└── logs/                     ← Training logs
```

---

## 🎯 TL;DR - Just Run This

```bash
# Test first
bash run_test.sh

# If test passes, run full pipeline
bash run_full_pipeline.sh

# Wait ~24 hours, then upload the ZIP file!
```

---

## 📞 Documentation

- **Quick Start:** [README.md](README.md)
- **Script Details:** [SCRIPTS_GUIDE.md](SCRIPTS_GUIDE.md)
- **Complete Setup:** [docs/READY_TO_TRAIN.md](docs/READY_TO_TRAIN.md)
- **Strategies:** [docs/strategies/](docs/strategies/)

---

**Ready?** Run: `bash run_test.sh` 🚀
