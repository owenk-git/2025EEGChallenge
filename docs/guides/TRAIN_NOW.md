# Train Now - Quick Reference

## 🎯 Simple Commands (Use These!)

### 1. Quick Test (5 minutes)
```bash
python train.py -c 1 -o -m --max 5 -e 3
```
**Tests**: Pipeline works correctly
**Data**: Mini subset (~10 subjects)

---

### 2. Full Training - Challenge 1 (12-24 hours)
```bash
python train.py -c 1 -o -e 100
```
**Trains**: ALL 3,387 subjects from R1-R11 + NC
**Output**: `checkpoints/c1_best.pth`

---

### 3. Full Training - Challenge 2 (12-24 hours)
```bash
python train.py -c 2 -o -e 100
```
**Trains**: ALL 3,387 subjects from R1-R11 + NC
**Output**: `checkpoints/c2_best.pth`

---

### 4. Create Submission
```bash
python create_submission.py \
  --model_c1 checkpoints/c1_best.pth \
  --model_c2 checkpoints/c2_best.pth
```
**Output**: `YYYYMMDD_HHMM_trained_submission.zip`

---

### 5. Submit!
Upload ZIP to: https://www.codabench.org/competitions/9975/

---

## ✅ What You Get

### Automatic Features:
- ✅ Streams ALL 3,387 subjects from S3 (R1-R11 + NC)
- ✅ No download required
- ✅ Behavioral targets auto-loaded
- ✅ Subject-wise validation split (prevents data leakage)
- ✅ Saves best model based on validation NRMSE
- ✅ Comprehensive metrics logged
- ✅ Predictions saved for analysis

### During Training You'll See:
```
📦 Loading EEGChallengeDataset
   Task: contrastChangeDetection
   Release: all (ALL RELEASES - 3,387 subjects)  ← ✅ Check!
   Mini: False 🌐 (FULL dataset)                 ← ✅ Check!
✅ Loaded 67231 recordings
   Unique subjects: 3387                         ← ✅ Check!

🧠 Creating model for Challenge 1
   Parameters: 156,873

📈 Training Progress:
Epoch 1/100
  Train loss: 0.1234
  Val NRMSE: 1.0234 ⭐ (Competition Metric)
  Val Pearson: 0.4567
  Val R²: 0.3456
  ✅ Best model saved! (improved: ∞ → 1.0234)
```

---

## 🚨 Common Issues

### Issue: "Unique subjects: 10"
**Problem**: Using mini dataset
**Fix**: Remove `-m` flag

```bash
# ❌ Wrong:
python train.py -c 1 -o -m -e 100

# ✅ Correct:
python train.py -c 1 -o -e 100
```

---

### Issue: "eegdash not installed"
**Problem**: Missing packages
**Fix**: Install packages

```bash
pip install eegdash braindecode s3fs boto3 mne pandas torch
```

---

### Issue: "Only loading one release"
**Problem**: Using old custom S3 streaming
**Fix**: Use official dataset with `-o` flag

```bash
# ❌ Wrong (old custom S3):
python train.py -c 1 -d s3://... -s -e 100

# ✅ Correct (official dataset):
python train.py -c 1 -o -e 100
```

---

## 📊 Expected Performance

### After Quick Test (5 min, mini data):
- **NRMSE**: ~1.5-2.0 (random, not meaningful)
- **Purpose**: Verify pipeline works

### After 100 Subjects (2 hours):
- **NRMSE**: ~1.2-1.4
- **Purpose**: First working baseline

### After ALL Subjects (12-24 hours):
- **NRMSE**: ~1.0-1.2
- **Target**: Beat 1.14 (current best)
- **Goal**: Reach 0.95-1.00 (near SOTA)

---

## 🎯 Training Timeline

### Today (Quick Validation)
1. Test pipeline: `python train.py -c 1 -o -m --max 5 -e 3` (5 min)
2. Verify it works ✅

### Tonight (Baseline)
1. Train C1: `python train.py -c 1 -o --max 100 -e 50` (2 hrs)
2. Train C2: `python train.py -c 2 -o --max 100 -e 50` (2 hrs)
3. Submit → Should beat 1.14 ✅

### Tomorrow (Full Training)
1. Train C1: `python train.py -c 1 -o -e 100` (12-24 hrs)
2. Train C2: `python train.py -c 2 -o -e 100` (12-24 hrs)
3. Submit → Aim for <1.0 ✅

### Next Week (Optimization)
1. Try exploration experiments (--num 1-10)
2. Ensemble multiple models
3. Add inference strategies
4. Beat SOTA (0.978)! 🏆

---

## 💡 Pro Tips

### Tip 1: Use GPU Selector
```bash
CUDA_VISIBLE_DEVICES=0 python train.py -c 1 -o -e 100
```

### Tip 2: Run Both Challenges in Parallel
```bash
# Terminal 1
CUDA_VISIBLE_DEVICES=0 python train.py -c 1 -o -e 100

# Terminal 2
CUDA_VISIBLE_DEVICES=1 python train.py -c 2 -o -e 100
```

### Tip 3: Monitor Progress
Training prints progress every epoch:
- Watch for improving validation NRMSE
- Best model saved automatically
- Check `checkpoints/c1_best.pth` exists

### Tip 4: Save Different Experiments
```bash
# Experiment 1: Baseline
python train.py -c 1 -o -e 100 --num 1

# Experiment 2: High dropout
python train.py -c 1 -o -e 100 --num 2 --drop 0.4

# Compare results in experiments/experiments.json
```

---

## 📁 Output Files

After training, you'll have:

```
BCI/
├── checkpoints/
│   ├── c1_best.pth              ← Best C1 model (use for submission)
│   ├── c2_best.pth              ← Best C2 model (use for submission)
│   ├── c1_epoch10.pth           ← Checkpoint at epoch 10
│   └── c2_epoch10.pth           ← Checkpoint at epoch 10
│
├── results/
│   ├── c1_results.pt            ← Predictions, metrics, config
│   └── c2_results.pt
│
├── experiments/
│   └── experiments.json         ← All experiments logged
│
└── data_cache/                  ← Small metadata cache (~MB)
    └── eeg_challenge/
```

---

## 🚀 Ready to Train?

### Step 1: Quick test (5 min)
```bash
python train.py -c 1 -o -m --max 5 -e 3
```

### Step 2: If test passes, run full training
```bash
python train.py -c 1 -o -e 100
python train.py -c 2 -o -e 100
```

### Step 3: Create submission
```bash
python create_submission.py \
  --model_c1 checkpoints/c1_best.pth \
  --model_c2 checkpoints/c2_best.pth
```

### Step 4: Submit and beat 1.14! 🎯

---

**That's it! Three simple commands to train on ALL competition data.** 🚀
