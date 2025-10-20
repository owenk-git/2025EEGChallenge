# Training Guide - Which Script to Use?

## 🎯 Quick Answer

**For normal training**: Use `train.py`
```bash
python train.py -c 1 -o -e 100
python train.py -c 2 -o -e 100
```

**For K-Fold ensemble**: Use `train_kfold.py` (advanced, later)
```bash
python train_kfold.py -c 1 -o -e 150 --n_folds 5
```

---

## 📊 Difference Between Scripts

### `train.py` - Main Training Script ⭐

**What it does**:
- Trains ONE model for one challenge
- Uses 80/20 train/val split
- Saves best model based on validation NRMSE
- Outputs: `checkpoints/c1_best.pth` or `c2_best.pth`

**When to use**:
- ✅ Normal training (most of the time)
- ✅ Quick iterations
- ✅ Exploration experiments
- ✅ First submission

**Example**:
```bash
# Train Challenge 1
python train.py -c 1 -o -e 100
# Output: checkpoints/c1_best.pth

# Train Challenge 2
python train.py -c 2 -o -e 100
# Output: checkpoints/c2_best.pth
```

---

### `train_kfold.py` - K-Fold Cross-Validation

**What it does**:
- Trains FIVE models for one challenge (5 folds)
- Uses K-Fold cross-validation
- Each fold gets different train/val split
- Outputs: `checkpoints_kfold/c1_fold0_best.pth` ... `c1_fold4_best.pth`

**When to use**:
- ✅ Creating ensemble (Week 4)
- ✅ Robust evaluation
- ✅ Final push for best performance

**Example**:
```bash
# Train 5 models for Challenge 1 via K-Fold
python train_kfold.py -c 1 -o -e 150 --n_folds 5
# Output: checkpoints_kfold/c1_fold0_best.pth ... c1_fold4_best.pth (5 models!)
```

---

## ✅ Confirming: Streaming, Not Downloading

### Your Commands WILL Stream (No Download!)

```bash
python train.py -c 1 -o -e 100
python train.py -c 2 -o -e 100
```

**Why these commands stream**:

1. **`-o` flag**: Uses official EEGChallengeDataset
2. **No `-m` flag**: Uses FULL dataset (not mini)
3. **No `--max`**: Uses ALL 3,387 subjects
4. **Default `release="all"`**: Loads all R1-R11 + NC automatically

### What You'll See (Confirms Streaming):

```
📦 Loading EEGChallengeDataset
   Task: contrastChangeDetection
   Release: all (ALL RELEASES - 3,387 subjects)  ← ✅ Streaming all data
   Mini: False 🌐 (FULL dataset)                 ← ✅ Not mini
✅ Loaded 67231 recordings
   Unique subjects: 3387                         ← ✅ ALL subjects loaded
   Expected ~3387 subjects (full competition dataset)
```

### Behind the Scenes:

**What gets cached locally** (~50 MB):
- `participants.tsv` files (behavioral targets)
- Channel names, sampling rates
- Subject metadata

**What streams from S3** (~100+ GB total, but streamed on-demand):
- Raw EEG recordings (.bdf files)
- Only loaded when needed for each batch
- Never fully downloaded!

**Data cache location**: `./data_cache/eeg_challenge/`
- Safe to delete anytime
- Will re-cache metadata on next run

---

## 🎯 What to Run Now

### Step 1: Quick Test (5 minutes)
```bash
python train.py -c 1 -o -m --max 5 -e 3
```
**Purpose**: Verify pipeline works
**Expected**: Loads ~10 subjects, trains 3 epochs
**Streaming**: Yes (mini subset from S3)

---

### Step 2: Full Training (12-24 hours each)
```bash
# Challenge 1 (Response time prediction)
python train.py -c 1 -o -e 100

# Challenge 2 (Externalizing factor prediction)
python train.py -c 2 -o -e 100
```
**Purpose**: Train on ALL 3,387 subjects
**Expected**: Best models saved to `checkpoints/c1_best.pth` and `c2_best.pth`
**Streaming**: Yes (ALL data from S3, no download!)

---

### Step 3: Create Submission
```bash
python create_submission.py \
  --model_c1 checkpoints/c1_best.pth \
  --model_c2 checkpoints/c2_best.pth
```
**Output**: ZIP file ready to upload

---

## 🚫 DON'T Use These (Yet)

### Don't use `-d` flag (manual S3 path):
```bash
# ❌ DON'T DO THIS (old manual way):
python train.py -c 1 -d s3://nmdatasets/... -s -e 100

# ✅ DO THIS INSTEAD (automatic):
python train.py -c 1 -o -e 100
```

### Don't use `train_kfold.py` yet:
```bash
# ❌ DON'T DO THIS YET (advanced, for Week 4):
python train_kfold.py -c 1 -o -e 150 --n_folds 5

# ✅ DO THIS FIRST (normal training):
python train.py -c 1 -o -e 100
```

**Reason**: K-Fold takes 5x longer. Use it AFTER you've optimized hyperparameters.

---

## 📊 Training Timeline

### Week 1-2 (NOW): Use `train.py`
```bash
python train.py -c 1 -o -e 100  # Get first baseline
```
**Goal**: Beat current 1.14, reach ~1.05

### Week 2-3: Still use `train.py`
```bash
python train.py -c 1 -o -e 150 --dropout 0.3 --lr 5e-4  # Optimize
```
**Goal**: Test different hyperparameters, reach ~1.00

### Week 4: Switch to `train_kfold.py`
```bash
python train_kfold.py -c 1 -o -e 150 --n_folds 5  # Ensemble
```
**Goal**: Train 5 models, create ensemble, beat SOTA (0.978)

---

## 💡 Pro Tips

### Tip 1: Run Both Challenges in Parallel
```bash
# Terminal 1
CUDA_VISIBLE_DEVICES=0 python train.py -c 1 -o -e 100

# Terminal 2
CUDA_VISIBLE_DEVICES=1 python train.py -c 2 -o -e 100
```
**Saves**: ~12 hours (run concurrently instead of sequentially)

### Tip 2: Monitor Disk Space
```bash
# Check cache size
du -sh data_cache/

# If it gets too big (>5GB), clear it
rm -rf data_cache/
```
**Note**: Only metadata cached, safe to delete

### Tip 3: Verify Streaming is Working
During training, check:
```
📦 Loading EEGChallengeDataset
   Release: all (ALL RELEASES - 3,387 subjects)  ← Must say "all"
   Mini: False 🌐 (FULL dataset)                 ← Must say "False"
   Unique subjects: 3387                         ← Must be ~3387
```

If you see:
- ❌ "Mini: True" → Remove `-m` flag
- ❌ "Unique subjects: 10" → You're using mini dataset
- ❌ Downloading messages → This is just metadata cache (OK!)

---

## 🎯 Summary

### Use `train.py` for:
- ✅ Normal training (90% of the time)
- ✅ First submission
- ✅ Hyperparameter tuning
- ✅ Quick iterations

### Use `train_kfold.py` for:
- ✅ Ensemble creation (Week 4)
- ✅ Final push for best score
- ✅ After hyperparameters optimized

### Your Commands are CORRECT ✅:
```bash
python train.py -c 1 -o -e 100  # ✅ Streams ALL data, no download
python train.py -c 2 -o -e 100  # ✅ Streams ALL data, no download
```

**These will**:
- ✅ Stream ALL 3,387 subjects from S3
- ✅ Not download raw data (only cache metadata)
- ✅ Train on complete competition dataset
- ✅ Save best models to checkpoints/

---

## 🚀 Ready to Train!

**Run this now**:
```bash
# Quick test (5 min)
python train.py -c 1 -o -m --max 5 -e 3

# If test passes, run full training
python train.py -c 1 -o -e 100
python train.py -c 2 -o -e 100
```

**You're all set!** 🎉
