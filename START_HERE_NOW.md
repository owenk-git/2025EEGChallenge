# 🚀 START HERE: What to Run First

## ✅ Pre-Flight Checklist

Before starting, verify your setup:

```bash
# 1. Check you're in the right directory
pwd
# Should show: /path/to/BCI

# 2. Check Python environment
python --version  # or python3 --version
# Should show: Python 3.10.x

# 3. Check CUDA available
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
# Should show: CUDA: True

# 4. Check GPU
nvidia-smi
# Should show your GPU

# 5. Pull latest changes
git pull
```

---

## 🎯 Option 1: QUICK TEST (Recommended First!)

**Time:** ~5 minutes
**Purpose:** Verify everything works before long training

```bash
# Test with 5 subjects, 3 epochs (very fast!)
python train.py -c 1 -d dummy -o -m --max 5 -e 3 --num 999

# What to expect:
# - Data loading works
# - Training starts
# - Validation runs with subject-wise split
# - Metrics displayed (NRMSE, Pearson, R², etc.)
# - Model saved
# - Experiment logged
```

**If this fails:** Fix the error before proceeding!
**If this works:** Continue to Option 2

---

## 🎯 Option 2: BASELINE EXPERIMENT (Recommended Second!)

**Time:** ~2 hours
**Purpose:** Establish baseline performance, ready for first submission

```bash
# Experiment 1: Challenge 1 Baseline
CUDA_VISIBLE_DEVICES=0 python train.py \
  -c 1 \
  -d dummy \
  -o \
  --max 50 \
  -e 50 \
  --drop 0.2 \
  --lr 1e-3 \
  -b 32 \
  --num 1

# Experiment 2: Challenge 2 Baseline
CUDA_VISIBLE_DEVICES=0 python train.py \
  -c 2 \
  -d dummy \
  -o \
  --max 50 \
  -e 50 \
  --drop 0.2 \
  --lr 1e-3 \
  -b 32 \
  --num 2
```

**What happens:**
```
======================================================================
🚀 Training EEG Challenge 1
======================================================================
Device: cuda

📊 Loading data...
📦 Using official EEGChallengeDataset
   Task: contrastChangeDetection
   Mini: True
   Limiting to 50 subjects
   Validation split: 20.0%
   Using subject-wise split (prevents data leakage)
✅ Loaded 150 recordings
   Unique subjects: 50
   Using 150 recordings
   Train: 120 recordings from 40 subjects (80.0%)
   Val:   30 recordings from 10 subjects (20.0%)

🧠 Creating model for Challenge 1
Parameters: 50,000

🎯 Training for 50 epochs
======================================================================

Epoch 1/50
  Train Loss: 0.1234
  Val Loss:  0.1456
  Val NRMSE: 1.2345 ⭐ (Competition Metric)
  Val RMSE:  0.0456
  Val MAE:   0.0321
  Val Pearson: 0.7234
  Val R²: 0.5123
  ✅ Saved best model (NRMSE: 1.2345)

... (continues for 50 epochs)

======================================================================
✅ Training complete! Best val NRMSE: 1.1234 (epoch 32)
   Best val RMSE: 0.0389
   Best val MAE:  0.0267
📁 Model saved to: checkpoints/c1_best.pth
======================================================================

💾 Saved predictions and metrics to: results/exp_1/c1_results.pt

✅ Experiment #1 logged to experiments/experiments.json
```

---

## 📊 After Training: Analyze Results

```bash
# 1. Compare experiments
python scripts/compare_exploration.py

# Output:
# 🔬 EXPLORATION RESULTS (Exp 1-10)
# ====================================================================================================
# Exp   Challenge  Group           Subjects   Epochs   Dropout    LR         Batch    Val NRMSE
# ----------------------------------------------------------------------------------------------------
# 1     C1         Baseline        50         50       0.20       1.0e-03    32       1.2345
# 2     C2         Baseline        50         50       0.20       1.0e-03    32       1.0543
# ====================================================================================================

# 2. View detailed metrics
python experiments/analyze_experiments.py

# 3. Check predictions
python -c "
import torch
r = torch.load('results/exp_1/c1_results.pt')
print(f'Val NRMSE: {r[\"metrics\"][\"nrmse\"]:.4f}')
print(f'Val Pearson: {r[\"metrics\"][\"pearson_r\"]:.4f}')
print(f'Val R²: {r[\"metrics\"][\"r2\"]:.4f}')
"
```

---

## 🚢 Create First Submission

```bash
# After both Exp 1 and 2 are done, create submission
python create_submission.py \
  --model_c1 checkpoints/c1_best.pth \
  --model_c2 checkpoints/c2_best.pth \
  --output baseline_exp1_2.zip

# What happens:
# ======================================================================
# 📦 Creating Submission Package
# ======================================================================
# ✅ Created submission.py
# ✅ Copied C1 weights: checkpoints/c1_best.pth
# ✅ Copied C2 weights: checkpoints/c2_best.pth
#
# ✅ Submission package created: baseline_exp1_2.zip
#    Size: 123.4 KB
# ======================================================================
# 🚀 Ready to submit!
#    Upload baseline_exp1_2.zip to Codabench
# ======================================================================
```

**Submission structure (VERIFIED ✅):**
```
baseline_exp1_2.zip
├── submission.py       # Contains model architecture and Submission class
├── c1_weights.pth      # Trained Challenge 1 weights
└── c2_weights.pth      # Trained Challenge 2 weights
```

**This is CORRECT!** The submission.py will load trained weights.

---

## 🌐 Submit to Codabench

1. **Go to:** https://www.codabench.org/competitions/9975/
2. **Click:** "Participate" → "Submit"
3. **Upload:** `baseline_exp1_2.zip`
4. **Wait:** 1-2 hours for evaluation
5. **Check:** Leaderboard for your score

**Expected first submission:**
- Val NRMSE: C1=1.15-1.35, C2=1.05-1.15
- Test NRMSE: Should be similar (±10%)
- If test >> val: Overfitting (increase dropout)
- If test << val: Lucky validation split (good!)

---

## 🎯 Option 3: RUN ALL EXPLORATIONS (If You Have Time)

**Time:** ~24 hours (overnight)
**Purpose:** Test all 5 hypotheses at once

```bash
# Run all 10 experiments (Groups 1-5)
./scripts/run_exploration.sh

# Or run on specific GPU:
./scripts/run_exploration.sh 1  # Use GPU 1
```

**What it does:**
- Exp 1-2: Baseline (50 subj, 50 epochs)
- Exp 3-4: More data (200 subj, 100 epochs)
- Exp 5-6: High dropout (100 subj, 100 epochs, drop=0.4)
- Exp 7-8: Lower LR (100 subj, 150 epochs, lr=5e-4)
- Exp 9-10: Large batch (100 subj, 100 epochs, batch=64)

**After completion:**
```bash
# Analyze all results
python scripts/compare_exploration.py --plot

# Creates:
# - Summary table
# - Best direction identified
# - Recommendations for next steps
# - Plot: experiments/exploration_results.png
```

---

## 🎯 Option 4: RUN BY GROUPS (Flexible)

**If you want to iterate faster:**

```bash
# Group 1: Baseline (2 hours)
./scripts/run_exp_group.sh 1
python scripts/compare_exploration.py
# Submit, observe leaderboard

# Group 2: More Data (8 hours)
./scripts/run_exp_group.sh 2
python scripts/compare_exploration.py
# Submit best so far

# Group 3: High Dropout (4 hours)
./scripts/run_exp_group.sh 3
# ... etc
```

---

## 📋 Decision Tree

```
START
  ↓
Option 1: Quick Test (5 min) ✅
  ↓
  Works? ──No──→ Fix errors
  ↓ Yes
  ↓
Option 2: Baseline (2 hours) ✅
  ↓
  ├─ Val NRMSE < 1.1 → 🎉 Excellent! Submit immediately
  ├─ Val NRMSE 1.1-1.3 → ✅ Good, continue explorations
  └─ Val NRMSE > 1.3 → ⚠️  Check setup
  ↓
Create Submission ✅
  ↓
Submit to Codabench 🌐
  ↓
Observe Leaderboard 📊
  ↓
  ├─ Test ≈ Val → ✅ Validation reliable, iterate fast
  ├─ Test > Val → ⚠️  Overfitting, increase regularization
  └─ Test < Val → 🎉 Test easier, be aggressive
  ↓
Option 3 or 4: More Explorations 🔬
  ↓
Find Best Direction 🎯
  ↓
Exploitation Phase 🚀
  ↓
Beat SOTA! 🏆
```

---

## ⚠️ Common Issues & Solutions

### Issue 1: Import Error
```bash
ModuleNotFoundError: No module named 'eegdash'
```
**Solution:**
```bash
pip install eegdash braindecode s3fs boto3 mne pandas torch scipy
```

### Issue 2: CUDA Out of Memory
```bash
RuntimeError: CUDA out of memory
```
**Solution:**
```bash
# Reduce batch size
python train.py ... -b 16  # instead of 32
```

### Issue 3: Data Loading Slow
```bash
# First epoch takes forever
```
**Solution:**
```bash
# Data is downloading from S3, wait for cache
# Future runs will be faster
```

### Issue 4: No Checkpoints Folder
```bash
FileNotFoundError: checkpoints/c1_best.pth
```
**Solution:**
```bash
# Training didn't complete or crashed
# Check training logs
# Rerun training
```

---

## 📁 Important Files

**After training, you'll have:**
```
checkpoints/
  ├── c1_best.pth          # Best C1 model (use for submission)
  ├── c2_best.pth          # Best C2 model (use for submission)
  ├── c1_epoch10.pth       # C1 checkpoint at epoch 10
  └── c2_epoch10.pth       # C2 checkpoint at epoch 10

results/
  ├── exp_1/
  │   └── c1_results.pt    # Predictions & metrics for Exp 1
  └── exp_2/
      └── c2_results.pt    # Predictions & metrics for Exp 2

experiments/
  ├── experiments.json     # All experiment configs & results
  └── EXPERIMENT_LOG.md    # Human-readable notes

baseline_exp1_2.zip        # Submission file
```

---

## 🚀 Quick Commands Reference

```bash
# 1. Quick test (5 min)
python train.py -c 1 -d dummy -o -m --max 5 -e 3 --num 999

# 2. Baseline experiments (2 hours each)
python train.py -c 1 -d dummy -o --max 50 -e 50 --num 1
python train.py -c 2 -d dummy -o --max 50 -e 50 --num 2

# 3. Analyze
python scripts/compare_exploration.py

# 4. Create submission
python create_submission.py \
  --model_c1 checkpoints/c1_best.pth \
  --model_c2 checkpoints/c2_best.pth

# 5. All explorations (overnight)
./scripts/run_exploration.sh

# 6. Run specific group
./scripts/run_exp_group.sh 1  # Baseline
./scripts/run_exp_group.sh 2  # More data
# ... etc
```

---

## ✅ Summary: YOUR NEXT COMMAND

```bash
# If you have 5 minutes: TEST
python train.py -c 1 -d dummy -o -m --max 5 -e 3 --num 999

# If you have 2 hours: BASELINE + SUBMIT
python train.py -c 1 -d dummy -o --max 50 -e 50 --num 1
python train.py -c 2 -d dummy -o --max 50 -e 50 --num 2
python create_submission.py --model_c1 checkpoints/c1_best.pth --model_c2 checkpoints/c2_best.pth
# Then submit baseline.zip to Codabench

# If you have 24 hours: FULL EXPLORATION
./scripts/run_exploration.sh
# Then analyze and submit best
```

**Recommendation:** Start with BASELINE (Option 2) - 2 hours to first submission! 🚀
