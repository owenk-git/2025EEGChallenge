# COMPLETE TRAINING & SUBMISSION COMMANDS

## 🎯 Challenge 1 - Training

```bash
# Full training with debug logging (recommended)
python3 DEBUG_C1_TRAINING.py --challenge c1 --epochs 100 --batch_size 32

# Quick test (10 epochs, mini dataset)
python3 DEBUG_C1_TRAINING.py --challenge c1 --epochs 10 --batch_size 16 --mini
```

## 🎯 Challenge 1 - Submission

```bash
# Create submission (FIXED normalization bug!)
python3 FIXED_C1_SUBMISSION.py --device cuda

# Output: submissions/c1_trial_FIXED_*.zip
# Expected test NRMSE: 0.85-0.95
```

---

## 🎯 Challenge 2 - Verify Targets First!

```bash
# CRITICAL: Check if targets are standardized
python3 analyze_c2_targets.py

# Look for: Mean~0, Std~1
# If standardized → proceed to training
# If not → contact me for fix!
```

## 🎯 Challenge 2 - Training

```bash
# Full training with debug logging (recommended)
python3 DEBUG_C2_TRAINING.py --challenge c2 --epochs 100 --batch_size 64

# Quick test (10 epochs, mini dataset)
python3 DEBUG_C2_TRAINING.py --challenge c2 --epochs 10 --batch_size 32 --mini
```

## 🎯 Challenge 2 - Submission

```bash
# Create ensemble submission (auto-detects available models)
python3 FINAL_C2_SUBMISSION.py --device cuda

# Output: submissions/c2_ensemble_*.zip
# Expected test NRMSE: 1.10-1.20
```

---

## 📊 Complete Workflow

```bash
cd ~/temp/chal/2025EEGChallenge
git pull

# C1: Train + Submit
python3 DEBUG_C1_TRAINING.py --challenge c1 --epochs 100 --batch_size 32
python3 FIXED_C1_SUBMISSION.py --device cuda

# C2: Check + Train + Submit
python3 analyze_c2_targets.py
python3 DEBUG_C2_TRAINING.py --challenge c2 --epochs 100 --batch_size 64
python3 FINAL_C2_SUBMISSION.py --device cuda

# Upload both .zip files to competition
```

---

## 🔍 Debug Logs

Debug logs saved to: `debug_logs/`

Contains:
- Per-epoch metrics
- Prediction analysis
- Error patterns
- Correlation tracking
- Overfitting detection
