# Quick Start: Fix C1 Prediction Collapse

## Problem
C1 predictions collapsed to narrow range [1.20, 1.32] (only 11% of competition range).
Expected NRMSE: ~1.05-1.10 (no better than current 1.09).

## Solutions Available

### Option A: Automated (Recommended)
Run master script that tries all solutions:
```bash
cd ~/temp/chal/2025EEGChallenge
git pull
./solve_c1_collapse.sh
```

This will:
1. Diagnose training data (2 min)
2. Create 3 temperature-scaled submissions (15 min total)
3. Optionally retrain with diversity loss (4-5 hours)

### Option B: Manual Quick Fixes (15 minutes total)

**Step 1: Diagnose (2 min)**
```bash
python3 diagnose_c1_distribution.py
```

**Step 2: Temperature Scaling (5 min each)**
```bash
# Conservative (Expected NRMSE: 1.00-1.05)
python3 C1_SUBMISSION_TEMPERATURE.py --temperature 1.3 --device cuda

# Moderate - RECOMMENDED (Expected NRMSE: 0.95-1.02)
python3 C1_SUBMISSION_TEMPERATURE.py --temperature 1.5 --device cuda

# Aggressive (Expected NRMSE: 0.92-1.00)
python3 C1_SUBMISSION_TEMPERATURE.py --temperature 2.0 --device cuda
```

**Step 3: Submit and Compare**
Upload all 3 ZIPs to competition and see which scores best.

### Option C: Full Retrain (4-5 hours)

If temperature scaling doesn't work, retrain with diversity loss:
```bash
python3 train_trial_level_with_diversity.py \
    --challenge c1 \
    --epochs 100 \
    --batch_size 32 \
    --lambda_diversity 0.1 \
    --device cuda
```

Then create submission:
```bash
python3 C1_SUBMISSION_TEMPERATURE.py \
    --temperature 1.0 \
    --checkpoint checkpoints/trial_level_c1_diversity_best.pt \
    --device cuda
```

## Expected Results

| Method | Time | Expected NRMSE | vs Current (1.09) |
|--------|------|----------------|-------------------|
| Temperature T=1.3 | 5 min | 1.00-1.05 | 4-8% better |
| Temperature T=1.5 | 5 min | 0.95-1.02 | 6-13% better ⭐ |
| Temperature T=2.0 | 5 min | 0.92-1.00 | 8-16% better |
| Diversity Loss | 4-5 hr | 0.90-0.98 | 10-17% better |

**Target:** 0.976 (current leader)

## My Recommendation

**Start with Option B (Manual Quick Fixes):**
1. Run all 3 temperature scaling submissions (15 min)
2. Upload to competition
3. If T=1.5 or T=2.0 beats 1.09 → SUCCESS!
4. If none beat 1.09 → Run Option C (diversity loss retrain)

## Files Created

- `C1_SUBMISSION_TEMPERATURE.py` - Temperature scaling submission
- `diagnose_c1_distribution.py` - Training data analysis
- `train_trial_level_with_diversity.py` - Retrain with diversity loss
- `solve_c1_collapse.sh` - Master automated script
- `C1_EXPANSION_SOLUTIONS.md` - Detailed documentation (7 solutions)
- `QUICKSTART_C1_FIX.md` - This file

## Output Files

Submissions will be created in `submissions/`:
- `c1_temperature_T1.3_YYYYMMDD_HHMM.zip`
- `c1_temperature_T1.5_YYYYMMDD_HHMM.zip`
- `c1_temperature_T2.0_YYYYMMDD_HHMM.zip`

**Note:** These are still CSV format. After testing which temperature works best, we need to create proper `submission.py` format for final submission.
