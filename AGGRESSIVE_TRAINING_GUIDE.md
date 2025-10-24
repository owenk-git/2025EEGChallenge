# Aggressive Training Guide

## Goal
Beat current best: **1.11** (C1: 1.33, C2: 1.01)
Target SOTA: **0.978**

## Strategy

### C1 (Target: 1.33 → 0.90)
- **Heavy regularization**: dropout=0.35-0.40, weight_decay=0.05
- **Robust loss**: Huber loss (handles outliers)
- **Lower LR**: 0.0001 (prevent overfitting)
- **Early stopping**: patience=10
- **RT extraction**: Try median, trimmed_mean (more robust than mean)

### C2 (Target: 1.01 → 0.83)
- **EXTREME regularization**: dropout=0.45, weight_decay=0.1
- **Very low LR**: 0.00005 (personality traits shouldn't overfit)
- **Huber loss**: More robust
- **Early stopping**: patience=15
- **More epochs**: 100 (but early stop prevents overfitting)

## Quick Start (Remote Server)

### Option 1: Run All Experiments (Recommended)
```bash
git pull

# Start 5 experiments in parallel
./run_experiments.sh

# Monitor progress (in separate terminals)
tail -f logs_experiments/c1_huber_median.log
tail -f logs_experiments/c2_extreme.log

# After ~1-2 hours, find best and create submission
python find_best_and_submit.py

# Upload aggressive_best.zip to Codabench
```

### Option 2: Run Single Best Config
```bash
git pull

# Run C1 with Huber loss + heavy regularization
python train_aggressive_c1.py -e 50 -b 64 --lr 0.0001 --dropout 0.35 --loss huber

# Run C2 with EXTREME regularization
python train_aggressive_c2.py -e 100 -b 32 --lr 0.00005 --dropout 0.45 --loss huber

# Create submission
python create_submission.py \
    --model_c1 checkpoints_aggressive_c1/c1_aggressive_best.pth \
    --model_c2 checkpoints_aggressive_c2/c2_aggressive_best.pth \
    --output aggressive_single.zip
```

## What's Different from Before?

| Aspect | Previous (1.11) | Aggressive |
|--------|----------------|------------|
| **Dropout** | 0.20 | 0.35-0.45 |
| **Weight Decay** | 0.01 | 0.05-0.10 |
| **Learning Rate** | 0.0005 | 0.0001-0.00005 |
| **Loss** | MSE | Huber (robust) |
| **Early Stop** | ❌ | ✅ (patience 10-15) |
| **Batch Size** | 128 | 32-64 (smaller) |
| **RT Method** | mean | median, trimmed_mean |

## Expected Improvements

**Conservative estimate:**
- C1: 1.33 → 1.15 (improvement: -0.18)
- C2: 1.01 → 0.95 (improvement: -0.06)
- **Overall: 1.00** (improvement: -0.11)

**Optimistic estimate:**
- C1: 1.33 → 1.05 (improvement: -0.28)
- C2: 1.01 → 0.88 (improvement: -0.13)
- **Overall: 0.93** (improvement: -0.18, **BEATS SOTA!**)

## Experiments Being Run

### C1 Experiments
1. **c1_huber_median**: Huber loss + median RT aggregation
2. **c1_huber_trimmed**: Huber loss + trimmed mean RT (remove outliers)
3. **c1_mae_wide**: MAE loss + wider output range [0.4, 1.6]

### C2 Experiments
1. **c2_extreme**: EXTREME regularization (dropout=0.45, wd=0.1)
2. **c2_mae**: MAE loss with slightly less regularization

Best checkpoint from each will be automatically selected.

## Monitoring

Check logs:
```bash
# C1
tail -f logs_experiments/c1_huber_median.log
tail -f logs_experiments/c1_huber_trimmed.log
tail -f logs_experiments/c1_mae_wide.log

# C2
tail -f logs_experiments/c2_extreme.log
tail -f logs_experiments/c2_mae.log
```

Check GPU usage:
```bash
nvidia-smi
```

## Troubleshooting

**If training gets NaN:**
- Reduce learning rate by 10x
- Increase gradient clipping: `--clip_grad 0.3`

**If NRMSE doesn't improve:**
- Let it run longer (early stopping will handle it)
- Check logs for convergence issues

**If one experiment crashes:**
- Others will continue running
- Can run failed one separately later

## After Training

1. Run `python find_best_and_submit.py`
2. Check expected score
3. If score > 1.11, DON'T SUBMIT (worse than current)
4. If score < 1.11, upload to Codabench!
5. If score < 0.978, **YOU BEAT SOTA!** 🎉

## Next Steps if This Doesn't Work

1. Try ensemble of multiple models
2. Try attention mechanisms
3. Try different architectures (Transformer, TCN)
4. Investigate test set distribution differences
5. Manual hyperparameter tuning based on results

---

**Remember:** You have limited submissions. Only submit if expected score < 1.11!
