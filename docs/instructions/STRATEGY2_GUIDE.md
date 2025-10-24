# Strategy 2: Beat 0.93/1.00

## Current Scores to Beat
- **Challenge 1**: 0.927 (NRMSE)
- **Challenge 2**: 0.998 (NRMSE)
- **Overall**: ~1.24 (current submission)

## What's New in Strategy 2

### 1. Proper RT Extraction for Challenge 1 ✅
**Problem**: Previous training used age as proxy for RT (completely wrong!)
**Solution**: Extract actual response times from task events
- Parses MNE annotations for stimulus→response timing
- Normalizes RT to [0, 1] range (0.2-2.0s)
- Falls back to age only if RT extraction fails

### 2. Data Augmentation ✅
- **Gaussian noise**: ±8% amplitude variation
- **Amplitude scaling**: 0.92-1.08× random scaling
- **DC shift**: Small baseline shifts
- **Channel dropout**: Randomly zero out 1-6 channels (robustness)

### 3. Better Optimization
- **AdamW** instead of Adam (better weight decay)
- **Cosine Annealing** with warm restarts
  - Restarts every 20→40→80 epochs
  - Helps escape local minima
- **Learning rate**: 0.0005 (default)
- **Weight decay**: 0.01 (L2 regularization)

### 4. Longer Training
- **200 epochs** (vs 100 before)
- Checkpoints every 50 epochs
- Larger batch size: 128 (better GPU utilization on RTX 4090)

## How to Run

### Challenge 1 (Target: < 0.93)
```bash
python strategy2.py -c 1 -e 200 -b 128 -w 8
```

### Challenge 2 (Target: < 1.00)
```bash
python strategy2.py -c 2 -e 200 -b 128 -w 8
```

### With specific learning rate
```bash
python strategy2.py -c 1 -e 200 -b 128 --learning_rate 0.001
```

### Disable augmentation (for testing)
```bash
python strategy2.py -c 1 -e 200 --no_augmentation
```

## Expected Results

### Challenge 1
- With RT extraction: Should beat 0.93 NRMSE
- Training time: ~1-2 hours on RTX 4090
- Watch validation NRMSE drop below 0.90

### Challenge 2
- Already close (1.01 validation)
- With augmentation + longer training: Should reach < 1.00
- Training time: ~1-2 hours on RTX 4090

## Checkpoints

Models saved to: `checkpoints_strategy2/`
- `c1_best.pth` - Best C1 model
- `c2_best.pth` - Best C2 model
- `c1_epoch50.pth`, `c1_epoch100.pth`, etc. - Intermediate checkpoints

## Create Submission

After training both challenges:

```bash
python create_submission.py \
    --model_c1 checkpoints_strategy2/c1_best.pth \
    --model_c2 checkpoints_strategy2/c2_best.pth \
    --output strategy2_submission.zip
```

Then upload `strategy2_submission.zip` to Codabench.

## Monitoring Training

Watch for:
1. **Validation NRMSE** dropping consistently
2. **Learning rate** restarting (cosine schedule)
3. **No overfitting** (train/val gap should be small)

Example good training:
```
Epoch 50/200
  Train Loss: 0.2814
  LR: 0.000123
  Val NRMSE: 0.8942 ⭐ (Target: 0.93)
  ✅ Saved best model
```

## If Results Are Still Bad

### Challenge 1
- Check RT extraction is working (not falling back to age)
- Try higher learning rate: `--learning_rate 0.001`
- Train even longer: `-e 300`

### Challenge 2
- Reduce dropout: `--dropout 0.20`
- Try different augmentation strength
- Ensemble multiple models (use different random seeds)

## Next Steps

1. **Run both trainings** (C1 and C2 in parallel on separate GPUs)
2. **Monitor progress** - should see NRMSE dropping
3. **Create submission** when both < target
4. **Submit and check leaderboard**

Target overall score: **(0.93 × 0.3) + (1.00 × 0.7) = 0.979**

This should place you in the **top tier** of the competition!
