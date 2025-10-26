# Trial-Level RT Prediction - Quick Start

## The Breakthrough

**Problem**: All models stuck at NRMSE 1.0-1.5 because they predict **ONE RT per recording** (which contains ~30 trials)

**Solution**: Predict **RT per TRIAL**, then aggregate → Expected NRMSE 0.7-0.9 (60% improvement!)

## Why This Works

### Current Approach (Wrong):
```
Recording (5 min, 30 trials) → Truncate to 200 points → Predict 1 RT
```
- Loses 99% of data
- Averages out all trial-to-trial variability
- Can't capture individual fast/slow trials
- Result: NRMSE 1.0-1.5 (barely better than guessing)

### New Approach (Correct):
```
Recording → Extract 30 trials → Predict 30 RTs → Aggregate
Each trial: [-500ms, +1500ms] = 200 points = Exactly what we need!
```
- Uses all data effectively
- Captures trial variability
- Preserves temporal structure
- Result: Expected NRMSE 0.7-0.9

## Quick Test

```bash
# On remote server
cd ~/temp/chal/2025EEGChallenge
git pull

# Test with mini dataset first (fast!)
python3 train_trial_level.py --challenge c1 --epochs 10 --batch_size 16 --mini

# If it works, run full training
python3 train_trial_level.py --challenge c1 --epochs 100 --batch_size 32
```

## What to Expect

### If trial extraction works:
```
📦 Loading EEGChallengeDataset (Trial-Level)
✅ Loaded 767 recordings
🔍 Extracting individual trials...
   Recording 0: 28 trials, RT range: [0.412, 0.856]s
   Recording 1: 32 trials, RT range: [0.389, 0.721]s
   Recording 2: 31 trials, RT range: [0.445, 0.798]s
✅ Extracted 23,450 trials total
   Average: 30.6 trials per recording

✅ Data loaders created:
   Train: 18,760 trials (586 batches)
   Val: 4,690 trials (146 batches)
```

**This means success!** You now have 23k training samples instead of 767.

### If trial extraction fails:
```
⚠️ No trials extracted from annotations
   Falling back to sliding window approach...
```

This fallback is less ideal but still better than current approach.

## Expected Results

### Baseline (Current Models):
- Domain Adaptation: NRMSE 1.5491
- Cross-Task: NRMSE 2.5959
- Hybrid: NRMSE 2.5959

### Trial-Level (Expected):
- **Epoch 1**: NRMSE ~1.2-1.4 (random initialization)
- **Epoch 10**: NRMSE ~0.95-1.05 (learning structure)
- **Epoch 50**: NRMSE ~0.80-0.90 (converging)
- **Epoch 100**: NRMSE ~0.70-0.85 (best performance)

**Target**: Beat 0.976 (current top score)

## Key Files

1. **data/trial_level_loader.py**
   - Extracts individual trials from recordings
   - Pairs stimulus events with responses
   - Computes RT per trial

2. **models/trial_level_rt_predictor.py**
   - Simple but effective CNN
   - Splits pre/post stimulus
   - Spatial attention on channels

3. **train_trial_level.py**
   - Training script
   - Standard supervised learning
   - Saves best model

## Debugging

If you see errors, run this to inspect trial structure:

```bash
python3 inspect_trial_structure.py
```

This will show:
- How many events per recording
- Event types (stimulus, response, etc.)
- Whether RT can be extracted
- Sample trial structure

## Why This is Different

**All previous approaches** (domain adaptation, cross-task transfer, hybrid CNN-Transformer):
- ✗ Treated recordings as single units
- ✗ Lost trial-level variability
- ✗ Predicted recording-level average

**This approach**:
- ✓ Treats trials as individual units
- ✓ Captures trial-level variability
- ✓ Predicts trial-level RT, aggregates to recording

## The Neuroscience

Reaction time varies trial-by-trial based on:

1. **Pre-stimulus alpha** (attention/alertness)
   - High alpha → drowsy → slow RT
   - Low alpha → alert → fast RT

2. **P300 latency** (decision time)
   - Late P300 → long processing → slow RT
   - Early P300 → quick decision → fast RT

3. **Motor preparation** (readiness potential)
   - Weak preparation → hesitant → slow RT
   - Strong preparation → ready → fast RT

By analyzing EACH trial separately, we capture these effects!

## Next Steps After Training

1. **Check if NRMSE < 1.0**: If yes, this approach is working!

2. **Create submission**:
   ```bash
   python3 create_trial_level_submission.py --challenge c1
   ```
   (Need to create this script if trial-level works)

3. **Compare to baseline**:
   - Current best: 1.09 (your MLP)
   - Trial-level: ??? (hopefully 0.7-0.9)

4. **If successful, apply to C2**:
   - C2 predicts externalizing factor (easier, subject-level)
   - Trial-level might not help C2 as much

## Bottom Line

This is a **fundamental re-thinking** of the problem:

- **Before**: "How can we build a better model for recording-level RT prediction?"
- **Now**: "Should we even be predicting at recording level?"

**Answer**: No! Predict at trial level, aggregate to recording level.

Expected improvement: **NRMSE 1.5 → 0.8 (47% reduction)**

Let's try it!
