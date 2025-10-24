# ULTRATHINK Investigation - Beating SOTA 0.978

## Current Situation
- Best score: **1.12** (C1: 1.36, C2: 1.01)
- SOTA target: **0.978** (C1: 0.928, C2: 1.00)
- Gap: C1 needs -0.43, C2 needs -0.01

## Key Mystery

**Random weights (Oct 14) got C1: 0.93**
**Trained weights now get C1: 1.36**

Training made it WORSE! Why?

## Investigation Plan

### CRITICAL: Check Dataset Targets

Run this first:
```bash
python investigate_c1_targets.py
```

This checks if `EEGChallengeDataset` provides `.y` attribute with correct RT targets!

If `.y` exists → **USE THOSE INSTEAD OF OUR RT EXTRACTION**

### What Challenge 1 Actually Predicts

From official website (eeg2025.github.io):
> "Predict response time from EEG data during the Contrast Change Detection (CCD) task"

So RT IS correct. But:
1. What RT exactly? (mean/median/per-trial?)
2. How is it normalized?
3. Dataset might provide the exact values used by competition

### Hypotheses to Test

#### Hypothesis 1: Dataset provides correct targets
- `dataset.y` might exist and contain competition-official RT values
- These would be preprocessed/normalized correctly
- **CHECK THIS FIRST!**

#### Hypothesis 2: RT normalization is wrong
- We use: (RT - 1.0) / 1.0 → range [0.228, 0.758]
- Random model outputs: ~1.0 (from sigmoid + scaling)
- Maybe true targets are ~1.0 not ~0.5?
- **Try different normalization ranges**

#### Hypothesis 3: RT aggregation method
- We use: mean RT across all trials
- Maybe should use: median, mode, 10th percentile?
- Different subjects might need different aggregation
- **Try different methods**

#### Hypothesis 4: Random initialization magic
- Random model with bias=0.5 outputs values around 1.0
- If true RT targets happen to be ~1.0, random works!
- Trained model learns wrong targets → worse performance
- **Ensemble random models with different seeds**

#### Hypothesis 5: Description table has scores
- Columns: `contrastchangedetection_1/2/3`
- These might be performance scores, not just "available"
- **Check what these columns contain**

## Action Items

### Immediate (Do Now)

1. **Run investigation script**
   ```bash
   python investigate_c1_targets.py
   ```

2. **Check results for:**
   - Does `dataset.y` exist?
   - What do `contrastchangedetection_X` columns contain?
   - What do random models output?

### Based on Results

#### If dataset.y EXISTS:
```python
# Update official_dataset_example.py to use it!
if hasattr(self.eeg_dataset, 'y') and self.eeg_dataset.y is not None:
    target_value = self.eeg_dataset.y[actual_idx]  # USE THIS!
```
Retrain immediately - this will likely fix C1!

#### If dataset.y DOESN'T exist:

**Option A: Try different RT normalizations**
```python
# Current: (RT - 1.0) / 1.0
# Try: (RT - min_rt) / (max_rt - min_rt)
# Try: No normalization (raw seconds)
# Try: Z-score normalization
```

**Option B: Ensemble random models**
```python
# Create 10 random initializations
# Average their predictions
# Random got 0.93, ensemble might get <0.90!
```

**Option C: Use behavioral scores from description**
```python
# If contrastchangedetection_X columns have numeric scores
# Use those instead of extracted RT
```

### For Challenge 2 (Easy Fix)

C2 is at 1.01, needs 1.00. Try:

**Quick wins:**
```bash
# Lower dropout
python strategy2.py -c 2 -e 200 -b 128 --dropout 0.17

# Or adjust learning rate
python strategy2.py -c 2 -e 200 -b 128 --lr 0.0003

# Or less augmentation
python strategy2.py -c 2 -e 200 -b 128 --no_augmentation
```

## Timeline

**Hour 1:** Run investigation, analyze results
**Hour 2:** Implement fix based on findings
**Hour 3:** Train new models
**Hour 4:** Create submission and test

## Success Metrics

**Minimum viable:**
- C1: < 1.00 (currently 1.36)
- C2: < 1.00 (currently 1.01)
- Overall: < 1.00 (currently 1.12)

**Target:**
- C1: < 0.93 (match SOTA)
- C2: < 1.00 (match SOTA)
- Overall: < 0.978 (BEAT SOTA!)

## Most Likely Path to Success

1. ✅ **dataset.y exists** → use it → retrain → beat SOTA
2. ❌ **dataset.y doesn't exist** → try random ensemble + C2 optimization → might reach ~1.00
3. 🤔 **Description table has scores** → use those → retrain → unknown result

Run `investigate_c1_targets.py` NOW to find out!
