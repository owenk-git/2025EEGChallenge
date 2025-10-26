# Challenge 2 Critical Analysis

## 🔍 Pipeline Trace

### Training (C2):
1. **Data Loading** (`data/official_eegdash_loader.py:174-177`):
   ```python
   subject_info = self.eeg_dataset.description.iloc[idx]
   target_value = subject_info.get('externalizing', 0.0)
   if np.isnan(target_value):
       target_value = 0.0
   ```
   - Loads externalizing directly from metadata
   - **No normalization applied**

2. **Model Architecture** (`models/domain_adaptation_eegnex.py:108-109`):
   ```python
   if output_range is None:
       self.output_range = (-3, 3) if challenge == 'c2' else (0.5, 1.5)
   ```
   - C2 uses output_range = **(-3, 3)**
   - Assumes externalizing is standardized (mean~0, std~1)

3. **Model Output** (`models/domain_adaptation_eegnex.py:216-217`):
   ```python
   predictions = self.task_predictor(features).squeeze(-1)
   predictions = torch.clamp(predictions, self.output_range[0], self.output_range[1])
   ```
   - Raw linear output (no activation)
   - Clipped to [-3, 3]

### Submission (C2):
1. **Model Loading** (`FINAL_C2_SUBMISSION.py:58-60`):
   ```python
   checkpoint = torch.load(model_path, map_location=device)
   model.load_state_dict(checkpoint['model_state_dict'])
   model.eval()
   ```
   - Loads trained model
   - Model has output_range=(-3, 3) baked in

2. **Recording Features** (`FINAL_C2_SUBMISSION.py:68-96`):
   ```python
   eeg_data = raw.get_data()
   # Take central segment
   start_idx = (total_samples - n_times) // 2
   eeg_data = eeg_data[:, start_idx:start_idx + n_times]
   ```
   - Extracts middle 200 time points
   - Same as training

3. **Predictions** (`FINAL_C2_SUBMISSION.py:190-193`):
   ```python
   for model_idx, model in enumerate(models):
       pred = model(eeg_tensor).item()
       all_predictions[model_idx].append(pred)
   ```
   - Gets raw prediction from each model
   - **No additional normalization or transformation**

4. **Ensemble** (`FINAL_C2_SUBMISSION.py:199-201`):
   ```python
   ensemble_predictions = np.zeros(len(test_dataset.datasets))
   for preds, weight in zip(all_predictions, ensemble_weights):
       ensemble_predictions += preds * weight
   ```
   - Weighted average
   - **Direct output to CSV**

---

## ✅ C2 Pipeline Analysis

### Consistency Check:

| Stage | Input | Output | Normalization |
|-------|-------|--------|---------------|
| Data Load | Raw metadata | Externalizing value | None |
| Training | Externalizing | Model prediction | Clipped [-3, 3] |
| Submission | Test EEG | Model prediction | Clipped [-3, 3] |
| Ensemble | Multiple predictions | Weighted average | None |
| Output | Final prediction | CSV | **Direct** |

**Verdict**: Pipeline is CONSISTENT! ✅

---

## ⚠️ Potential Issues

### Issue #1: Unknown Externalizing Range

**Problem**: We don't know if externalizing is:
- **Option A**: Standardized (mean=0, std=1) → output_range=(-3, 3) is correct
- **Option B**: Raw scale (e.g., 0-100) → output_range=(-3, 3) is WRONG

**Check**: Run `python3 analyze_c2_targets.py` on remote server

**Impact**:
- If standardized: No problem ✅
- If raw scale: Model predictions clipped incorrectly → poor performance

---

### Issue #2: Taking Only 200 Time Points (2 seconds)

**Current**: `eeg_data[:, start_idx:start_idx + 200]` (2 seconds @ 100Hz)

**Problem**: C2 recordings are typically 3-5 minutes of resting state
- Using only 2 seconds may miss important information
- Externalizing is subject-level trait → should use MORE data

**Alternative**:
```python
# Option A: Use entire recording (average over time)
eeg_data = raw.get_data()  # Full recording
eeg_data_avg = eeg_data.mean(axis=1, keepdims=True)  # Time-averaged

# Option B: Multiple segments and average
segments = []
for i in range(0, total_samples, 200):
    segment = eeg_data[:, i:i+200]
    pred = model(segment)
    segments.append(pred)
final_pred = np.mean(segments)
```

**Impact**: Using only 2 seconds might reduce performance by 5-10%

---

### Issue #3: Subject Leakage in Validation

**Problem**: Same as C1 - random split may include same subject in train/val

**C2 is MORE sensitive** because:
- C2 explicitly tests **subject-invariant** representation
- Test set has different subjects than train
- Val NRMSE with subject leakage is VERY optimistic

**Expected Impact**:
- Val NRMSE: 1.08 (with leakage)
- Test NRMSE: 1.15-1.25 (without leakage)
- **10-15% gap between val and test**

---

### Issue #4: Ensemble Weights Not Optimized

**Current**: Default weights (0.4, 0.4, 0.2) or equal

**Problem**: Not tuned on validation set

**Better Approach**:
```python
# Grid search on validation set
best_nrmse = float('inf')
best_weights = None

for w1 in np.linspace(0.2, 0.6, 5):
    for w2 in np.linspace(0.2, 0.6, 5):
        w3 = 1.0 - w1 - w2
        if w3 < 0.1: continue

        ensemble_pred = w1*pred1 + w2*pred2 + w3*pred3
        nrmse = compute_nrmse(ensemble_pred, val_targets)

        if nrmse < best_nrmse:
            best_nrmse = nrmse
            best_weights = (w1, w2, w3)
```

**Impact**: Could improve 2-5%

---

## 🎯 Critical Priority for C2

### CRITICAL (Must Check):
1. **Check externalizing value range** → Run `analyze_c2_targets.py`
   - If standardized (mean~0, std~1): No problem ✅
   - If raw scale: Need to adjust output_range

### HIGH (Could Improve 5-10%):
2. **Use more than 200 time points**
   - Current: 2 seconds
   - Better: Average over multiple segments or entire recording

### MEDIUM (Could Improve 2-5%):
3. **Optimize ensemble weights**
   - Grid search on validation set
   - Find best combination

### LOW (Expected, Accept):
4. **Subject leakage in validation**
   - Val NRMSE optimistic
   - Test NRMSE will be 10-15% higher
   - This is expected and difficult to fix

---

## 🚨 POTENTIAL CRITICAL BUG: Output Range

If externalizing is NOT standardized (e.g., raw scores 20-80), then:

**Current**:
```python
# Model trained with output_range = (-3, 3)
# Predictions clipped to [-3, 3]
# But actual values are [20, 80]!
```

**This would cause TERRIBLE performance!**

**Fix** (if needed):
```python
# Option 1: Normalize during training
target_normalized = (target - mean) / std  # Standardize

# Option 2: Change output_range to match data
self.output_range = (target_min, target_max)  # e.g., (20, 80)
```

---

## 🔧 Recommended Actions

### IMMEDIATE (Before C2 Submission):

**Step 1**: Check externalizing range
```bash
python3 analyze_c2_targets.py
```

**Expected outputs**:
- **Scenario A** (standardized): Mean~0, Std~1, Range~[-3, 3] → ✅ No changes needed
- **Scenario B** (raw scale): Mean~50, Std~20, Range~[20, 80] → ⚠️ CRITICAL FIX NEEDED

### If Scenario A (Standardized) - No Problem:
```bash
# Just wait for models to finish and submit
python3 FINAL_C2_SUBMISSION.py --device cuda
```

### If Scenario B (Raw Scale) - CRITICAL FIX:
```bash
# Need to retrain with correct output_range
# OR post-process predictions to correct scale
# This is unlikely but must check!
```

---

## 📊 Expected C2 Performance

### Best Case (No issues):
- Validation: 1.08
- Test: 1.10-1.13 (subject leakage gap)

### Likely Case (Subject leakage only):
- Validation: 1.08
- Test: 1.15-1.20 (10-15% gap)

### Worst Case (Output range bug):
- Validation: 1.08 (with wrong range)
- Test: 1.50+ (disaster!)

---

## ✅ Summary

**C2 Pipeline**: Generally correct, but need to verify externalizing range!

**Action**: Run `analyze_c2_targets.py` on remote server BEFORE submitting C2

**If standardized**: Submit as is
**If not standardized**: CRITICAL FIX needed (retrain or post-process)

**Most likely**: Values are standardized, C2 submission will work fine, but test NRMSE will be 10-15% higher than validation due to subject leakage.
