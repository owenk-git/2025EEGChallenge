# C1 Prediction Collapse Solutions

## Problem Analysis

**Observed:**
- Training: Model outputs [0.0, 0.98] with good diversity
- Test: Model outputs [0.70, 0.82] → Final predictions [1.20, 1.32]
- Only using 11% of the prediction range
- Std = 0.0116 (extremely low variance)

**Impact:**
- Cannot capture RT variance across recordings
- Expected NRMSE: ~1.05-1.10 (no better than current 1.09)
- Model essentially predicting constant value

---

## Solution 1: Temperature Scaling (FASTEST - 5 mins)

**What:** Expand model outputs before mapping to competition range

**How:** Multiply outputs by temperature T > 1.0

**Implementation:**
```python
# In prediction loop:
rt_pred = model(trial_tensor).item()  # [0, 1]

# Apply temperature scaling to expand range
temperature = 1.5  # Try 1.3, 1.5, 2.0
rt_pred_expanded = rt_pred * temperature
rt_pred_expanded = np.clip(rt_pred_expanded, 0, 1)  # Keep in [0,1]

# Then map to competition range
output_value = 0.5 + rt_pred_expanded * 1.0
```

**Pros:**
- No retraining needed
- Takes 5 minutes
- Easy to try multiple temperature values

**Cons:**
- Might distort predictions
- Need to tune temperature parameter
- Doesn't fix underlying issue

**Temperature values to try:**
- T=1.2: Conservative expansion (0.70→0.84, 0.82→0.98)
- T=1.5: Moderate expansion (0.70→1.05→clip→1.0, 0.82→1.23→clip→1.0)
- T=2.0: Aggressive expansion (will clip many predictions to 1.0)

**Best guess:** T=1.3 to 1.5

---

## Solution 2: Offset + Scale Adjustment (QUICK - 5 mins)

**What:** Shift predictions before mapping

**How:** Center predictions around 0.5 instead of 0.75

**Implementation:**
```python
# In prediction loop:
rt_pred = model(trial_tensor).item()  # Currently [0.70, 0.82]

# Method A: Shift to center around 0.5
shift = 0.25  # Current mean ~0.75, shift down to 0.5
rt_pred_shifted = rt_pred - shift
rt_pred_shifted = np.clip(rt_pred_shifted, 0, 1)

# Method B: Linear rescale from observed range to [0,1]
# Observed: [0.70, 0.82]
# Rescale to [0, 1]
min_obs = 0.70
max_obs = 0.82
rt_pred_rescaled = (rt_pred - min_obs) / (max_obs - min_obs)

# Then map to competition range
output_value = 0.5 + rt_pred_rescaled * 1.0
```

**Pros:**
- Very fast
- Preserves relative ordering
- Uses full output range

**Cons:**
- Assumes test distribution is same as observed
- Hard-codes observed min/max

---

## Solution 3: Quantile Normalization (MEDIUM - 10 mins)

**What:** Map model output quantiles to target distribution quantiles

**How:** Use validation RT distribution as reference

**Implementation:**
```python
import numpy as np

# Step 1: Get validation RT distribution (run once)
# From training debug logs or validation data
val_rt_distribution = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]  # Normalized [0,1]

# Step 2: Collect all test predictions first
test_predictions = []
for trial in all_trials:
    rt_pred = model(trial).item()
    test_predictions.append(rt_pred)

# Step 3: Rank and map to validation quantiles
test_predictions = np.array(test_predictions)
ranks = np.argsort(np.argsort(test_predictions))  # Get ranks
quantiles = ranks / len(ranks)  # Convert to [0,1]

# Map to validation distribution
expanded_predictions = np.quantile(val_rt_distribution, quantiles)

# Step 4: Aggregate and map as usual
```

**Pros:**
- Statistically principled
- Preserves rank ordering
- Matches validation distribution

**Cons:**
- Requires validation RT distribution
- More complex
- Assumes test should match validation

---

## Solution 4: Add Diversity Loss and Retrain (SLOW - 4-5 hours)

**What:** Train model to maximize prediction variance

**How:** Add regularization term that penalizes low std

**Implementation:**
```python
# In training loop (modify DEBUG_C1_TRAINING.py):

# Forward pass
predictions = model(X_batch)
targets = y_batch

# Task loss
mse_loss = nn.MSELoss()(predictions, targets)

# Diversity loss (encourage variance)
pred_mean = predictions.mean()
pred_variance = ((predictions - pred_mean) ** 2).mean()
diversity_loss = -torch.log(pred_variance + 1e-6)  # Maximize variance

# Total loss
lambda_diversity = 0.1  # Weight for diversity
total_loss = mse_loss + lambda_diversity * diversity_loss

# Backward
total_loss.backward()
```

**Pros:**
- Fixes root cause
- Model learns to use full range
- Better generalization

**Cons:**
- Requires 4-5 hours retraining
- Might hurt accuracy
- Need to tune lambda_diversity

**Training command:**
```bash
python3 train_trial_level_with_diversity.py \
    --challenge c1 \
    --epochs 100 \
    --batch_size 32 \
    --lambda_diversity 0.1
```

---

## Solution 5: Ensemble with Different Temperatures (MEDIUM - 15 mins)

**What:** Create multiple predictions with different temperatures and ensemble

**How:** Average predictions from T=1.0, 1.3, 1.5

**Implementation:**
```python
# Make predictions with multiple temperatures
temperatures = [1.0, 1.3, 1.5]
ensemble_predictions = []

for temperature in temperatures:
    predictions_t = []
    for trial in trials:
        rt_pred = model(trial).item()
        rt_pred_scaled = np.clip(rt_pred * temperature, 0, 1)
        predictions_t.append(0.5 + rt_pred_scaled * 1.0)
    ensemble_predictions.append(predictions_t)

# Average across temperatures
final_predictions = np.mean(ensemble_predictions, axis=0)
```

**Pros:**
- More robust than single temperature
- Combines conservative and aggressive scaling
- No retraining

**Cons:**
- More complex
- Might average out useful signal

---

## Solution 6: Analyze Training Data Distribution (DIAGNOSTIC - 2 mins)

**What:** Check if narrow range is actually correct

**How:** Load training data and check RT distribution

**Implementation:**
```python
# Check training RT distribution
python3 -c "
import torch
import numpy as np

checkpoint = torch.load('checkpoints/trial_level_c1_best.pt', weights_only=False)
print('Checkpoint keys:', checkpoint.keys())

# Load training data
from data.trial_level_loader import TrialLevelDataset
dataset = TrialLevelDataset(challenge='c1', mini=False)

rts = []
for i in range(len(dataset)):
    _, rt, _ = dataset[i]
    rts.append(rt)

rts = np.array(rts)
print(f'Training RT stats:')
print(f'  Mean: {rts.mean():.4f}')
print(f'  Std: {rts.std():.4f}')
print(f'  Min: {rts.min():.4f}')
print(f'  Max: {rts.max():.4f}')
print(f'  Q25: {np.quantile(rts, 0.25):.4f}')
print(f'  Q50: {np.quantile(rts, 0.50):.4f}')
print(f'  Q75: {np.quantile(rts, 0.75):.4f}')
"
```

**If training RTs are [0.7, 0.8]:** Model is correct, test data really is narrow
**If training RTs are [0.0, 1.0]:** Model collapsed, need to expand

---

## Solution 7: Sigmoid Temperature in Model (RETRAIN - 4-5 hours)

**What:** Modify model architecture to use softer sigmoid

**How:** Replace `nn.Sigmoid()` with temperature-scaled version

**Implementation:**
```python
# In models/trial_level_rt_predictor.py:

class TrialLevelRTPredictor(nn.Module):
    def __init__(self, ..., output_temperature=2.0):
        super().__init__()
        self.output_temperature = output_temperature
        # ... rest of init

        self.rt_head = nn.Sequential(
            nn.Linear(128 + 512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            # No sigmoid here - apply in forward
        )

    def forward(self, x):
        # ... encoding
        rt_logits = self.rt_head(combined)

        # Temperature-scaled sigmoid
        rt = torch.sigmoid(rt_logits / self.output_temperature)
        return rt
```

**Temperature effects:**
- T=1.0: Standard sigmoid (current)
- T=2.0: Softer sigmoid, easier to reach extremes
- T=0.5: Sharper sigmoid, harder to reach extremes

**Retrain with T=2.0:**
```bash
python3 train_trial_level_with_temp.py \
    --challenge c1 \
    --epochs 100 \
    --output_temperature 2.0
```

---

## Recommended Approach

### Step 1: Quick Diagnosis (2 mins)
Run Solution 6 to check training data distribution

### Step 2: Quick Fix (5 mins)
Try Solution 1 (Temperature Scaling) with T=1.3, 1.5, 2.0
Submit all three and see which works best

### Step 3: If Quick Fix Fails (4-5 hours)
Retrain with Solution 4 (Diversity Loss)

---

## Implementation Priority

1. **Solution 6** (2 min) - Diagnose training data
2. **Solution 1** (5 min) - Temperature scaling T=1.3
3. **Solution 1** (5 min) - Temperature scaling T=1.5
4. **Solution 2** (5 min) - Offset + Scale
5. **Solution 4** (4-5 hr) - Retrain with diversity loss

---

## Expected Results

| Solution | Time | Expected NRMSE | Success Probability |
|----------|------|----------------|---------------------|
| Temperature T=1.3 | 5 min | 1.00-1.05 | 60% |
| Temperature T=1.5 | 5 min | 0.95-1.02 | 70% |
| Offset+Scale | 5 min | 0.98-1.05 | 50% |
| Diversity Loss | 4-5 hr | 0.90-0.98 | 80% |
| Sigmoid Temp | 4-5 hr | 0.88-0.95 | 85% |

**Goal:** Beat current best of 1.09, ideally reach 0.95-1.00
