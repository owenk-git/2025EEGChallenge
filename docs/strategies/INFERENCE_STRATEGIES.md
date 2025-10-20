# Inference Strategies (Test-Time Improvements)

Techniques to improve predictions during competition evaluation without retraining.

---

## 🎯 Key Insight

**Inference strategies improve scores WITHOUT changing training!**

You can submit the same trained weights with different inference code.

---

## 🚀 TOP 5 INFERENCE STRATEGIES

### Strategy 1: Test-Time Augmentation (TTA)

**Concept:** Make predictions on multiple augmented versions of test data, average results.

**How it works:**
```python
# submission.py - Modified inference

def predict_with_tta(model, eeg_data, n_augmentations=5):
    """
    Test-time augmentation: Predict on multiple versions, average
    """
    predictions = []

    for i in range(n_augmentations):
        # Original
        if i == 0:
            aug_data = eeg_data
        # Slight time shifts
        elif i == 1:
            aug_data = np.roll(eeg_data, 5, axis=-1)
        elif i == 2:
            aug_data = np.roll(eeg_data, -5, axis=-1)
        # Small noise
        elif i == 3:
            aug_data = eeg_data + np.random.randn(*eeg_data.shape) * 0.001
        # Horizontal flip (if symmetric)
        elif i == 4:
            aug_data = eeg_data[:, ::-1]

        # Predict
        with torch.no_grad():
            pred = model(torch.tensor(aug_data).unsqueeze(0))
            predictions.append(pred.item())

    # Average predictions
    return np.mean(predictions)
```

**Expected improvement:** 0.01-0.03 reduction in NRMSE

**Pros:**
- ✅ Easy to implement
- ✅ No retraining needed
- ✅ Proven to work in computer vision

**Cons:**
- ⚠️ Slower inference (5x predictions)
- ⚠️ May violate time limits if strict

---

### Strategy 2: Prediction Smoothing / Clipping

**Concept:** Constrain predictions to reasonable ranges.

**Why it helps:** Models sometimes predict outliers that hurt RMSE.

**Implementation:**
```python
# submission.py

def predict_with_smoothing(model, eeg_data):
    with torch.no_grad():
        pred = model(torch.tensor(eeg_data).unsqueeze(0))
        pred = pred.item()

    # Challenge 1: Response time
    # Observed range from training: [0.80, 1.20]
    # Clip predictions to safe range
    pred = np.clip(pred, 0.85, 1.15)

    # Challenge 2: Externalizing
    # Standardized score, reasonable range: [-2, 2]
    # pred = np.clip(pred, -2.0, 2.0)

    return pred
```

**Statistics-based smoothing:**
```python
# Use training set statistics
TRAIN_MEAN_C1 = 0.95
TRAIN_STD_C1 = 0.12

def predict_with_statistical_smoothing(model, eeg_data):
    pred = model(torch.tensor(eeg_data).unsqueeze(0)).item()

    # If prediction is > 2 std from mean, pull it back
    if abs(pred - TRAIN_MEAN_C1) > 2 * TRAIN_STD_C1:
        # Shrink towards mean
        pred = TRAIN_MEAN_C1 + 0.8 * (pred - TRAIN_MEAN_C1)

    return pred
```

**Expected improvement:** 0.005-0.02

---

### Strategy 3: Confidence-Based Weighting

**Concept:** Weight predictions by model confidence.

**Using model uncertainty:**
```python
def predict_with_confidence(model, eeg_data, n_samples=10):
    """
    Monte Carlo Dropout: Enable dropout during inference
    to estimate uncertainty
    """
    model.train()  # Enable dropout

    predictions = []
    for _ in range(n_samples):
        with torch.no_grad():
            pred = model(torch.tensor(eeg_data).unsqueeze(0))
            predictions.append(pred.item())

    model.eval()

    # Mean prediction
    mean_pred = np.mean(predictions)
    # Uncertainty (std deviation)
    uncertainty = np.std(predictions)

    # If high uncertainty, shrink towards population mean
    if uncertainty > 0.05:  # Threshold
        mean_pred = 0.7 * mean_pred + 0.3 * POPULATION_MEAN

    return mean_pred
```

**Expected improvement:** 0.01-0.02

---

### Strategy 4: Multi-Window Predictions

**Concept:** Use multiple time windows from same recording, average predictions.

**Why:** Test data may have multiple windows available.

**Implementation:**
```python
def predict_multi_window(model, eeg_recording):
    """
    If recording is longer than 2 seconds, extract multiple windows
    """
    sfreq = 100  # 100 Hz
    window_size = int(2 * sfreq)  # 200 samples

    # If recording has multiple windows available
    if eeg_recording.shape[-1] > window_size:
        # Extract overlapping windows
        predictions = []

        # Window 1: First 2 seconds
        window1 = eeg_recording[:, :window_size]
        pred1 = model(torch.tensor(window1).unsqueeze(0)).item()
        predictions.append(pred1)

        # Window 2: Middle 2 seconds
        start = (eeg_recording.shape[-1] - window_size) // 2
        window2 = eeg_recording[:, start:start + window_size]
        pred2 = model(torch.tensor(window2).unsqueeze(0)).item()
        predictions.append(pred2)

        # Window 3: Last 2 seconds
        window3 = eeg_recording[:, -window_size:]
        pred3 = model(torch.tensor(window3).unsqueeze(0)).item()
        predictions.append(pred3)

        # Average predictions
        return np.mean(predictions)
    else:
        # Single window
        return model(torch.tensor(eeg_recording).unsqueeze(0)).item()
```

**Expected improvement:** 0.01-0.03

---

### Strategy 5: Temperature Scaling (For C1 Sigmoid Output)

**Concept:** Adjust the "confidence" of sigmoid outputs.

**For Challenge 1 (has sigmoid output):**
```python
class EEGNeX(nn.Module):
    def __init__(self, ..., temperature=1.0):
        # ...
        self.temperature = temperature

    def forward(self, x):
        # ... convolutions ...

        # Classifier with temperature-scaled sigmoid
        x = self.classifier_linear(x)  # Before sigmoid
        x = torch.sigmoid(x / self.temperature)  # Temperature scaling

        # Scale output
        x = self.output_min + x * (self.output_max - self.output_min)
        return x
```

**Finding best temperature:**
```python
# Use validation set to find optimal temperature
temperatures = [0.5, 0.8, 1.0, 1.2, 1.5]

for temp in temperatures:
    model.temperature = temp
    val_loss = evaluate(model, val_loader)
    print(f"Temp {temp}: Loss {val_loss}")

# Use best temperature for submission
```

**Expected improvement:** 0.01-0.02

---

## 🎯 COMBINED INFERENCE STRATEGY

**Best approach:** Combine multiple strategies!

```python
# submission.py - Combined strategy

class Submission:
    def __init__(self, SFREQ, DEVICE):
        self.sfreq = SFREQ
        self.device = DEVICE
        self.model_path = load_model_path()

        # Load models
        self.model_c1 = self._load_model('c1')
        self.model_c2 = self._load_model('c2')

        # Training statistics (compute from training data)
        self.c1_mean = 0.95
        self.c1_std = 0.12
        self.c2_mean = 0.0
        self.c2_std = 0.8

    def predict_c1(self, eeg_data):
        """Challenge 1 prediction with TTA + smoothing"""
        predictions = []

        # TTA: Multiple augmentations
        for shift in [0, 5, -5]:
            aug_data = np.roll(eeg_data, shift, axis=-1)

            with torch.no_grad():
                pred = self.model_c1(
                    torch.tensor(aug_data).unsqueeze(0).to(self.device)
                )
                predictions.append(pred.item())

        # Average TTA predictions
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)

        # If high uncertainty, shrink towards mean
        if std_pred > 0.05:
            mean_pred = 0.8 * mean_pred + 0.2 * self.c1_mean

        # Clip to safe range
        mean_pred = np.clip(mean_pred, 0.85, 1.15)

        return mean_pred

    def predict_c2(self, eeg_data):
        """Challenge 2 prediction with confidence weighting"""
        predictions = []

        # Monte Carlo dropout (if using dropout)
        self.model_c2.train()  # Enable dropout
        for _ in range(5):
            with torch.no_grad():
                pred = self.model_c2(
                    torch.tensor(eeg_data).unsqueeze(0).to(self.device)
                )
                predictions.append(pred.item())
        self.model_c2.eval()

        mean_pred = np.mean(predictions)
        uncertainty = np.std(predictions)

        # Confidence-based shrinkage
        if uncertainty > 0.1:
            mean_pred = 0.7 * mean_pred + 0.3 * self.c2_mean

        # Clip to reasonable range
        mean_pred = np.clip(mean_pred, -2.5, 2.5)

        return mean_pred
```

**Expected total improvement:** 0.03-0.08 reduction in NRMSE

**Example:**
- Base model: 0.95
- + TTA: 0.93
- + Smoothing: 0.92
- + Confidence: 0.91
- **Total: 0.04 improvement!**

---

## 📊 Inference Strategy Comparison

| Strategy | Improvement | Speed | Complexity | Works For |
|----------|-------------|-------|------------|-----------|
| **Test-Time Augmentation** | 0.01-0.03 | Slow (5x) | Low | Both C1, C2 |
| **Prediction Clipping** | 0.005-0.02 | Fast | Very Low | Both C1, C2 |
| **Confidence Weighting** | 0.01-0.02 | Medium | Medium | Both C1, C2 |
| **Multi-Window** | 0.01-0.03 | Slow | Low | Both C1, C2 |
| **Temperature Scaling** | 0.01-0.02 | Fast | Medium | C1 only |

---

## 🔬 Advanced Inference Techniques

### 1. Bayesian Model Averaging

**Concept:** Treat predictions as probability distributions.

```python
# Instead of point prediction, output distribution
# Use multiple models or MC dropout
predictions = [0.94, 0.96, 0.95, 0.93, 0.97]

# Bayesian average (weighted by inverse variance)
weights = 1.0 / np.var(predictions)
bayesian_avg = np.average(predictions, weights=weights)
```

### 2. Prediction Calibration

**Concept:** Calibrate predictions to match true distribution.

```python
# Learn calibration on validation set
# If model consistently over-predicts by 0.02, subtract 0.02

# Calibration learned from validation
CALIBRATION_BIAS_C1 = -0.015  # Model predicts 0.015 higher
CALIBRATION_SCALE_C1 = 0.98   # Model predictions slightly too wide

def calibrate_prediction(pred):
    # Apply calibration
    pred = (pred - CALIBRATION_BIAS_C1) * CALIBRATION_SCALE_C1
    return pred
```

### 3. Quantile Regression

**Concept:** Predict multiple quantiles, use median for robustness.

```python
# Train model to predict 10th, 50th, 90th percentile
# At test time, use 50th percentile (median) for robustness
# This is more robust to outliers than mean
```

---

## ⚡ Quick Wins (Easy Inference Improvements)

### 1. Simple Clipping (2 minutes to implement)
```python
# In submission.py
pred_c1 = np.clip(pred_c1, 0.88, 1.12)  # Safe C1 range
pred_c2 = np.clip(pred_c2, -2.0, 2.0)   # Safe C2 range
```
**Expected:** 0.005-0.01 improvement

### 2. Basic TTA (10 minutes)
```python
# Average 3 predictions: original, +5 shift, -5 shift
preds = []
for shift in [0, 5, -5]:
    aug = np.roll(eeg_data, shift, axis=-1)
    preds.append(model(aug))
return np.mean(preds)
```
**Expected:** 0.01-0.02 improvement

### 3. Outlier Detection (5 minutes)
```python
# If prediction is extreme, shrink towards mean
if pred > 1.3 or pred < 0.7:  # Outlier for C1
    pred = 0.7 * pred + 0.3 * 0.95  # Shrink towards mean
```
**Expected:** 0.005-0.015 improvement

---

## 🎯 Recommended Inference Pipeline

### For Quick Submission (Minimal Inference):
```python
def predict(model, eeg_data):
    pred = model(eeg_data).item()
    pred = np.clip(pred, 0.88, 1.12)  # Simple clipping
    return pred
```

### For Better Submission (Medium Effort):
```python
def predict(model, eeg_data):
    # TTA with 3 augmentations
    preds = []
    for shift in [0, 5, -5]:
        aug = np.roll(eeg_data, shift, axis=-1)
        preds.append(model(aug).item())

    pred = np.mean(preds)
    pred = np.clip(pred, 0.88, 1.12)
    return pred
```

### For Best Submission (Full Strategy):
```python
def predict(model, eeg_data):
    # 1. TTA (5 augmentations)
    preds = []
    for i in range(5):
        if i == 0:
            aug = eeg_data
        elif i <= 2:
            aug = np.roll(eeg_data, [0, 5, -5][i], axis=-1)
        else:
            aug = eeg_data + np.random.randn(*eeg_data.shape) * 0.001
        preds.append(model(aug).item())

    # 2. Compute mean and uncertainty
    mean_pred = np.mean(preds)
    uncertainty = np.std(preds)

    # 3. Confidence-based shrinkage
    if uncertainty > 0.05:
        mean_pred = 0.8 * mean_pred + 0.2 * POPULATION_MEAN

    # 4. Clip to safe range
    mean_pred = np.clip(mean_pred, 0.85, 1.15)

    return mean_pred
```

---

## 📝 Implementation in Submission

**Modified submission template:**

```python
# submission.py with inference strategies

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# ... EEGNeX model definition ...

class Submission:
    def __init__(self, SFREQ, DEVICE):
        self.sfreq = SFREQ
        self.device = DEVICE
        self.model_path = load_model_path()

        # Training statistics (IMPORTANT: Compute these from your training data!)
        self.c1_stats = {'mean': 0.95, 'std': 0.12}
        self.c2_stats = {'mean': 0.0, 'std': 0.8}

        # Load models
        self.model_c1 = self._load_model_c1()
        self.model_c2 = self._load_model_c2()

    def _predict_with_tta(self, model, eeg_data, n_aug=5):
        """Test-time augmentation"""
        predictions = []

        for i in range(n_aug):
            if i == 0:
                aug_data = eeg_data
            elif i == 1:
                aug_data = np.roll(eeg_data, 5, axis=-1)
            elif i == 2:
                aug_data = np.roll(eeg_data, -5, axis=-1)
            elif i == 3:
                aug_data = eeg_data + np.random.randn(*eeg_data.shape) * 0.001
            else:
                aug_data = np.roll(eeg_data, -10, axis=-1)

            tensor = torch.tensor(aug_data, dtype=torch.float32).unsqueeze(0).to(self.device)

            with torch.no_grad():
                pred = model(tensor)
                predictions.append(pred.item())

        return np.mean(predictions), np.std(predictions)

    def get_model_challenge_1(self):
        """Return C1 model with inference wrapper"""
        return lambda x: self._inference_c1(x)

    def get_model_challenge_2(self):
        """Return C2 model with inference wrapper"""
        return lambda x: self._inference_c2(x)

    def _inference_c1(self, eeg_data):
        """Challenge 1 inference with strategies"""
        # TTA + smoothing
        mean_pred, uncertainty = self._predict_with_tta(
            self.model_c1, eeg_data, n_aug=5
        )

        # Confidence-based shrinkage
        if uncertainty > 0.05:
            mean_pred = 0.8 * mean_pred + 0.2 * self.c1_stats['mean']

        # Clip to safe range
        mean_pred = np.clip(mean_pred, 0.85, 1.15)

        return torch.tensor([[mean_pred]])

    def _inference_c2(self, eeg_data):
        """Challenge 2 inference with strategies"""
        mean_pred, uncertainty = self._predict_with_tta(
            self.model_c2, eeg_data, n_aug=5
        )

        # Confidence-based shrinkage
        if uncertainty > 0.1:
            mean_pred = 0.7 * mean_pred + 0.3 * self.c2_stats['mean']

        # Clip to reasonable range
        mean_pred = np.clip(mean_pred, -2.5, 2.5)

        return torch.tensor([[mean_pred]])
```

**This submission includes:**
- ✅ Test-time augmentation (TTA)
- ✅ Confidence-based weighting
- ✅ Prediction clipping
- ✅ Statistical smoothing

**Expected improvement over base model:** 0.03-0.08

---

## 🎯 Summary: Top 3 Inference Strategies

### #1: Test-Time Augmentation (TTA)
- **Improvement:** 0.01-0.03
- **Effort:** Low (10 min to implement)
- **Use:** Always! Easy win

### #2: Prediction Clipping
- **Improvement:** 0.005-0.02
- **Effort:** Very Low (2 min)
- **Use:** Always! Free improvement

### #3: Confidence-Based Weighting
- **Improvement:** 0.01-0.02
- **Effort:** Medium (20 min)
- **Use:** For final submissions

**Combined:** Expected 0.03-0.08 total improvement!

**Example:**
- Base model score: 0.95
- + TTA: 0.93
- + Clipping: 0.92
- + Confidence: 0.90
- **Final: 0.90** (0.05 improvement!)

Next: See ENSEMBLE_STRATEGY.md for multi-model approaches.
