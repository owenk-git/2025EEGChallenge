# ULTRATHINK: Robust Evaluation & Kaggle-Style Winning Strategies

## 🔴 Critical Issues with Current Approach

### Current Setup (Basic):
- ✅ Train/val split (80/20) - DONE
- ✅ NRMSE metric - DONE
- ❌ **Random split** - can have same subject in train/val!
- ❌ **No test set** - using all non-validation data
- ❌ **Single metric** - NRMSE only
- ❌ **No cross-validation** - single split might be lucky/unlucky
- ❌ **No ensembling** - single model approach

### What Competition Winners Do:
- ✅ **Subject-wise splitting** - prevent data leakage
- ✅ **Multiple evaluation metrics** - robust performance assessment
- ✅ **K-fold cross-validation** - reduce variance in estimates
- ✅ **Independent test set** - true generalization measure
- ✅ **Ensemble methods** - combine multiple models
- ✅ **Boosting techniques** - LightGBM, XGBoost
- ✅ **Test-Time Augmentation** - improve predictions

---

## 1. EEG-Specific Evaluation Metrics

### 🎯 Competition Metric (Primary)
**Normalized RMSE (NRMSE)**
```python
nrmse = sqrt(mean((y_pred - y_true)²)) / std(y_true)
```
- ✅ Already implemented
- Used for model selection
- Final score: 0.3 × C1_NRMSE + 0.7 × C2_NRMSE

### 📊 Additional Metrics (For Validation)

#### 1. Pearson Correlation Coefficient
**Why:** Common in neuroscience, measures linear relationship
```python
from scipy.stats import pearsonr

r, p_value = pearsonr(predictions, targets)
# r ∈ [-1, 1], higher is better
# p_value < 0.05 means significant
```

#### 2. R² Score (Coefficient of Determination)
**Why:** Explains variance, interpretable (% variance explained)
```python
from sklearn.metrics import r2_score

r2 = r2_score(targets, predictions)
# r2 ∈ (-∞, 1], higher is better
# r2 = 1.0 means perfect prediction
# r2 = 0.0 means predictions = mean(targets)
```

#### 3. Concordance Correlation Coefficient (CCC)
**Why:** Gold standard for agreement between methods
```python
def concordance_correlation_coefficient(y_true, y_pred):
    """Lin's Concordance Correlation Coefficient"""
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))

    ccc = (2 * covariance) / (var_true + var_pred + (mean_true - mean_pred)**2)
    return ccc
# ccc ∈ [-1, 1], higher is better
```

#### 4. Mean Absolute Percentage Error (MAPE)
**Why:** Interpretable (% error), robust to outliers
```python
mape = np.mean(np.abs((targets - predictions) / targets)) * 100
# Lower is better, units = %
```

### 📈 Recommended Metric Suite
```python
from utils.metrics import compute_comprehensive_metrics

metrics = compute_comprehensive_metrics(predictions, targets)
# Returns:
# {
#   'nrmse': 1.23,      # Competition metric
#   'rmse': 0.045,
#   'mae': 0.032,
#   'r2': 0.65,         # % variance explained
#   'pearson_r': 0.82,  # Correlation
#   'ccc': 0.78,        # Agreement
#   'mape': 3.2         # % error
# }
```

**Model Selection Priority:**
1. **NRMSE** (competition metric) - primary
2. **Pearson r** (correlation) - secondary
3. **R²** (variance explained) - tertiary

---

## 2. Robust Cross-Validation Strategy

### 🔴 Problem: Data Leakage in EEG

**Critical Issue:** Each subject has multiple recordings!
- Subject A: 5 recordings
- If 4 recordings in train, 1 in val → **DATA LEAKAGE**
- Model learns subject-specific patterns
- Overestimates performance

**Solution:** **Subject-wise (Group) K-Fold**

### ✅ Subject-Wise K-Fold Cross-Validation

```python
from sklearn.model_selection import GroupKFold

# Get subject IDs for each recording
subject_ids = dataset.get_subject_ids()  # e.g., ['sub-001', 'sub-001', 'sub-002', ...]

# 5-fold cross-validation grouped by subject
gkf = GroupKFold(n_splits=5)

fold_results = []
for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=subject_ids)):
    # Train on train_idx subjects
    # Validate on val_idx subjects (completely unseen subjects!)

    # Train model...
    metrics = evaluate(model, val_data)
    fold_results.append(metrics)

# Average across folds
mean_nrmse = np.mean([r['nrmse'] for r in fold_results])
std_nrmse = np.std([r['nrmse'] for r in fold_results])

print(f"Cross-Val NRMSE: {mean_nrmse:.4f} ± {std_nrmse:.4f}")
```

**Benefits:**
- ✅ No data leakage (subjects never in both train/val)
- ✅ Robust performance estimate (average across 5 folds)
- ✅ Confidence interval (std across folds)
- ✅ Reduces variance from lucky/unlucky single split

### 📊 Recommended Split Strategy

**Option A: 5-Fold Cross-Validation (Aggressive)**
```
Fold 1: Train on 80%, Val on 20%
Fold 2: Train on 80%, Val on 20%
Fold 3: Train on 80%, Val on 20%
Fold 4: Train on 80%, Val on 20%
Fold 5: Train on 80%, Val on 20%
→ Average 5 models or select best fold
```

**Option B: Train/Val/Test Split (Conservative)**
```
Train:      60% of subjects (1800 subjects)
Validation: 20% of subjects (600 subjects)  - for hyperparameter tuning
Test:       20% of subjects (600 subjects)  - for final evaluation (NEVER TOUCH!)

→ Tune on validation, report on test
```

**Option C: Nested Cross-Validation (Gold Standard, Expensive)**
```
Outer Loop: 5-fold for test performance estimation
  Inner Loop: 4-fold for hyperparameter tuning

→ Most robust, but 5 × 4 = 20 training runs
```

**Recommendation for Competition:**
- Use **Option B** (Train/Val/Test)
- Reason: Fast iteration, clear separation
- Test set provides unbiased performance estimate

---

## 3. Independent Test Set Design

### 📁 Dataset Split Architecture

```python
from sklearn.model_selection import train_test_split

# Get unique subjects
unique_subjects = dataset.description['subject'].unique()
n_subjects = len(unique_subjects)

# Split subjects (not recordings!)
train_subjects, temp_subjects = train_test_split(
    unique_subjects, test_size=0.4, random_state=42
)
val_subjects, test_subjects = train_test_split(
    temp_subjects, test_size=0.5, random_state=42
)

# Result:
# Train: 60% subjects (1800 subjects)
# Val:   20% subjects (600 subjects)
# Test:  20% subjects (600 subjects)

print(f"Train subjects: {len(train_subjects)}")
print(f"Val subjects: {len(val_subjects)}")
print(f"Test subjects: {len(test_subjects)}")

# Filter dataset by subjects
train_data = dataset[dataset['subject'].isin(train_subjects)]
val_data = dataset[dataset['subject'].isin(val_subjects)]
test_data = dataset[dataset['subject'].isin(test_subjects)]
```

### 🎯 Usage Strategy

**During Development:**
1. Train on `train_data`
2. Tune hyperparameters on `val_data`
3. **NEVER look at `test_data`**

**After Hyperparameter Tuning:**
1. Final model trained on `train_data`
2. Select best checkpoint using `val_data`
3. **Evaluate once on `test_data`** → true performance

**For Submission:**
1. Retrain on `train_data + val_data` (80% total)
2. Monitor with `test_data` (20%)
3. OR: Train on 100% data (no validation) if confident

### 🔒 Test Set Rules

**NEVER:**
- ❌ Train on test set
- ❌ Tune hyperparameters using test set
- ❌ Select models based on test set
- ❌ Look at test predictions during development

**ONLY:**
- ✅ Evaluate once after all decisions made
- ✅ Use for final performance report
- ✅ Compare to validation to check overfitting

---

## 4. Kaggle-Winning Techniques

### 🏆 Common Competition Strategies

#### A. Ensemble Methods

**1. Model Averaging (Simple)**
```python
# Train 5 models with different seeds
models = []
for seed in [42, 123, 456, 789, 1011]:
    model = train_model(seed=seed)
    models.append(model)

# Average predictions
predictions = np.mean([m.predict(X) for m in models], axis=0)
```

**2. Weighted Ensemble**
```python
# Weight models by validation performance
weights = [0.3, 0.25, 0.2, 0.15, 0.1]  # Best model gets highest weight
predictions = np.average(
    [m.predict(X) for m in models],
    axis=0,
    weights=weights
)
```

**3. Stacking (Meta-Model)**
```python
# Level 1: Base models
model1 = EEGNeX(dropout=0.2)
model2 = EEGNet(dropout=0.3)
model3 = DeepConvNet(dropout=0.25)

# Get predictions from base models
pred1 = model1.predict(X_val)
pred2 = model2.predict(X_val)
pred3 = model3.predict(X_val)

# Level 2: Meta-model (train on base predictions)
meta_features = np.column_stack([pred1, pred2, pred3])
meta_model = Ridge()  # Simple linear model
meta_model.fit(meta_features, y_val)

# Final prediction
final_pred = meta_model.predict(meta_features)
```

#### B. Gradient Boosting on Features

**Extract features from CNN, use GBM:**
```python
import lightgbm as lgb

# 1. Train CNN to extract features
cnn_model = EEGNeX(...)
cnn_model.train()

# 2. Extract features (before final layer)
def extract_features(model, data):
    model.eval()
    features = []
    with torch.no_grad():
        for batch in data:
            # Get features from second-to-last layer
            feat = model.feature_extractor(batch)
            features.append(feat.cpu().numpy())
    return np.concatenate(features)

train_features = extract_features(cnn_model, train_loader)
val_features = extract_features(cnn_model, val_loader)

# 3. Train LightGBM on features
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}

lgb_train = lgb.Dataset(train_features, train_targets)
lgb_val = lgb.Dataset(val_features, val_targets, reference=lgb_train)

gbm = lgb.train(
    lgb_params,
    lgb_train,
    num_boost_round=1000,
    valid_sets=[lgb_val],
    early_stopping_rounds=50
)

# 4. Predict
predictions = gbm.predict(val_features)
```

**Benefits:**
- ✅ LightGBM captures non-linear relationships
- ✅ Fast training
- ✅ Often beats end-to-end neural networks

#### C. Pseudo-Labeling (Semi-Supervised)

**Use test data (carefully!):**
```python
# 1. Train model on labeled train data
model.train_on(train_data)

# 2. Predict on test data
test_predictions = model.predict(test_data)

# 3. Select high-confidence predictions
confidence_threshold = 0.9  # Top 10%
high_conf_idx = get_high_confidence(test_predictions, threshold=confidence_threshold)

# 4. Add to training set with pseudo-labels
pseudo_train = test_data[high_conf_idx]
pseudo_labels = test_predictions[high_conf_idx]

# 5. Retrain on train + pseudo-labeled data
model.train_on(train_data + (pseudo_train, pseudo_labels))
```

**⚠️ Caution:** Can amplify errors if initial predictions are poor

#### D. Test-Time Augmentation (TTA)

**Already documented, but critical:**
```python
def predict_with_tta(model, data, n_augmentations=10):
    predictions = []

    for _ in range(n_augmentations):
        # Apply random augmentations
        augmented = augment(data)
        pred = model.predict(augmented)
        predictions.append(pred)

    # Average predictions
    return np.mean(predictions, axis=0)
```

**Typical improvement:** 0.01-0.03 reduction in NRMSE

---

## 5. Recommended Implementation Plan

### Phase 1: Robust Validation (Week 1)

**Goal:** Establish reliable performance estimates

```python
# 1. Implement subject-wise split
def create_subject_split(dataset, train_size=0.6, val_size=0.2, test_size=0.2):
    """Split by subjects, not recordings"""
    pass

# 2. Add comprehensive metrics
from utils.metrics import compute_comprehensive_metrics

# 3. Implement independent test set
train_loader, val_loader, test_loader = create_split_loaders(...)

# 4. Add CCC, Pearson, R² to logging
```

**Expected:** More accurate performance estimates, detect overfitting

### Phase 2: Model Improvements (Week 2)

**Goal:** Improve single model performance

```python
# 1. Hyperparameter search
learning_rates = [1e-4, 5e-4, 1e-3, 5e-3]
dropouts = [0.1, 0.2, 0.3, 0.4]
architectures = ['EEGNeX', 'EEGNet', 'DeepConvNet']

# 2. Data augmentation
- Time shifts
- Channel dropout
- Gaussian noise
- Mixup

# 3. Regularization
- Early stopping
- Weight decay
- Dropout
- Batch normalization
```

**Expected:** 0.05-0.10 improvement in NRMSE

### Phase 3: Ensemble & Boosting (Week 3)

**Goal:** Combine multiple models for best performance

```python
# 1. Train multiple models
models = []
for seed in seeds:
    model = train_with_seed(seed)
    models.append(model)

# 2. Weighted ensemble based on val performance
weights = compute_weights_from_val(models)
ensemble_pred = weighted_average(models, weights)

# 3. LightGBM on CNN features
features = extract_cnn_features(model, data)
gbm = train_lightgbm(features, targets)

# 4. Stack ensemble
meta_model = train_meta_model([cnn_pred, gbm_pred], targets)
```

**Expected:** 0.03-0.08 improvement in NRMSE

### Phase 4: Final Optimization (Week 4)

**Goal:** Squeeze every last bit of performance

```python
# 1. Test-Time Augmentation
predictions = tta_predict(model, test_data, n=10)

# 2. Pseudo-labeling
high_conf_data = select_high_confidence(test_pred)
retrain_with_pseudo_labels(model, high_conf_data)

# 3. Hyperparameter fine-tuning
final_lr = 1e-5  # Fine-tune with very small LR
model.fine_tune(final_lr, epochs=10)
```

**Expected:** 0.01-0.03 improvement in NRMSE

---

## 6. Expected Performance Trajectory

### Current Status
- **Best:** 1.14 (C1: 1.45, C2: 1.01)
- **SOTA:** 0.978
- **Gap:** 0.162

### With Robust Validation (Phase 1)
- **Expected:** 1.10-1.15 (more accurate estimate)
- **Reason:** Better model selection, no overfitting

### With Model Improvements (Phase 2)
- **Expected:** 1.00-1.05
- **Reason:** Better hyperparameters, data augmentation

### With Ensemble (Phase 3)
- **Expected:** 0.95-1.00
- **Reason:** Multiple models, LightGBM boost

### With Final Optimization (Phase 4)
- **Expected:** 0.90-0.95
- **Reason:** TTA, pseudo-labeling, fine-tuning

### Target
- **Goal:** < 0.978 (beat SOTA)
- **Realistic:** 0.95-1.00 with 4-week effort
- **Stretch:** 0.90-0.95 with all techniques

---

## 7. Immediate Action Items

### Priority 1: Fix Data Leakage (CRITICAL)
```bash
# Implement subject-wise splitting
python -c "from data.official_dataset_example import create_subject_wise_split; help(create_subject_wise_split)"
```

### Priority 2: Add Comprehensive Metrics
```bash
# Update utils/metrics.py with Pearson, R², CCC
git pull
python utils/metrics.py  # Test new metrics
```

### Priority 3: Create Test Set
```bash
# Update train.py to support train/val/test split
python train.py -c 1 -d dummy -o --max 100 -e 100 --split_mode train_val_test --num 1
```

### Priority 4: Start Ensemble Experiments
```bash
# Train 5 models with different seeds
for seed in 42 123 456 789 1011; do
    python train.py -c 1 -d dummy -o --max 100 -e 100 --seed $seed --num $seed
done

# Average predictions
python ensemble_models.py --models exp_42 exp_123 exp_456 exp_789 exp_1011
```

---

## 8. Implementation Checklist

### Validation & Metrics
- [ ] Implement subject-wise GroupKFold splitting
- [ ] Add Pearson correlation to metrics
- [ ] Add R² score to metrics
- [ ] Add CCC to metrics
- [ ] Create train/val/test split (60/20/20)
- [ ] Update experiment logging with all metrics

### Model Training
- [ ] Add random seed argument to train.py
- [ ] Implement data augmentation (time shift, noise, channel dropout)
- [ ] Hyperparameter search script
- [ ] Early stopping based on validation metrics

### Ensemble & Boosting
- [ ] Train multiple models with different seeds
- [ ] Implement weighted ensemble
- [ ] Extract CNN features for LightGBM
- [ ] Train LightGBM on features
- [ ] Implement stacking meta-model
- [ ] Test-Time Augmentation

### Advanced Techniques
- [ ] Pseudo-labeling pipeline
- [ ] Cross-validation script
- [ ] Model selection based on multiple metrics
- [ ] Confidence interval estimation

---

## 9. Resources & References

### Papers
- "Deep Learning with EEG Spectrograms" (EEGNet)
- "EEGNeX: Novel Deep Learning Framework" (EEGNeX)
- "Ensemble Methods in Machine Learning" (Dietterich, 2000)
- "A Survey on Transfer Learning" (Pan & Yang, 2010)

### Kaggle Competitions (Similar)
- Grasp-and-Lift EEG Detection
- Melbourne University AES/MathWorks/NIH Seizure Prediction
- BCI Challenge @ NER 2015

### Tools
- LightGBM: https://lightgbm.readthedocs.io/
- Optuna (hyperparameter optimization): https://optuna.org/
- MNE-Python (EEG analysis): https://mne.tools/

---

## Summary

**Key Insights:**
1. **Subject-wise splitting is CRITICAL** - prevents data leakage
2. **Multiple metrics give robust view** - NRMSE + Pearson + R²
3. **Ensemble beats single models** - 0.03-0.08 improvement
4. **LightGBM on features often wins** - Kaggle secret sauce
5. **Test-Time Augmentation is free performance** - 0.01-0.03 improvement

**Recommended Path:**
1. Fix data leakage (subject-wise split)
2. Add comprehensive metrics
3. Create independent test set
4. Train 5+ models with different configs
5. Ensemble with LightGBM
6. Apply TTA at inference

**Expected Result:**
- Start: 1.14
- After Phase 1-2: 1.00-1.05
- After Phase 3-4: 0.90-0.95
- **Target: Beat 0.978 SOTA**
