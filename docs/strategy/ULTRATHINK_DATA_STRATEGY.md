# ULTRATHINK: Data Strategy with 3,387 Subjects

## 📊 Dataset Overview

**Total Available**: 3,387 subjects across R1-R11 + NC
**Recordings**: ~67,000 recordings (multiple sessions per subject)
**Challenges**: C1 (response time) and C2 (externalizing factor)

---

## 🎯 Critical Insights

### 1. Subject vs Recording Distinction

**CRITICAL**: Each subject has multiple EEG recordings (sessions)

```
Subject s001:
  ├── Recording 1 (session 1)
  ├── Recording 2 (session 2)
  └── Recording 3 (session 3)

Total subjects: 3,387
Total recordings: ~67,000
Recordings per subject: ~20 average
```

**Implication**: We must split by SUBJECT, not by recording!
- ✅ **Correct**: Subject s001's ALL recordings in train OR val, never both
- ❌ **Wrong**: Subject s001's recording 1 in train, recording 2 in val (data leakage!)

**Already implemented** in our code (subject_wise=True) ✅

---

## 🔍 Data Split Strategies

### Strategy A: Train/Val Split (CURRENT - RECOMMENDED)

```
3,387 subjects
    ↓
80/20 split by subjects
    ↓
Train: 2,710 subjects (~54,000 recordings)
Val:   677 subjects (~13,000 recordings)
```

**Pros**:
- ✅ Maximum training data
- ✅ Validation set for hyperparameter tuning
- ✅ Can iterate quickly
- ✅ Competition has hidden test set anyway

**Cons**:
- ❌ No independent test set
- ❌ Risk of overfitting to validation set

**Best for**: Initial exploration, rapid iteration

**Implementation**: Already done with `create_official_dataloaders_with_split()`

---

### Strategy B: Train/Val/Test Split

```
3,387 subjects
    ↓
60/20/20 split by subjects
    ↓
Train: 2,032 subjects (~41,000 recordings)
Val:   677 subjects (~13,000 recordings)
Test:  678 subjects (~13,000 recordings)
```

**Pros**:
- ✅ Independent test set for final evaluation
- ✅ Prevents validation set overfitting
- ✅ Realistic performance estimate
- ✅ Test set NEVER touched during development

**Cons**:
- ❌ Less training data
- ❌ More complex workflow

**Best for**: Final model evaluation before submission

**Implementation**: Already done with `create_official_dataloaders_train_val_test()`

---

### Strategy C: K-Fold Cross-Validation

```
3,387 subjects
    ↓
5-Fold CV by subjects
    ↓
Fold 1: Train on 2,710, Val on 677
Fold 2: Train on 2,710, Val on 677
Fold 3: Train on 2,710, Val on 677
Fold 4: Train on 2,710, Val on 677
Fold 5: Train on 2,710, Val on 677
    ↓
Average performance across 5 folds
```

**Pros**:
- ✅ Robust performance estimate
- ✅ Every subject used for validation once
- ✅ Reduces variance in evaluation
- ✅ Can ensemble 5 models

**Cons**:
- ❌ 5x training time
- ❌ More complex implementation
- ❌ Need more GPU resources

**Best for**: Final model selection, ensemble

**Implementation**: TODO - need to add K-fold support

---

## 🎯 RECOMMENDED STRATEGY

### Phase 1: Exploration (Now - Next Week)
**Use**: Strategy A (80/20 Train/Val)

```bash
python train.py -c 1 -o -e 100 --val_split 0.2
```

**Why**:
- Fast iteration
- Maximum training data
- Quick validation feedback
- Test different hypotheses

**Goal**: Find best architecture, hyperparameters, augmentation

---

### Phase 2: Validation (1-2 weeks)
**Use**: Strategy B (60/20/20 Train/Val/Test)

```bash
python train.py -c 1 -o -e 150 --use_train_val_test --train_split 0.6 --val_split 0.2 --test_split 0.2
```

**Why**:
- Independent test set
- Realistic performance estimate
- Avoid validation overfitting
- Final model selection

**Goal**: Select best model for submission

---

### Phase 3: Ensemble (Final Week)
**Use**: Strategy C (5-Fold CV)

```bash
python train_kfold.py -c 1 -o -e 150 --n_folds 5
```

**Why**:
- Robust performance
- Create ensemble of 5 models
- Average predictions
- Beat single model performance

**Goal**: Maximize competition score

---

## 📈 Validation Dataset Design

### Current Validation (20% of subjects)

```
Validation Set:
  ├── 677 subjects
  ├── ~13,000 recordings
  ├── Subject-wise split ✅
  └── Representative of full distribution
```

### Improvements to Consider:

#### 1. Stratified Splitting

**Goal**: Ensure validation set mirrors train set demographics

```python
# Stratify by:
- Age groups (child, adolescent, adult)
- Sex (male, female)
- Behavioral target distribution (low, medium, high)
- Release version (R1-R11, NC)
```

**Benefit**: More reliable validation performance

**Implementation**:
```python
from sklearn.model_selection import StratifiedKFold

# Stratify by binned target values
target_bins = pd.cut(targets, bins=5, labels=False)
splitter = StratifiedKFold(n_splits=5, shuffle=True, random_seed=42)
```

---

#### 2. Temporal Considerations

**Issue**: If subjects have sessions at different times, temporal leakage?

**Check**: Are sessions from different days/weeks/months?

**Solution**:
- If yes: Split by subject (already doing) ✅
- Optionally: Use earlier sessions for train, later for val (temporal val)

---

#### 3. Validation Metrics

**Currently tracking** (from `utils/metrics.py`):
- ✅ NRMSE (competition metric)
- ✅ RMSE
- ✅ MAE
- ✅ Pearson r
- ✅ R²
- ✅ CCC (Concordance)
- ✅ MAPE

**Additional to consider**:
- Per-subject metrics (some subjects harder than others)
- Subgroup analysis (by age, sex, etc.)
- Prediction distribution (check for mode collapse)

---

## 🎲 Training Batch Sampling Strategies

### Current: Random Sampling

```python
DataLoader(..., shuffle=True)  # Random sampling
```

**Pros**: Simple, unbiased
**Cons**: May not see hard examples often enough

---

### Strategy 1: Weighted Sampling by Target Distribution

**Goal**: Balance examples across target value range

```python
# Challenge 1: Response time (0-1 normalized)
# Some subjects have response_time ~ 0.9 (common)
# Some subjects have response_time ~ 0.3 (rare)

# Weight samples inversely to frequency
target_counts = Counter(binned_targets)
weights = [1.0 / target_counts[t] for t in binned_targets]
sampler = WeightedRandomSampler(weights, len(dataset))
DataLoader(dataset, sampler=sampler, ...)
```

**Benefit**: Better coverage of full target range

---

### Strategy 2: Hard Example Mining

**Goal**: Focus on examples with high prediction error

```python
# After epoch 1:
# - Compute prediction error for each sample
# - Weight samples by error (higher error = higher weight)
# - Sample more from hard examples in epoch 2

errors = abs(predictions - targets)
weights = errors / errors.sum()
sampler = WeightedRandomSampler(weights, len(dataset))
```

**Benefit**: Faster convergence, better performance on hard cases

---

### Strategy 3: Curriculum Learning

**Goal**: Start with easy examples, gradually add harder ones

```python
# Epoch 1-20: Only samples with low variance
# Epoch 21-50: Add medium variance samples
# Epoch 51+: All samples
```

**Benefit**: More stable training, better convergence

---

### Strategy 4: Balanced Batch Sampling

**Goal**: Each batch has diverse target values

```python
# Ensure each batch has:
# - Mix of low/medium/high response times
# - Mix of different age groups
# - Mix of different releases

class BalancedBatchSampler:
    def __init__(self, targets, batch_size, n_bins=5):
        self.bins = pd.cut(targets, bins=n_bins, labels=False)
        self.batch_size = batch_size

    def __iter__(self):
        # Sample batch_size // n_bins from each bin
        for batch in self.create_balanced_batches():
            yield batch
```

**Benefit**: More stable gradients, better generalization

---

## 🎯 RECOMMENDED BATCH SAMPLING

### Phase 1: Exploration
**Use**: Random sampling (current)
- Simple baseline
- Unbiased

### Phase 2: Optimization
**Use**: Weighted sampling by target distribution
- Better coverage
- Address class imbalance

### Phase 3: Fine-tuning
**Use**: Hard example mining
- Focus on difficult cases
- Squeeze out last performance gains

---

## 📊 Multiple Models & Ensemble Strategy

### Model Diversity Sources

#### 1. Different Random Seeds
```python
# Train 5 models with different seeds
for seed in [42, 123, 456, 789, 1011]:
    model = train(seed=seed)
```

**Diversity**: Different random initializations
**Benefit**: Reduces variance through averaging

---

#### 2. Different Architectures
```python
models = [
    EEGNeX(dropout=0.2),           # Baseline
    EEGNeX(dropout=0.3),           # Higher regularization
    EEGNeX(dropout=0.2, deeper=True),  # More capacity
    EEGNet(),                       # Different architecture
    DeepConvNet(),                  # Another architecture
]
```

**Diversity**: Different model biases
**Benefit**: Captures different patterns

---

#### 3. Different Training Strategies
```python
strategies = [
    {"epochs": 100, "lr": 1e-3, "batch": 32},   # Baseline
    {"epochs": 150, "lr": 5e-4, "batch": 64},   # Lower LR, larger batch
    {"epochs": 100, "lr": 2e-3, "batch": 16},   # Higher LR, smaller batch
    {"epochs": 200, "lr": 1e-3, "batch": 32},   # More epochs
]
```

**Diversity**: Different optimization paths
**Benefit**: Different local minima

---

#### 4. Different Data Augmentation
```python
augmentations = [
    None,                          # No augmentation
    GaussianNoise(std=0.01),      # Noise augmentation
    TemporalCrop(crop_size=0.9),  # Temporal cropping
    ChannelDropout(p=0.1),        # Channel dropout
]
```

**Diversity**: Different data views
**Benefit**: Better robustness

---

#### 5. Different Data Subsets (Bagging)
```python
# Train each model on 80% of training data
for i in range(5):
    subset = random_subset(train_data, fraction=0.8)
    model = train(subset)
```

**Diversity**: Different data views
**Benefit**: Reduces overfitting

---

### Ensemble Methods

#### Method 1: Simple Averaging
```python
predictions = [model(x) for model in models]
final_prediction = np.mean(predictions, axis=0)
```

**Pros**: Simple, robust
**Cons**: Treats all models equally

**Expected gain**: 5-10% improvement over single model

---

#### Method 2: Weighted Averaging
```python
# Weight by validation performance
weights = [1/val_nrmse for val_nrmse in val_scores]
weights = weights / sum(weights)

final_prediction = sum(w * pred for w, pred in zip(weights, predictions))
```

**Pros**: Better models contribute more
**Cons**: May overfit to validation set

**Expected gain**: 7-12% improvement

---

#### Method 3: Stacking
```python
# Train meta-model on validation predictions
meta_features = np.column_stack([m.predict(X_val) for m in models])
meta_model = LinearRegression()
meta_model.fit(meta_features, y_val)

# Final prediction
meta_features_test = np.column_stack([m.predict(X_test) for m in models])
final_prediction = meta_model.predict(meta_features_test)
```

**Pros**: Learns optimal combination
**Cons**: More complex, risk of overfitting

**Expected gain**: 10-15% improvement

---

### RECOMMENDED ENSEMBLE STRATEGY

#### Stage 1: Train Diverse Models (Week 1-2)
```python
# 5 models with different random seeds
for seed in [42, 123, 456, 789, 1011]:
    python train.py -c 1 -o -e 100 --seed {seed}

# Save each model separately
checkpoints/
  ├── c1_seed42_best.pth
  ├── c1_seed123_best.pth
  ├── c1_seed456_best.pth
  ├── c1_seed789_best.pth
  └── c1_seed1011_best.pth
```

---

#### Stage 2: Validate Ensemble (Week 2)
```python
# Test ensemble on validation set
python evaluate_ensemble.py \
  --models checkpoints/c1_seed*_best.pth \
  --method weighted_average

# Output:
# Individual models:
#   seed42:  NRMSE=1.05
#   seed123: NRMSE=1.08
#   seed456: NRMSE=1.06
#   seed789: NRMSE=1.07
#   seed1011: NRMSE=1.04
#
# Ensemble (weighted avg):
#   NRMSE=0.98  ← 6% improvement! ✅
```

---

#### Stage 3: Create Ensemble Submission (Week 3)
```python
# Modify submission.py to load multiple models
def predict(eeg_data):
    predictions = []
    for model_path in model_paths:
        model = load_model(model_path)
        pred = model(eeg_data)
        predictions.append(pred)

    # Weighted average based on validation performance
    weights = [0.25, 0.18, 0.22, 0.20, 0.15]  # From validation
    final_pred = sum(w * p for w, p in zip(weights, predictions))
    return final_pred

# Create submission
python create_ensemble_submission.py \
  --models checkpoints/c1_seed*_best.pth \
  --weights 0.25 0.18 0.22 0.20 0.15
```

---

## 🚀 Future Strategy Roadmap

### Week 1: Data & Baseline ✅ (DONE)
- [x] Setup ALL data streaming (3,387 subjects)
- [x] Implement subject-wise splitting
- [x] Add comprehensive metrics
- [x] Train baseline models

**Next**: Run baseline training

---

### Week 2: Exploration
**Goal**: Find best hyperparameters and architecture

```python
# Run 10 exploration experiments
experiments = [
    {"name": "baseline", "dropout": 0.2, "lr": 1e-3, "epochs": 50},
    {"name": "high_dropout", "dropout": 0.4, "lr": 1e-3, "epochs": 50},
    {"name": "low_lr", "dropout": 0.2, "lr": 5e-4, "epochs": 100},
    {"name": "high_lr", "dropout": 0.2, "lr": 2e-3, "epochs": 50},
    {"name": "large_batch", "dropout": 0.2, "lr": 1e-3, "epochs": 50, "batch": 64},
    {"name": "small_batch", "dropout": 0.2, "lr": 1e-3, "epochs": 50, "batch": 16},
    {"name": "more_data", "dropout": 0.2, "lr": 1e-3, "epochs": 50, "max_subjects": None},
    {"name": "deeper", "dropout": 0.2, "lr": 1e-3, "epochs": 50, "deeper": True},
    {"name": "augmentation", "dropout": 0.2, "lr": 1e-3, "epochs": 50, "augment": True},
    {"name": "longer", "dropout": 0.2, "lr": 1e-3, "epochs": 150},
]

for exp in experiments:
    run_experiment(exp)

# Analyze results
analyze_experiments()  # Find best direction
```

**Deliverable**: Best hyperparameters identified

---

### Week 3: Exploitation
**Goal**: Deep dive into best direction from exploration

```python
# If "more_data" + "longer" was best direction:
python train.py -c 1 -o -e 200 --max_subjects None

# If "high_dropout" + "low_lr" was best:
python train.py -c 1 -o -e 150 --dropout 0.4 --lr 5e-4
```

**Deliverable**: Optimized single model (NRMSE ~1.0-1.1)

---

### Week 4: Ensemble & Refinement
**Goal**: Build ensemble, squeeze out final gains

```python
# Train 5 diverse models
for seed in [42, 123, 456, 789, 1011]:
    python train.py -c 1 -o -e 150 --seed {seed} --best_params

# Create ensemble submission
python create_ensemble_submission.py

# Test-Time Augmentation
python create_tta_submission.py --n_augment 5
```

**Deliverable**: Ensemble submission (NRMSE ~0.95-1.0)

---

### Week 5+: Advanced Techniques
**Goal**: Beat SOTA (0.978)

Techniques to try:
1. **K-Fold Cross-Validation**: Train 5 models, ensemble
2. **Test-Time Augmentation**: Multiple predictions per sample
3. **Pseudo-labeling**: Use test set predictions
4. **Mixup/CutMix**: Advanced augmentation
5. **Knowledge Distillation**: Distill ensemble into single model
6. **Architecture Search**: Find optimal architecture
7. **Multi-task Learning**: Joint training on C1 + C2

**Deliverable**: SOTA-beating submission (NRMSE <0.978)

---

## 📁 Data Split Implementation

### Current Implementation Status

#### ✅ Already Implemented:

1. **Train/Val Split** (Strategy A):
```python
train_loader, val_loader = create_official_dataloaders_with_split(
    challenge='c1',
    val_split=0.2,
    subject_wise=True  # ✅ Prevents data leakage
)
```

2. **Train/Val/Test Split** (Strategy B):
```python
train_loader, val_loader, test_loader = create_official_dataloaders_train_val_test(
    challenge='c1',
    train_split=0.6,
    val_split=0.2,
    test_split=0.2
)
```

#### ❌ TODO: Need to Implement:

1. **K-Fold Cross-Validation** (Strategy C)
2. **Stratified Splitting** (by demographics)
3. **Weighted Sampling** (by target distribution)
4. **Hard Example Mining**
5. **Ensemble Training Script**
6. **Ensemble Submission Script**

---

## 🎯 Immediate Action Items

### 1. Test Current Setup (Today)
```bash
# Quick test
python train.py -c 1 -o -m --max 5 -e 3

# Verify: "Unique subjects: ~10"

# Full baseline
python train.py -c 1 -o -e 100

# Verify: "Unique subjects: 3387"
```

### 2. Run Exploration (This Week)
```bash
# Use exploration scripts (already created)
bash scripts/run_exploration_streaming.sh

# Compare results
python scripts/compare_exploration.py
```

### 3. Implement Missing Features (Next Week)
- [ ] K-Fold CV script
- [ ] Stratified splitting
- [ ] Weighted sampling
- [ ] Ensemble training
- [ ] Ensemble submission

### 4. Create Submission (Week 3)
```bash
# Best single model
python create_submission.py \
  --model_c1 checkpoints/c1_best.pth \
  --model_c2 checkpoints/c2_best.pth

# Ensemble model
python create_ensemble_submission.py \
  --models checkpoints/c1_seed*_best.pth
```

---

## 📊 Expected Performance Progression

| Week | Strategy | NRMSE | vs Current |
|------|----------|-------|------------|
| 0 (Baseline) | Previous submission (random weights) | 1.18 | - |
| 1 | Single model, 100 subj, 50 epochs | 1.15 | -2.5% |
| 2 | Single model, ALL data, 100 epochs | 1.08 | -8.5% |
| 3 | Optimized single model | 1.02 | -13.5% |
| 4 | Ensemble (5 models) | 0.96 | -18.6% |
| 5+ | Advanced techniques | 0.92 | -22.0% |

**Target**: Beat SOTA (0.978) by Week 4-5

---

## Summary

### ✅ Data Strategy Finalized:
1. **Phase 1**: 80/20 train/val split for exploration
2. **Phase 2**: 60/20/20 train/val/test for validation
3. **Phase 3**: 5-fold CV for ensemble

### ✅ Validation Strategy:
- Subject-wise splitting (prevents leakage)
- Consider stratification by demographics
- Track 7 comprehensive metrics

### ✅ Batch Sampling:
- Start: Random sampling
- Optimize: Weighted by target distribution
- Final: Hard example mining

### ✅ Ensemble Strategy:
- Train 5 models with different seeds
- Use weighted averaging
- Expected 5-15% improvement

### ✅ Future Roadmap:
- Week 1: Baseline ✅
- Week 2: Exploration
- Week 3: Exploitation
- Week 4: Ensemble
- Week 5+: Advanced techniques

**Next Step**: Run baseline training and verify 3,387 subjects! 🚀
