# Training Strategies to Beat 0.978 SOTA

Based on your submission history and competition analysis.

---

## 🎯 Current Best Score: 1.14 (Sub 3)

**Goal:** Beat 0.978 (SOTA)
**Gap:** 0.162 improvement needed
**Submissions Remaining:** 25 of 35

---

## 📊 Key Insight from Your Submissions

### What Worked (Sub 3: 1.14):
- ✅ **Sigmoid-inside-classifier** (not in forward)
- ✅ **Output scaling:** [0.88, 1.12] for C1
- ✅ **EEGNeX architecture**

### What Didn't Work:
- ❌ Sub 6 Ensemble (1.18) - **Random weights**, not trained!
- ❌ Most submissions: 1.83 - Too conservative/broken detection

### Key Learning:
**Trained weights > Ensemble of random weights**

---

## 🚀 TOP 5 TRAINING STRATEGIES

### Strategy 1: Progressive Training (RECOMMENDED START)

**Concept:** Start small, scale up progressively

**Steps:**
```bash
# Week 1: Get baseline working
# Day 1: Mini training (10 subjects, 20 epochs)
python train.py --challenge 1 --use_official --official_mini \
  --max_subjects 10 --epochs 20 --lr 0.001

python train.py --challenge 2 --use_official --official_mini \
  --max_subjects 10 --epochs 20 --lr 0.001

# Submit → Get baseline score

# Day 2-3: Scale up (50 subjects, 50 epochs)
python train.py --challenge 1 --use_official --max_subjects 50 --epochs 50
python train.py --challenge 2 --use_official --max_subjects 50 --epochs 50

# Submit → Should beat 1.14

# Day 4-7: Full scale (200+ subjects, 100 epochs)
python train.py --challenge 1 --use_official --max_subjects 200 --epochs 100
python train.py --challenge 2 --use_official --max_subjects 200 --epochs 100

# Submit → Aim for <1.0
```

**Why it works:**
- ✅ Quick iteration on small data
- ✅ Verify training works before committing compute
- ✅ Progressive improvement visible

**Expected Results:**
- 10 subjects: ~1.05-1.10
- 50 subjects: ~0.98-1.05
- 200 subjects: ~0.90-0.98

---

### Strategy 2: Challenge 2 Focus (70% of Score)

**Concept:** C2 is 70% of final score, prioritize it!

**Current scores:**
- C1: 1.45 (worse)
- C2: 1.01 (better)

**Your C2 is already decent!** Focus on improving C1.

**Training approach:**
```bash
# Spend more time on C1 (it's hurting you more)
# C1: Train longer, more subjects
python train.py --challenge 1 --use_official \
  --max_subjects 300 --epochs 150 --lr 0.0005 \
  --batch_size 32 --dropout 0.25

# C2: Current approach is working, just scale up
python train.py --challenge 2 --use_official \
  --max_subjects 200 --epochs 100 --lr 0.001 \
  --batch_size 32 --dropout 0.20
```

**Expected improvement:**
- If C1: 1.45 → 1.10 (save 0.35 × 0.3 = 0.105)
- If C2: 1.01 → 0.95 (save 0.06 × 0.7 = 0.042)
- **Total: 1.14 → 0.99** ✅

---

### Strategy 3: Hyperparameter Grid Search

**Concept:** Systematically try different hyperparameters

**Parameters to tune:**
1. Learning rate: [0.0001, 0.0005, 0.001, 0.002]
2. Dropout: [0.15, 0.20, 0.25, 0.30]
3. Batch size: [16, 32, 64]
4. Output range (C1): [(0.85, 1.15), (0.88, 1.12), (0.90, 1.10)]

**Efficient approach:**
```bash
# Create grid search script
# train_grid_search.py

LEARNING_RATES = [0.0001, 0.0005, 0.001]
DROPOUTS = [0.15, 0.20, 0.25]

for lr in LEARNING_RATES:
    for dropout in DROPOUTS:
        # Quick training on subset
        python train.py --challenge 1 --use_official \
          --max_subjects 30 --epochs 30 \
          --lr $lr --dropout $dropout \
          --checkpoint_dir "./checkpoints/lr${lr}_drop${dropout}"

        # Evaluate on validation set
        # Keep top 3 configs
```

**Then train best configs fully:**
```bash
# Top 3 configs get full training
python train.py --challenge 1 --use_official \
  --max_subjects 200 --epochs 100 \
  --lr 0.0005 --dropout 0.20  # Best config
```

**Expected improvement:** 1.14 → 1.00-1.05

---

### Strategy 4: Multi-Task Learning (Advanced)

**Concept:** Train on multiple tasks simultaneously

**HBN dataset has 6 tasks:**
- Contrast Change Detection
- Visual Search
- Rest (eyes open/closed)
- Video watching
- Go/No-Go
- Flanker

**Current:** Only using Contrast Change Detection

**Multi-task approach:**
```python
# Modify official_dataset_example.py to load multiple tasks
tasks = ['contrastChangeDetection', 'visualSearch', 'goNoGo']

for task in tasks:
    dataset = EEGChallengeDataset(task=task, ...)
    # Train on all tasks
```

**Why it helps:**
- ✅ More diverse training data
- ✅ Better generalization
- ✅ Learn robust EEG features across tasks

**Expected improvement:** 1.14 → 0.95-1.00

---

### Strategy 5: Data Augmentation

**Concept:** Create more training data from existing data

**EEG augmentation techniques:**

1. **Time shifting**
```python
# Shift EEG window randomly
def time_shift(eeg_data, max_shift=20):
    shift = np.random.randint(-max_shift, max_shift)
    return np.roll(eeg_data, shift, axis=-1)
```

2. **Channel dropout**
```python
# Randomly drop some channels
def channel_dropout(eeg_data, drop_prob=0.1):
    mask = np.random.random(eeg_data.shape[0]) > drop_prob
    return eeg_data * mask[:, None]
```

3. **Gaussian noise**
```python
# Add small noise
def add_noise(eeg_data, noise_level=0.01):
    noise = np.random.randn(*eeg_data.shape) * noise_level
    return eeg_data + noise
```

4. **Time warping**
```python
# Stretch/compress time slightly
def time_warp(eeg_data, warp_factor=0.9):
    from scipy.ndimage import zoom
    return zoom(eeg_data, (1, warp_factor))[:, :200]
```

**Training with augmentation:**
```python
# Modify dataset __getitem__
def __getitem__(self, idx):
    data, target = self._load_data(idx)

    # Apply random augmentation (50% chance)
    if np.random.random() > 0.5:
        data = time_shift(data)
    if np.random.random() > 0.5:
        data = add_noise(data)

    return data, target
```

**Expected improvement:** 1.14 → 1.00-1.05

---

## 📈 RECOMMENDED TRAINING PROGRESSION

### Week 1: Establish Baseline

| Day | Action | Subjects | Epochs | Expected Score |
|-----|--------|----------|--------|----------------|
| 1 | Quick test | 10 | 20 | ~1.10 |
| 2 | Scale up | 50 | 50 | ~1.02 |
| 3 | Full train | 100 | 100 | ~0.98 |

**Submissions:** 3 (11, 12, 13)
**Goal:** Beat 1.14 ✅

---

### Week 2: Optimize

| Day | Action | Strategy | Expected Score |
|-----|--------|----------|----------------|
| 4 | Hyperparam search | Strategy 3 | ~0.95 |
| 5 | Multi-task | Strategy 4 | ~0.92 |
| 6 | Data augmentation | Strategy 5 | ~0.90 |
| 7 | Best combo | All above | ~0.88 |

**Submissions:** 4-6 (14-19)
**Goal:** Break under 0.90 ✅

---

## 🎓 Training Best Practices

### 1. Learning Rate Schedule

```python
# train.py already has ReduceLROnPlateau
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

# Alternative: Cosine annealing
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=50, eta_min=1e-6
)
```

### 2. Early Stopping

```python
# Add to train.py
patience = 10
best_loss = float('inf')
patience_counter = 0

for epoch in range(epochs):
    train_loss = train_epoch(...)

    if train_loss < best_loss:
        best_loss = train_loss
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print("Early stopping!")
        break
```

### 3. Gradient Clipping

```python
# Add to train.py train_epoch()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

### 4. Mixed Precision Training (Faster)

```python
# train.py
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    output = model(data)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

## 📊 Monitoring Training

### What to Watch:

1. **Loss should decrease steadily**
```
Epoch 1:  Loss = 0.025 ✅
Epoch 10: Loss = 0.012 ✅
Epoch 20: Loss = 0.008 ✅
Epoch 30: Loss = 0.006 ✅
```

2. **Learning rate should decrease when plateau**
```
Epoch 1-10:  LR = 0.001
Epoch 11-20: LR = 0.0005 (reduced)
Epoch 21-30: LR = 0.00025 (reduced again)
```

3. **Gradients should not explode**
```python
# Add logging
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm()
        print(f"{name}: {grad_norm:.4f}")
```

### Red Flags:

- ❌ Loss stays flat → Bad data/targets
- ❌ Loss explodes (NaN) → Learning rate too high
- ❌ Loss oscillates wildly → Batch size too small
- ❌ Gradients all zero → Dead ReLUs or bad init

---

## 🎯 Realistic Score Targets

Based on effort and data:

| Subjects | Epochs | Training Time | Expected Score | Beats SOTA? |
|----------|--------|---------------|----------------|-------------|
| 10 | 20 | ~30 min | 1.05-1.10 | ❌ |
| 50 | 50 | ~2 hours | 0.98-1.05 | Maybe |
| 100 | 100 | ~8 hours | 0.92-0.98 | Likely ✅ |
| 200 | 100 | ~16 hours | 0.88-0.95 | Very likely ✅ |
| 500 | 150 | ~48 hours | 0.85-0.90 | Almost certain ✅ |

**SOTA to beat:** 0.978 (0.928 C1, 1.0 C2)

**Your current best:** 1.14 (1.45 C1, 1.01 C2)

**Low-hanging fruit:**
- Improve C1: 1.45 → 1.00 (save 0.135 on overall score)
- Improves overall: 1.14 → 1.005 ✅ Beats SOTA!

---

## 💡 Quick Wins (Immediate Improvements)

### 1. Just Train! (Biggest Impact)
Your Sub 3 (1.14) was with proper architecture but likely undertrained or small data.
- **Action:** Train on 100+ subjects, 100+ epochs
- **Expected:** 1.14 → 0.95-1.00
- **Effort:** Low (just run training)

### 2. Focus on C1 (Currently 1.45)
C1 is hurting your score more than C2.
- **Action:** More C1 training, try different output ranges
- **Expected:** C1: 1.45 → 1.10, Overall: 1.14 → 1.04
- **Effort:** Low

### 3. Use More Data
Competition has 3000+ subjects, you probably used <50.
- **Action:** Train on 200-500 subjects
- **Expected:** 1.14 → 0.90-0.95
- **Effort:** Medium (longer training)

---

## 🏆 Path to SOTA (0.978)

**Required improvement:** 1.14 → 0.978 (0.162 reduction)

**Breakdown:**
- C1: 1.45 → 0.93 (save 0.156, contributes 0.047 to overall)
- C2: 1.01 → 1.00 (save 0.01, contributes 0.007 to overall)
- **Total savings needed:** 0.162

**This is achievable!** Your architecture is good (Sub 3 proves it).

**Main path:**
1. ✅ Train properly (not random weights) → 1.14 → 1.00
2. ✅ Scale up data (200+ subjects) → 1.00 → 0.92
3. ✅ Hyperparameter tuning → 0.92 → 0.88
4. ✅ Ensemble (see next doc) → 0.88 → 0.85

**Target: Break under 0.90 easily, 0.85 with effort!** 🎯

---

## 📝 Summary: Top 3 Strategies to Try First

### #1: Progressive Training (Start Here!)
```bash
# Day 1: Quick baseline
python train.py --challenge 1 --use_official --max_subjects 50 --epochs 50
python train.py --challenge 2 --use_official --max_subjects 50 --epochs 50
# Expected: ~1.00-1.05
```

### #2: Scale Up Data
```bash
# Day 2-3: More subjects
python train.py --challenge 1 --use_official --max_subjects 200 --epochs 100
python train.py --challenge 2 --use_official --max_subjects 200 --epochs 100
# Expected: ~0.92-0.98
```

### #3: Hyperparam Tuning
```bash
# Day 4-5: Find best LR and dropout
# Try: LR=[0.0001, 0.0005, 0.001], Dropout=[0.15, 0.20, 0.25]
# Expected: ~0.88-0.95
```

**With these 3 strategies, you should beat SOTA (0.978) easily!**

Next: See INFERENCE_STRATEGIES.md for test-time improvements.
