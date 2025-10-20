# 🔬 Exploration Strategy: 5-10 Submissions to Find Best Direction

## 🎯 Goal: Strategic Exploration → Directed Exploitation

**Phase 1 (5-10 submissions):** Test different hypotheses to understand:
- What works best for THIS specific challenge
- Test data distribution characteristics
- Model capacity vs. regularization tradeoffs
- Data amount vs. training time tradeoffs

**Phase 2 (15+ submissions):** Deep dive into best direction found in Phase 1

---

## 📋 Exploration Experiments (--num 1-10)

### Experiment 1: BASELINE (Small & Fast)
**Hypothesis:** Establish baseline with minimal resources

```bash
CUDA_VISIBLE_DEVICES=1 python train.py \
  -c 1 -d dummy -o \
  --max 50 \
  -e 50 \
  --drop 0.2 \
  --lr 1e-3 \
  -b 32 \
  --num 1

CUDA_VISIBLE_DEVICES=1 python train.py \
  -c 2 -d dummy -o \
  --max 50 \
  -e 50 \
  --drop 0.2 \
  --lr 1e-3 \
  -b 32 \
  --num 2
```

**What to learn:**
- Baseline val NRMSE for C1 and C2
- Does model converge in 50 epochs?
- Is 50 subjects enough?

**Expected C1 NRMSE:** 1.3-1.5
**Expected C2 NRMSE:** 1.1-1.3
**Expected Combined:** 1.15-1.35

**Decision criteria:**
- If val NRMSE still decreasing at epoch 50 → Need more epochs
- If val NRMSE plateaus early → Model capacity issue or overfitting
- Compare val metrics to leaderboard score → Understand test distribution

---

### Experiment 3: MORE DATA
**Hypothesis:** More subjects = better generalization

```bash
CUDA_VISIBLE_DEVICES=1 python train.py \
  -c 1 -d dummy -o \
  --max 200 \
  -e 100 \
  --drop 0.2 \
  --lr 1e-3 \
  -b 32 \
  --num 3

CUDA_VISIBLE_DEVICES=1 python train.py \
  -c 2 -d dummy -o \
  --max 200 \
  -e 100 \
  --drop 0.2 \
  --lr 1e-3 \
  -b 32 \
  --num 4
```

**What to learn:**
- Does more data improve performance?
- Is data quantity a bottleneck?
- Training time vs. performance tradeoff

**Expected improvement over Exp 1:** 5-10% (NRMSE: 1.15-1.25)

**Decision criteria:**
- If big improvement (>10%) → Data quantity is key, use ALL data
- If small improvement (<5%) → Data quality or model capacity is issue
- If no improvement → Model capacity saturated or need better architecture

---

### Experiment 5: HIGHER DROPOUT (Regularization Test)
**Hypothesis:** Model overfitting, need more regularization

```bash
CUDA_VISIBLE_DEVICES=1 python train.py \
  -c 1 -d dummy -o \
  --max 100 \
  -e 100 \
  --drop 0.4 \
  --lr 1e-3 \
  -b 32 \
  --num 5

CUDA_VISIBLE_DEVICES=1 python train.py \
  -c 2 -d dummy -o \
  --max 100 \
  -e 100 \
  --drop 0.4 \
  --lr 1e-3 \
  -b 32 \
  --num 6
```

**What to learn:**
- Is overfitting a problem?
- Gap between train and val performance
- Does stronger regularization help generalization?

**Decision criteria:**
- If val improves, train worsens → Overfitting was issue, use higher dropout
- If both worsen → Model capacity too limited, need lower dropout
- Compare train-val gap → Understand overfitting severity

---

### Experiment 7: LOWER LEARNING RATE (Optimization Test)
**Hypothesis:** Model converging too fast, missing optimal solution

```bash
CUDA_VISIBLE_DEVICES=1 python train.py \
  -c 1 -d dummy -o \
  --max 100 \
  -e 150 \
  --drop 0.2 \
  --lr 5e-4 \
  -b 32 \
  --num 7

CUDA_VISIBLE_DEVICES=1 python train.py \
  -c 2 -d dummy -o \
  --max 100 \
  -e 150 \
  --drop 0.2 \
  --lr 5e-4 \
  -b 32 \
  --num 8
```

**What to learn:**
- Is learning rate too high?
- Does slower training find better solution?
- Does model need more iterations?

**Decision criteria:**
- If val improves → LR was too high, use 5e-4 or 1e-4
- If no improvement → LR is fine, focus elsewhere
- Check convergence curve → Early plateau or slow improvement?

---

### Experiment 9: LARGER BATCH SIZE (Training Dynamics Test)
**Hypothesis:** Larger batch = more stable gradients = better convergence

```bash
CUDA_VISIBLE_DEVICES=1 python train.py \
  -c 1 -d dummy -o \
  --max 100 \
  -e 100 \
  --drop 0.2 \
  --lr 2e-3 \
  -b 64 \
  --num 9

CUDA_VISIBLE_DEVICES=1 python train.py \
  -c 2 -d dummy -o \
  --max 100 \
  -e 100 \
  --drop 0.2 \
  --lr 2e-3 \
  -b 64 \
  --num 10
```

**What to learn:**
- Effect of batch size on convergence
- Larger batch needs higher LR (2e-3 vs 1e-3)
- Training speed vs. generalization

**Decision criteria:**
- If val improves → Batch size matters, use 64
- If worsens → Keep batch 32
- Check training stability → Smoother loss curves?

---

## 📊 Exploration Matrix Summary

| Exp # | Challenge | Subjects | Epochs | Dropout | LR | Batch | Hypothesis |
|-------|-----------|----------|--------|---------|----|----|------------|
| 1 | C1 | 50 | 50 | 0.2 | 1e-3 | 32 | Baseline (fast) |
| 2 | C2 | 50 | 50 | 0.2 | 1e-3 | 32 | Baseline (fast) |
| 3 | C1 | 200 | 100 | 0.2 | 1e-3 | 32 | More data helps |
| 4 | C2 | 200 | 100 | 0.2 | 1e-3 | 32 | More data helps |
| 5 | C1 | 100 | 100 | 0.4 | 1e-3 | 32 | Overfitting issue |
| 6 | C2 | 100 | 100 | 0.4 | 1e-3 | 32 | Overfitting issue |
| 7 | C1 | 100 | 150 | 0.2 | 5e-4 | 32 | LR too high |
| 8 | C2 | 100 | 150 | 0.2 | 5e-4 | 32 | LR too high |
| 9 | C1 | 100 | 100 | 0.2 | 2e-3 | 64 | Batch size matters |
| 10 | C2 | 100 | 100 | 0.2 | 2e-3 | 64 | Batch size matters |

---

## 🎯 Decision Tree After Exploration

### After Running Exp 1-10:

```
START: Analyze validation metrics

├─ Best performer: Exp 3-4 (More data)
│  └─ Direction: DATA QUANTITY
│     ├─ Exp 11-15: Train with 300, 400, 500 subjects
│     ├─ Exp 16-20: Use full dataset, longer training
│     └─ Exp 21-25: Ensemble multiple data-trained models
│
├─ Best performer: Exp 5-6 (Higher dropout)
│  └─ Direction: REGULARIZATION
│     ├─ Exp 11-15: Test dropout 0.3, 0.5, varying by layer
│     ├─ Exp 16-20: Add data augmentation (noise, time shift)
│     └─ Exp 21-25: Combine dropout + augmentation + ensemble
│
├─ Best performer: Exp 7-8 (Lower LR)
│  └─ Direction: OPTIMIZATION
│     ├─ Exp 11-15: Test LR 1e-4, cosine annealing, warmup
│     ├─ Exp 16-20: Longer training (200-300 epochs)
│     └─ Exp 21-25: Fine-tuning with very low LR
│
├─ Best performer: Exp 9-10 (Larger batch)
│  └─ Direction: TRAINING DYNAMICS
│     ├─ Exp 11-15: Test batch 96, 128, gradient accumulation
│     ├─ Exp 16-20: Learning rate schedule optimization
│     └─ Exp 21-25: Mixed precision training
│
└─ All similar performance
   └─ Direction: MODEL ARCHITECTURE
      ├─ Exp 11-15: Try different architectures (EEGNet, DeepConvNet)
      ├─ Exp 16-20: Wider/deeper models
      └─ Exp 21-25: Ensemble different architectures
```

---

## 📈 Metrics to Track

### For Each Experiment:

1. **Validation Metrics:**
   - Val NRMSE (primary)
   - Val Pearson r (correlation)
   - Val R² (variance explained)

2. **Leaderboard Metrics:**
   - Submission NRMSE
   - Gap between val and test
   - Ranking change

3. **Training Dynamics:**
   - Convergence speed (epochs to plateau)
   - Train-val gap (overfitting indicator)
   - Best epoch number

4. **Computational Cost:**
   - Training time
   - GPU memory
   - Data loading time

---

## 📊 Expected Patterns to Observe

### Pattern 1: Val vs. Test Performance

**Scenario A: Val ≈ Test**
```
Exp 1: Val=1.30, Test=1.28 → Good generalization
      → Subject-wise split working!
      → Can trust validation metrics
```

**Scenario B: Val > Test (Val worse)**
```
Exp 1: Val=1.30, Test=1.15 → Test easier than val
      → Validation split too hard
      → Can be aggressive with complexity
```

**Scenario C: Val < Test (Val better)**
```
Exp 1: Val=1.20, Test=1.35 → Overfitting validation
      → May have data leakage or lucky split
      → Need stronger regularization
```

### Pattern 2: Data Scaling

**Scenario A: Strong scaling**
```
Exp 1 (50 subj):  NRMSE = 1.35
Exp 3 (200 subj): NRMSE = 1.10  → 18% improvement!
      → Data is bottleneck
      → Use ALL available data
      → Maybe try external datasets
```

**Scenario B: Weak scaling**
```
Exp 1 (50 subj):  NRMSE = 1.30
Exp 3 (200 subj): NRMSE = 1.28  → 1.5% improvement
      → Data not bottleneck
      → Focus on model/training
      → Don't waste time on more data
```

### Pattern 3: Regularization Response

**Scenario A: Overfitting**
```
Exp 1 (drop=0.2): Train=0.80, Val=1.30  → 62% gap!
Exp 5 (drop=0.4): Train=1.00, Val=1.20  → 20% gap
      → Model overfitting
      → Use drop=0.4 or higher
      → Add data augmentation
```

**Scenario B: Underfitting**
```
Exp 1 (drop=0.2): Train=1.25, Val=1.30  → 4% gap
Exp 5 (drop=0.4): Train=1.40, Val=1.42  → Worse!
      → Model capacity limited
      → Use drop=0.1 or lower
      → Try bigger model
```

---

## 🚀 Execution Plan

### Week 1: Run Explorations (Experiments 1-10)

**Day 1-2: Baseline & Data**
```bash
# Run Exp 1-4 (Baseline + More Data)
./scripts/run_exploration_1_4.sh
```

**Day 3-4: Regularization & Optimization**
```bash
# Run Exp 5-8 (Dropout + LR)
./scripts/run_exploration_5_8.sh
```

**Day 5: Batch Size & Analysis**
```bash
# Run Exp 9-10 (Batch size)
./scripts/run_exploration_9_10.sh

# Analyze results
python experiments/analyze_experiments.py
python scripts/compare_exploration.py
```

**Day 6-7: Submit & Observe**
```bash
# Create 5 submissions from best experiments
python create_submission.py --model_c1 checkpoints_exp3/c1_best.pth --model_c2 checkpoints_exp4/c2_best.pth --output exp3_4.zip
# ... submit all 5 best combinations

# Compare leaderboard scores to validation
python scripts/analyze_val_test_gap.py
```

### Week 2: Exploitation (Experiments 11-25)

Based on Week 1 findings, deep dive into best direction.

---

## 📝 Analysis Template

After running each experiment, fill this out:

```markdown
## Experiment X Results

**Config:**
- Challenge: X
- Subjects: X
- Epochs: X
- Dropout: X
- LR: X
- Batch: X

**Validation Metrics:**
- Val NRMSE: X.XXXX
- Val Pearson: X.XXXX
- Val R²: X.XXXX
- Best Epoch: XX

**Submission Metrics:**
- Test NRMSE: X.XXXX
- Leaderboard Rank: #XX
- Val-Test Gap: X.XX%

**Training Dynamics:**
- Train NRMSE: X.XXXX
- Train-Val Gap: X.XX%
- Converged at epoch: XX

**Observations:**
- [What worked well?]
- [What didn't work?]
- [Surprising findings?]

**Next Steps:**
- [What to try next?]
```

---

## 🎯 Success Metrics

### After 10 Explorations:

**Minimum Success:**
- Understand val-test correlation
- Find at least 1 direction with >5% improvement
- Achieve combined NRMSE < 1.10

**Good Success:**
- Clear best direction identified
- Achieve combined NRMSE < 1.05
- Beat current best (1.14)

**Excellent Success:**
- Multiple promising directions
- Achieve combined NRMSE < 1.00
- Beat SOTA (0.978)

---

## 💡 Key Insights to Extract

1. **Val-Test Correlation:**
   - If high (>0.9): Trust validation, iterate fast
   - If low (<0.7): Need different validation strategy

2. **Primary Bottleneck:**
   - Data quantity?
   - Model capacity?
   - Regularization?
   - Optimization?

3. **Challenge Difficulty:**
   - Is C1 or C2 easier?
   - Which needs more focus?
   - Can we transfer knowledge?

4. **Resource Efficiency:**
   - Is 50 subjects + 50 epochs enough for fast iteration?
   - Does 200 subjects justify 4x training time?

5. **Submission Strategy:**
   - How many submissions to "waste" on exploration?
   - When to start ensembling?
   - Conservative or aggressive?

---

## 🔄 Feedback Loop

```
Run Exp → Analyze Metrics → Submit Best → Observe Leaderboard
    ↑                                              ↓
    └──────────── Adjust Strategy ←───────────────┘
```

After each batch of submissions:
1. Update this document with findings
2. Adjust hypotheses based on evidence
3. Plan next explorations
4. Iterate!

---

## 📚 Output Files

**For Analysis:**
- `experiments/experiments.json` - All experiment configs & metrics
- `experiments/EXPLORATION_RESULTS.md` - Human-readable findings
- `results/exp_X/c{1,2}_results.pt` - Predictions for analysis
- `scripts/exploration_analysis.py` - Automated comparison

**For Submissions:**
- `submissions/exp_X_Y.zip` - Combined C1 + C2 models
- `submissions/SUBMISSION_LOG.md` - Track what was submitted

---

## ✅ Ready to Explore!

**Start with:**
```bash
# Experiment 1 & 2 (Baseline)
CUDA_VISIBLE_DEVICES=1 python train.py -c 1 -d dummy -o --max 50 -e 50 --num 1
CUDA_VISIBLE_DEVICES=1 python train.py -c 2 -d dummy -o --max 50 -e 50 --num 2

# Wait for completion, analyze:
python experiments/analyze_experiments.py

# If looks good, submit:
python create_submission.py \
  --model_c1 checkpoints/c1_best.pth \
  --model_c2 checkpoints/c2_best.pth \
  --output baseline_exp1_2.zip
```

**Then iterate through Exp 3-10!**
