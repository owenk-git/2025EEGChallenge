# 🚀 Exploration Phase - Quick Start

## Goal: Run 10 strategic experiments to find best direction

**Strategy:** Explore → Observe → Exploit

---

## ⚡ Option 1: Run All at Once (Recommended for overnight)

```bash
# Run all 10 experiments sequentially
./scripts/run_exploration.sh

# Or specify GPU:
./scripts/run_exploration.sh 1  # Use GPU 1
```

**Time:** ~20-30 hours total (varies by GPU)

---

## ⚡ Option 2: Run by Groups (Recommended for manual iteration)

### Group 1: Baseline (FASTEST - Start Here!)
```bash
./scripts/run_exp_group.sh 1

# Exp 1: C1, 50 subjects, 50 epochs (~1 hour)
# Exp 2: C2, 50 subjects, 50 epochs (~1 hour)
```

**After completion:**
```bash
# Analyze
python scripts/compare_exploration.py

# If looks good, submit
python create_submission.py \
  --model_c1 checkpoints/c1_best.pth \
  --model_c2 checkpoints/c2_best.pth \
  --output baseline.zip
```

### Group 2: More Data
```bash
./scripts/run_exp_group.sh 2

# Exp 3: C1, 200 subjects, 100 epochs (~4 hours)
# Exp 4: C2, 200 subjects, 100 epochs (~4 hours)
```

### Group 3: Higher Dropout
```bash
./scripts/run_exp_group.sh 3

# Exp 5: C1, 100 subjects, 100 epochs, drop=0.4 (~2 hours)
# Exp 6: C2, 100 subjects, 100 epochs, drop=0.4 (~2 hours)
```

### Group 4: Lower Learning Rate
```bash
./scripts/run_exp_group.sh 4

# Exp 7: C1, 100 subjects, 150 epochs, lr=5e-4 (~3 hours)
# Exp 8: C2, 100 subjects, 150 epochs, lr=5e-4 (~3 hours)
```

### Group 5: Larger Batch
```bash
./scripts/run_exp_group.sh 5

# Exp 9: C1, 100 subjects, 100 epochs, batch=64 (~2 hours)
# Exp 10: C2, 100 subjects, 100 epochs, batch=64 (~2 hours)
```

---

## 📊 After Each Group: Analyze Results

```bash
# Compare all experiments
python scripts/compare_exploration.py

# With plots
python scripts/compare_exploration.py --plot

# View detailed results
python experiments/analyze_experiments.py
```

**Example output:**
```
🔬 EXPLORATION RESULTS (Exp 1-10)
====================================================================================================
Exp   Challenge  Group           Subjects   Epochs   Dropout    LR         Batch    Val NRMSE
----------------------------------------------------------------------------------------------------
1     C1         Baseline        50         50       0.20       1.0e-03    32       1.3456
2     C2         Baseline        50         50       0.20       1.0e-03    32       1.1234
3     C1         More Data       200        100      0.20       1.0e-03    32       1.2345
4     C2         More Data       200        100      0.20       1.0e-03    32       1.0543
====================================================================================================

📊 DIRECTION ANALYSIS
====================================================================================================

Baseline (50 subj):
  Average Val NRMSE: 1.2345
  Combined Score: 1.1234 (0.3×C1 + 0.7×C2)

More Data (200 subj):
  Average Val NRMSE: 1.1444
  Combined Score: 1.0432 (0.3×C1 + 0.7×C2)
  vs Baseline: +7.3% improvement ✅

====================================================================================================
💡 RECOMMENDATIONS
====================================================================================================

Baseline performance: 1.2345
🏆 Best direction: More Data (200 subj) (+7.3% improvement)

✅ Direction: DATA QUANTITY
Next steps:
  - Use 300-500 subjects
  - Try full dataset
  - Longer training (150-200 epochs)
====================================================================================================
```

---

## 🎯 Decision Flow

### After Group 1 (Baseline):

**If Val NRMSE > 1.3:**
- ❌ Something wrong with setup
- Check data loading, model architecture
- Verify training is working

**If Val NRMSE 1.1-1.3:**
- ✅ Good baseline
- Continue to other groups

**If Val NRMSE < 1.1:**
- 🎉 Excellent baseline!
- May beat current best already
- Submit immediately

### After All Groups:

**Best is "More Data":**
→ Use 300+ subjects, longer training

**Best is "High Dropout":**
→ Try dropout 0.3, 0.5, add augmentation

**Best is "Lower LR":**
→ Try lr 1e-4, 2e-4, cosine schedule

**Best is "Large Batch":**
→ Try batch 96, 128, gradient accumulation

**All similar:**
→ Try different architecture (Phase 2)

---

## 📝 Experiment Tracking

### View All Experiments:
```bash
# JSON format
cat experiments/experiments.json

# Human-readable
python experiments/analyze_experiments.py

# Compare specific experiments
python scripts/compare_exploration.py
```

### Check Predictions:
```python
import torch

# Load experiment 1 results
results = torch.load('results/exp_1/c1_results.pt')

print(results['metrics'])
# {'nrmse': 1.2345, 'pearson_r': 0.78, 'r2': 0.65, ...}

print(f"Predictions: {results['predictions'].shape}")
print(f"Targets: {results['targets'].shape}")
```

---

## 🚢 Creating Submissions

### After Each Group:
```bash
# Create submission from best experiments so far
python create_submission.py \
  --model_c1 checkpoints/c1_best.pth \
  --model_c2 checkpoints/c2_best.pth \
  --output group1_submission.zip

# Upload to Codabench
# Record score in experiments/SUBMISSION_LOG.md
```

### Submission Log Template:
```markdown
## Submission 1: Baseline (Exp 1-2)
- Date: 2025-10-20
- Models: Exp 1 (C1), Exp 2 (C2)
- Val NRMSE: C1=1.35, C2=1.12, Combined=1.19
- Test NRMSE: ??? (after submission)
- Leaderboard: ???
- Notes: Baseline, 50 subjects, 50 epochs
```

---

## 📈 Expected Timeline

### Fast Track (Manual iteration):
- **Day 1:** Group 1 (Baseline) → Submit
- **Day 2:** Best of Groups 2-5 → Submit
- **Day 3:** Analyze, pick direction
- **Day 4-7:** Exploitation phase

### Thorough Track (Full exploration):
- **Day 1-2:** Run all 10 experiments
- **Day 3:** Analyze results, submit top 3
- **Day 4:** Wait for leaderboard feedback
- **Day 5-10:** Exploitation phase

---

## ⚠️ Important Notes

1. **Subject-wise splitting is ON by default** - More realistic validation
2. **Val NRMSE will be 5-15% worse** than old random split
3. **This is CORRECT** - previous estimates were inflated
4. **Comprehensive metrics logged** - NRMSE, Pearson, R², CCC, etc.
5. **Predictions saved** - Can analyze errors later

---

## 🎓 What Each Group Tests

| Group | Hypothesis | If Wins → Means |
|-------|------------|-----------------|
| 1. Baseline | Establish performance | N/A (reference) |
| 2. More Data | Data quantity bottleneck | Need more subjects/data |
| 3. High Dropout | Overfitting issue | Model too complex, need regularization |
| 4. Lower LR | Optimization issue | Converging too fast, need slower training |
| 5. Large Batch | Training dynamics | Batch size affects generalization |

---

## 🚀 Getting Started NOW

```bash
# 1. Start with baseline (fastest)
./scripts/run_exp_group.sh 1

# 2. While training, read strategy
cat docs/EXPLORATION_STRATEGY.md

# 3. After completion, analyze
python scripts/compare_exploration.py

# 4. Create first submission
python create_submission.py \
  --model_c1 checkpoints/c1_best.pth \
  --model_c2 checkpoints/c2_best.pth

# 5. Upload to Codabench
# https://www.codabench.org/competitions/9975/

# 6. Continue with other groups based on results
```

---

## 📚 Full Documentation

- [EXPLORATION_STRATEGY.md](docs/EXPLORATION_STRATEGY.md) - Complete strategy & decision tree
- [ROBUST_EVALUATION_IMPLEMENTED.md](docs/ROBUST_EVALUATION_IMPLEMENTED.md) - Evaluation system details
- [ULTRATHINK_ROBUST_EVALUATION.md](docs/ULTRATHINK_ROBUST_EVALUATION.md) - Analysis & roadmap

---

## ✅ Ready to Explore!

**All systems production-ready. Start with Group 1 (Baseline)!** 🚀
