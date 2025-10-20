# Experiment Log

Track all training experiments with configurations, results, and insights.

## Quick Reference

| Exp # | Challenge | Description | Best Loss | Status |
|-------|-----------|-------------|-----------|--------|
| 1 | C1 | Baseline - 100 subjects, 100 epochs | TBD | 🟡 Running |

---

## Experiment Details

### Experiment 1: Baseline Training (C1)
**Date:** TBD
**Challenge:** 1 (Response Time Prediction)
**Status:** 🟡 Planned

**Configuration:**
```bash
CUDA_VISIBLE_DEVICES=1 python train.py -c 1 -d dummy -o --max 100 -e 100 --num 1
```

**Hyperparameters:**
- Max subjects: 100
- Epochs: 100
- Batch size: 32 (default)
- Learning rate: 1e-3 (default)
- Dropout: 0.20 (default)
- Dataset: Official mini

**Approach:**
- First full training run with official dataset
- Using proven EEGNeX architecture
- Sigmoid-inside-classifier (proven in Sub 5)
- Baseline to establish performance metrics

**Expected Outcome:**
- Establish baseline performance on Challenge 1
- Validate data loading and training pipeline
- Get initial loss metrics for comparison

**Results:**
- Final validation loss: TBD
- Best epoch: TBD
- Training time: TBD

**Insights:**
- TBD after training completes

**Next Steps:**
- TBD based on results

---

## Experiment Template

Copy this template for new experiments:

### Experiment X: [Brief Description]
**Date:** YYYY-MM-DD
**Challenge:** [1 or 2]
**Status:** [🟡 Planned | 🔵 Running | 🟢 Complete | 🔴 Failed]

**Configuration:**
```bash
CUDA_VISIBLE_DEVICES=X python train.py -c X -d dummy -o --max X -e X --num X
```

**Hyperparameters:**
- Max subjects: X
- Epochs: X
- Batch size: X
- Learning rate: X
- Dropout: X
- Dataset: [Official mini / Official full / Custom]

**Approach:**
- What strategy are you testing?
- What changes from previous experiments?
- What hypothesis are you validating?

**Expected Outcome:**
- What do you hope to achieve?
- What metrics are you targeting?

**Results:**
- Final validation loss: X.XXXX
- Best epoch: X
- Training time: X hours

**Insights:**
- What worked?
- What didn't work?
- Surprising findings?

**Next Steps:**
- Based on results, what to try next?

---

## Strategy Reference

### Training Strategies (from docs/strategies/TRAINING_STRATEGIES.md)
1. **Progressive Training** - Start small, gradually increase
2. **C2 Focus** - Prioritize Challenge 2 (70% weight)
3. **Hyperparameter Search** - Systematic exploration
4. **Multi-Task Learning** - Train both challenges together
5. **Data Augmentation** - Add noise, time shifts, channel drops

### Inference Strategies
1. **Test-Time Augmentation (TTA)** - Average multiple augmented predictions
2. **Prediction Clipping** - Clip to valid ranges
3. **Confidence Weighting** - Weight by model confidence
4. **Multi-Window** - Use multiple time windows
5. **Temperature Scaling** - Calibrate predictions

---

## Performance Tracking

### Best Results
- **Challenge 1:** TBD (Exp #X)
- **Challenge 2:** TBD (Exp #X)
- **Combined:** TBD (weighted 30/70)

### Target Metrics
- Current best: 1.14 (C1: 1.45, C2: 1.01)
- SOTA target: 0.978
- Submissions remaining: 25

---

## Notes

- All experiments automatically logged to `experiments/experiments.json`
- Use `--num X` to track experiment number
- Update this document with insights after each experiment
- Focus on reproducibility - document everything!
