# Experiment Tracking System

Automated experiment logging and analysis for the NeurIPS 2025 EEG Challenge.

## Quick Start

### 1. Run an experiment with tracking:
```bash
CUDA_VISIBLE_DEVICES=1 python train.py -c 1 -d dummy -o --max 100 -e 100 --num 1
```

The `--num 1` parameter:
- Automatically logs configuration and results to `experiments.json`
- Links to [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) for detailed notes
- Enables trend analysis and performance tracking

### 2. Analyze results:
```bash
# View all experiments
python experiments/analyze_experiments.py

# Filter by challenge
python experiments/analyze_experiments.py --challenge 1

# Show top 10 best
python experiments/analyze_experiments.py --best 10
```

## File Structure

```
experiments/
├── README.md              # This file
├── EXPERIMENT_LOG.md      # Human-readable experiment notes
├── experiments.json       # Auto-generated experiment data
└── analyze_experiments.py # Analysis script
```

## Workflow

### Before Training
1. Check [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) to see what's been tried
2. Choose next experiment number (incrementing from last)
3. Plan your experiment approach

### During Training
```bash
# Run with experiment tracking
CUDA_VISIBLE_DEVICES=1 python train.py \
  -c 1 \              # Challenge 1 or 2
  -d dummy \          # Data path (ignored with -o)
  -o \                # Use official dataset
  --max 100 \         # Max subjects
  -e 100 \            # Epochs
  --num 1             # Experiment number (IMPORTANT!)
```

### After Training
1. Results auto-logged to `experiments.json`
2. Update [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) with:
   - Insights and observations
   - What worked / didn't work
   - Ideas for next experiment
3. Run analysis to find trends:
   ```bash
   python experiments/analyze_experiments.py
   ```

## Example Experiment Sequence

```bash
# Exp 1: Baseline C1
CUDA_VISIBLE_DEVICES=1 python train.py -c 1 -d dummy -o --max 100 -e 100 --num 1

# Exp 2: Baseline C2
CUDA_VISIBLE_DEVICES=1 python train.py -c 2 -d dummy -o --max 100 -e 100 --num 2

# Exp 3: C1 with higher dropout
CUDA_VISIBLE_DEVICES=1 python train.py -c 1 -d dummy -o --max 100 -e 100 --drop 0.3 --num 3

# Exp 4: C2 with more subjects
CUDA_VISIBLE_DEVICES=1 python train.py -c 2 -d dummy -o --max 200 -e 100 --num 4

# Exp 5: C1 with lower learning rate
CUDA_VISIBLE_DEVICES=1 python train.py -c 1 -d dummy -o --max 100 -e 150 --lr 5e-4 --num 5
```

## Analysis Features

The analysis script provides:

1. **Summary Table** - All experiments with key metrics
2. **Best Performers** - Top N experiments by validation loss
3. **Trend Analysis** - Hyperparameter correlations
4. **Suggestions** - Data-driven recommendations for next experiments

### Example Output:
```
====================================================================================================
EXPERIMENT SUMMARY
====================================================================================================
Exp   Challenge  Subjects   Epochs   LR         Dropout    Val Loss     Best Epoch
----------------------------------------------------------------------------------------------------
1     C1         100        100      1.0e-03    0.20       0.1234       67
2     C2         100        100      1.0e-03    0.20       0.0987       82
3     C1         100        100      1.0e-03    0.30       0.1189       71
====================================================================================================

🏆 Top 3 Best Experiments:
   1. Exp #3 (C1) - Loss: 0.1189
   2. Exp #1 (C1) - Loss: 0.1234
   3. Exp #2 (C2) - Loss: 0.0987

====================================================================================================
TREND ANALYSIS
====================================================================================================

📊 Challenge 1 Trends:
   Best: Exp #3 - Loss: 0.1189
   Average Loss: 0.1212

   Dropout Impact:
      0.2: 0.1234 (n=1)
      0.3: 0.1189 (n=1)

====================================================================================================
SUGGESTIONS FOR NEXT EXPERIMENTS
====================================================================================================

✅ Best C1: Exp #3 - Loss: 0.1189
   Config: {
      "epochs": 100,
      "dropout": 0.3,
      ...
   }

💡 Recommendations:
   1. Try higher max_subjects (200+) if performance is improving with more data
   2. Experiment with dropout (0.1-0.3) if overfitting is observed
   3. Adjust learning rate (1e-4 to 5e-3) based on loss curves
   4. Consider longer training (150+ epochs) if still improving
   5. Test data augmentation strategies (see docs/strategies/)
====================================================================================================
```

## Tips

1. **Always use `--num`** - This is how we track experiments!
2. **Update EXPERIMENT_LOG.md** - Automation can't capture insights
3. **Run analysis often** - Spot trends early
4. **Be systematic** - Change one thing at a time when possible
5. **Document failures** - Failed experiments provide valuable info

## Integration with Competition

- 25 submissions remaining
- Current best: 1.14 (C1: 1.45, C2: 1.01)
- SOTA target: 0.978
- Focus on C2 (70% of final score)

Track which experiments lead to submissions and their leaderboard scores in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

## See Also

- [Training Strategies](../docs/strategies/TRAINING_STRATEGIES.md)
- [Inference Strategies](../docs/strategies/INFERENCE_STRATEGIES.md)
- [Ensemble Strategy](../docs/strategies/ENSEMBLE_STRATEGY.md)
