# Quick Start Guide

Fast reference for running experiments on the NeurIPS 2025 EEG Challenge.

## One-Line Commands

### Test Run (5 minutes)
```bash
CUDA_VISIBLE_DEVICES=1 python train.py -c 1 -d dummy -o -m --max 5 -e 3
```

### Experiment 1: Baseline C1
```bash
CUDA_VISIBLE_DEVICES=1 python train.py -c 1 -d dummy -o --max 100 -e 100 --num 1
```

### Experiment 2: Baseline C2
```bash
CUDA_VISIBLE_DEVICES=1 python train.py -c 2 -d dummy -o --max 100 -e 100 --num 2
```

### Train Both Challenges in Parallel
```bash
# Terminal 1 (GPU 0)
CUDA_VISIBLE_DEVICES=0 python train.py -c 1 -d dummy -o --max 100 -e 100 --num 1

# Terminal 2 (GPU 1)
CUDA_VISIBLE_DEVICES=1 python train.py -c 2 -d dummy -o --max 100 -e 100 --num 2
```

## Argument Reference

| Short | Long | Description | Example |
|-------|------|-------------|---------|
| `-c` | `--challenge` | Challenge 1 or 2 | `-c 1` |
| `-d` | `--data_path` | Data path (use "dummy" with `-o`) | `-d dummy` |
| `-o` | `--use_official` | Use official dataset | `-o` |
| `-m` | `--official_mini` | Use mini dataset (faster) | `-m` |
| `-e` | `--epochs` | Number of epochs | `-e 100` |
| `-b` | `--batch_size` | Batch size | `-b 32` |
| `--lr` | `--lr` | Learning rate | `--lr 1e-3` |
| `--max` | `--max_subjects` | Max subjects | `--max 100` |
| `--drop` | `--dropout` | Dropout rate | `--drop 0.2` |
| `-w` | `--num_workers` | Data workers | `-w 4` |
| `--num` | `--exp_num` | Experiment number | `--num 1` |

## Experiment Tracking

### Run with tracking:
```bash
python train.py -c 1 -d dummy -o --max 100 -e 100 --num 1
```

### View results:
```bash
python experiments/analyze_experiments.py
```

### See experiment history:
```bash
cat experiments/EXPERIMENT_LOG.md
```

## Common Workflows

### Quick Test
```bash
# 5 subjects, 3 epochs (5 min)
python train.py -c 1 -d dummy -o -m --max 5 -e 3
```

### Standard Training
```bash
# 100 subjects, 100 epochs (~6 hours)
python train.py -c 1 -d dummy -o --max 100 -e 100 --num 1
```

### Large-Scale Training
```bash
# 200 subjects, 150 epochs (~18 hours)
python train.py -c 1 -d dummy -o --max 200 -e 150 --num 1
```

### Hyperparameter Search

**Try different dropouts:**
```bash
python train.py -c 1 -d dummy -o --max 100 -e 100 --drop 0.1 --num 3
python train.py -c 1 -d dummy -o --max 100 -e 100 --drop 0.3 --num 4
```

**Try different learning rates:**
```bash
python train.py -c 1 -d dummy -o --max 100 -e 100 --lr 5e-4 --num 5
python train.py -c 1 -d dummy -o --max 100 -e 100 --lr 5e-3 --num 6
```

## File Locations

- **Trained models:** `checkpoints/c{1,2}_best.pth`
- **Experiment logs:** `experiments/experiments.json`
- **Experiment notes:** `experiments/EXPERIMENT_LOG.md`
- **Analysis script:** `experiments/analyze_experiments.py`

## Next Steps

1. Run baseline experiments (Exp 1 & 2)
2. Analyze results: `python experiments/analyze_experiments.py`
3. Update `experiments/EXPERIMENT_LOG.md` with insights
4. Try suggested improvements
5. Create submission when ready

## See Also

- [Full Documentation](docs/START_HERE.md)
- [Training Strategies](docs/strategies/TRAINING_STRATEGIES.md)
- [Experiment Tracking](experiments/README.md)
- [Scripts Guide](docs/SCRIPTS_GUIDE.md)
