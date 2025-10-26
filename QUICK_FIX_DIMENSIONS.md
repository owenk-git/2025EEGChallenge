# Quick Fix for Dimension Mismatch

## Problem

The models expect `n_times=900` but actual data has ~200 time points.

**Error**: `RuntimeError: mat1 and mat2 shapes cannot be multiplied (64x1536 and 7168x256)`

- Expected: `128 * 56 = 7168` (for 900 time points)
- Actual: `128 * 12 = 1536` (for ~200 time points)

## Quick Fix

Add `--n_times 200` to all training commands:

```bash
# Domain Adaptation
python3 train_domain_adaptation_direct.py --challenge c1 --epochs 100 --batch_size 64 --n_times 200

# Cross-Task
python3 train_cross_task_direct.py --challenge c1 --epochs 100 --pretrain_epochs 50 --batch_size 64 --n_times 200

# Hybrid
python3 train_hybrid_direct.py --challenge c1 --epochs 100 --batch_size 64 --n_times 200
```

But the training scripts don't have `--n_times` parameter yet. Let me add it...
