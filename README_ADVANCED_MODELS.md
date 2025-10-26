# Advanced Models for NeurIPS 2025 EEG Challenge

Three state-of-the-art models designed to beat 0.976 score through domain adaptation and cross-subject optimization.

## Quick Start

### Train All Models (Recommended)

```bash
chmod +x train_all_advanced_models.sh
./train_all_advanced_models.sh
```

This trains all 3 models and creates submissions (~6-8 hours on GPU).

### Or Train Individual Models

```bash
# Domain Adaptation EEGNeX
python3 train_domain_adaptation.py --challenge c1 --epochs 100
python3 train_domain_adaptation.py --challenge c2 --epochs 100

# Cross-Task Pre-Training
python3 train_cross_task.py --challenge c1 --epochs 100 --pretrain_epochs 50
python3 train_cross_task.py --challenge c2 --epochs 100 --pretrain_epochs 50

# Hybrid CNN-Transformer-DA
python3 train_hybrid.py --challenge c1 --epochs 100
python3 train_hybrid.py --challenge c2 --epochs 100
```

### Create Submissions

```bash
# Domain Adaptation
python3 create_advanced_submission.py \
    --model domain_adaptation \
    --name domain_adaptation_v1 \
    --checkpoint_c1 checkpoints/domain_adaptation_c1_best.pt \
    --checkpoint_c2 checkpoints/domain_adaptation_c2_best.pt

# Cross-Task
python3 create_advanced_submission.py \
    --model cross_task \
    --name cross_task_pretrain_v1 \
    --checkpoint_c1 checkpoints/cross_task_c1_best.pt \
    --checkpoint_c2 checkpoints/cross_task_c2_best.pt

# Hybrid
python3 create_advanced_submission.py \
    --model hybrid \
    --name hybrid_cnn_transformer_v1 \
    --checkpoint_c1 checkpoints/hybrid_c1_best.pt \
    --checkpoint_c2 checkpoints/hybrid_c2_best.pt
```

## Models Overview

| Model | Key Features | Expected Score |
|-------|-------------|----------------|
| **Domain Adaptation EEGNeX** | MMD + Entropy + Adversarial Training | 1.05-1.10 |
| **Cross-Task Pre-Training** | Multi-task learning + Fine-tuning | 1.02-1.08 |
| **Hybrid CNN-Transformer-DA** | CNN + Transformer + ERP + DA | 1.01-1.07 |

**Current Best**: 1.09 (C1: 1.31, C2: 1.00)
**Target**: 0.976 (C1: 0.92, C2: 1.00)

## Why These Models?

All previous models (CNN Ensemble, Transformer, EEGNeX, ERP MLP) hit the same ceiling (~1.10) due to **cross-subject generalization failure**. These new models specifically address this with:

1. **Domain Adaptation**: Aligns feature distributions across subjects
2. **Cross-Task Pre-Training**: Competition-recommended approach
3. **Hybrid Architecture**: Combines 2024 SOTA methods (CNN + Transformer)

## Documentation

- [**IMPLEMENTATION_SUMMARY.md**](IMPLEMENTATION_SUMMARY.md): Complete technical details
- [**TRAINING_GUIDE.md**](TRAINING_GUIDE.md): Step-by-step training guide
- [**LITERATURE_REVIEW.md**](LITERATURE_REVIEW.md): Comprehensive literature review (19 papers)

## Files

### Models
- `models/domain_adaptation_eegnex.py`
- `models/cross_task_pretrain.py`
- `models/hybrid_cnn_transformer_da.py`

### Training
- `train_domain_adaptation.py`
- `train_cross_task.py`
- `train_hybrid.py`
- `train_all_advanced_models.sh` (master script)

### Submission
- `create_advanced_submission.py`

## Requirements

- PyTorch (only dependency)
- NumPy
- CUDA-capable GPU (recommended)

✅ **No external dependencies** (scipy, sklearn, XGBoost, torchvision)
✅ **Docker compatible**: `sylvchev/codalab-eeg2025:v14`

## Expected Timeline

- Training: 6-8 hours (all 3 models)
- Submission creation: 15 minutes
- Competition evaluation: 3-6 hours (3 submissions)
- **Total: ~10-12 hours to results**

## Next Steps

1. Run `./train_all_advanced_models.sh`
2. Wait for training to complete
3. Submit all 3 models to competition
4. Analyze which approach works best
5. Iterate based on results

Good luck! 🚀
