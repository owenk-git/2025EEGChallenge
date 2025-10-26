# Simple Training Commands

## 🚀 Quick Start (No Preprocessing!)

### Option 1: Train Domain Adaptation ONLY (Recommended)

**Fastest way to get results:**

```bash
# Make script executable
chmod +x train_domain_adaptation_only.sh

# Train (takes ~2-3 hours for both C1 and C2)
./train_domain_adaptation_only.sh
```

This will:
1. ✅ Train Domain Adaptation for C1 and C2
2. ✅ Create submission automatically
3. ✅ No preprocessing needed!

**Expected improvement**: 1.09 → 1.05-1.10

---

### Option 2: Manual Training Commands

```bash
# Train C1
python3 train_domain_adaptation_direct.py --challenge c1 --epochs 100 --batch_size 64

# Train C2
python3 train_domain_adaptation_direct.py --challenge c2 --epochs 100 --batch_size 64

# Create submission
python3 create_advanced_submission.py \
    --model domain_adaptation \
    --name domain_adaptation_v1 \
    --checkpoint_c1 checkpoints/domain_adaptation_c1_best.pt \
    --checkpoint_c2 checkpoints/domain_adaptation_c2_best.pt
```

---

### Option 3: Quick Test with Mini Dataset

```bash
# Test with small dataset first (takes ~5-10 minutes)
python3 train_domain_adaptation_direct.py --challenge c1 --epochs 10 --batch_size 32 --mini

# If it works, run full training
python3 train_domain_adaptation_direct.py --challenge c1 --epochs 100 --batch_size 64
python3 train_domain_adaptation_direct.py --challenge c2 --epochs 100 --batch_size 64
```

---

## ⚠️ Important Note

Currently **only Domain Adaptation has direct loading** implemented.

### For Other Models (Cross-Task, Hybrid):

You need to preprocess first:

```bash
# Step 1: Preprocess (one time, 10-30 minutes)
python3 preprocess_data_for_advanced_models.py --challenges c1,c2

# Step 2: Check data is ready
python3 check_data_ready.py

# Step 3: Train other models
python3 train_cross_task.py --challenge c1 --epochs 100 --pretrain_epochs 50
python3 train_hybrid.py --challenge c1 --epochs 100
```

---

## Which Command to Use?

### If you want fastest results:
```bash
./train_domain_adaptation_only.sh
```

### If you want to test first:
```bash
python3 train_domain_adaptation_direct.py --challenge c1 --epochs 10 --mini
```

### If you want to train all 3 models:
```bash
# Preprocess first
python3 preprocess_data_for_advanced_models.py --challenges c1,c2

# Then train all
./train_all_advanced_models.sh
```

---

## Summary Table

| Method | Command | Preprocessing | Time | Models |
|--------|---------|--------------|------|--------|
| **Quick Test** | `--mini --epochs 10` | ❌ No | 5-10 min | Domain Adaptation only |
| **Domain Adaptation Only** | `./train_domain_adaptation_only.sh` | ❌ No | 2-3 hours | Domain Adaptation only |
| **All 3 Models** | `./train_all_advanced_models.sh` | ✅ Yes | 6-8 hours | All 3 models |

---

## Recommendation

**Start with Domain Adaptation only:**
```bash
chmod +x train_domain_adaptation_only.sh
./train_domain_adaptation_only.sh
```

This gives you:
- ✅ Fastest path to submission
- ✅ No preprocessing needed
- ✅ Most promising model (10-15% improvement expected)
- ✅ Ready in 2-3 hours

Then if results are good, train the other 2 models!
