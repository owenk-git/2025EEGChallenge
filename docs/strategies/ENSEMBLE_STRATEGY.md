# Ensemble Strategy: Multiple Models in One Submission

**Your Question:** "If we want to do ensemble, do we need to train multiple models?"

**Short Answer:** YES, but you can make submission easy! ✅

---

## 🎯 The Ensemble Problem (Your Sub 6 Experience)

### What Happened with Sub 6:

**Your ensemble submission (Sub 6):**
- Used 3 models
- Score: **1.18** ❌ (worse than Sub 3's 1.14)

**Why it failed:**
- Models had **random weights** (not trained!)
- Ensemble averaging doesn't help random predictions
- Research shows: "32% variance reduction" **only works with trained models**

**Key lesson:**
```
❌ Ensemble of random weights = Bad
✅ Ensemble of TRAINED weights = Good
```

---

## ✅ How to Do Ensemble Correctly

### Option 1: Train Multiple Models with Different Configurations

**Concept:** Train same architecture with different hyperparameters/data.

```bash
# Train Model 1: Low dropout, high LR
python train.py --challenge 1 --use_official \
  --max_subjects 100 --epochs 100 \
  --lr 0.001 --dropout 0.15 \
  --checkpoint_dir ./checkpoints/model1

# Train Model 2: High dropout, medium LR
python train.py --challenge 1 --use_official \
  --max_subjects 100 --epochs 100 \
  --lr 0.0005 --dropout 0.25 \
  --checkpoint_dir ./checkpoints/model2

# Train Model 3: Medium dropout, low LR
python train.py --challenge 1 --use_official \
  --max_subjects 100 --epochs 100 \
  --lr 0.0001 --dropout 0.20 \
  --checkpoint_dir ./checkpoints/model3
```

**Result:** 3 different trained models

**Expected improvement:** Research shows ~5-10% reduction in error (not 32%, that's overly optimistic)

---

### Option 2: Single Model with Different Random Seeds

**Concept:** Train same configuration multiple times with different random initialization.

```bash
# Model 1: Seed 42
python train.py --challenge 1 --use_official \
  --max_subjects 150 --epochs 100 --seed 42 \
  --checkpoint_dir ./checkpoints/seed42

# Model 2: Seed 123
python train.py --challenge 1 --use_official \
  --max_subjects 150 --epochs 100 --seed 123 \
  --checkpoint_dir ./checkpoints/seed123

# Model 3: Seed 999
python train.py --challenge 1 --use_official \
  --max_subjects 150 --epochs 100 --seed 999 \
  --checkpoint_dir ./checkpoints/seed999
```

**Pros:**
- ✅ Easy (same command, different seed)
- ✅ Proven to reduce variance

**Cons:**
- ⚠️ 3x training time

---

### Option 3: Train on Different Data Subsets (Bagging)

**Concept:** Train each model on different random subset of data.

```bash
# Model 1: Subjects 0-100
python train.py --challenge 1 --use_official \
  --max_subjects 100 --subject_offset 0 \
  --checkpoint_dir ./checkpoints/subset1

# Model 2: Subjects 50-150
python train.py --challenge 1 --use_official \
  --max_subjects 100 --subject_offset 50 \
  --checkpoint_dir ./checkpoints/subset2

# Model 3: Subjects 100-200
python train.py --challenge 1 --use_official \
  --max_subjects 100 --subject_offset 100 \
  --checkpoint_dir ./checkpoints/subset3
```

**Expected improvement:** 3-7% error reduction

---

## 🚀 EASY SUBMISSION: Ensemble in Single ZIP

**The key:** Package multiple weights in one submission.py!

### Submission Structure:

```
submission.zip
├── submission.py          # Main submission file
├── c1_model1.pth         # Challenge 1, Model 1
├── c1_model2.pth         # Challenge 1, Model 2
├── c1_model3.pth         # Challenge 1, Model 3
├── c2_model1.pth         # Challenge 2, Model 1
├── c2_model2.pth         # Challenge 2, Model 2
└── c2_model3.pth         # Challenge 2, Model 3
```

### Submission Code:

```python
# submission.py - Ensemble version

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# ... EEGNeX model definition ...

class Submission:
    def __init__(self, SFREQ, DEVICE):
        self.sfreq = SFREQ
        self.device = DEVICE
        self.model_path = load_model_path()

        # Load multiple models for ensemble
        self.models_c1 = self._load_ensemble_c1()
        self.models_c2 = self._load_ensemble_c2()

        print(f"🎯 Ensemble submission loaded")
        print(f"   C1 models: {len(self.models_c1)}")
        print(f"   C2 models: {len(self.models_c2)}")

    def _load_ensemble_c1(self):
        """Load all C1 models"""
        models = []
        n_times = int(2 * self.sfreq)

        # Try loading model1, model2, model3
        for i in range(1, 4):
            weight_path = self.model_path / f"c1_model{i}.pth"
            if weight_path.exists():
                model = EEGNeX(
                    in_chans=129,
                    n_times=n_times,
                    challenge_name='c1',
                    dropout=0.20,
                    output_range=(0.88, 1.12)
                ).to(self.device)

                checkpoint = torch.load(weight_path, map_location=self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                models.append(model)
                print(f"   ✅ Loaded C1 model {i}")
            else:
                print(f"   ⚠️  C1 model {i} not found, skipping")

        if len(models) == 0:
            raise ValueError("No C1 models found!")

        return models

    def _load_ensemble_c2(self):
        """Load all C2 models"""
        models = []
        n_times = int(2 * self.sfreq)

        for i in range(1, 4):
            weight_path = self.model_path / f"c2_model{i}.pth"
            if weight_path.exists():
                model = EEGNeX(
                    in_chans=129,
                    n_times=n_times,
                    challenge_name='c2',
                    dropout=0.20
                ).to(self.device)

                checkpoint = torch.load(weight_path, map_location=self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                models.append(model)
                print(f"   ✅ Loaded C2 model {i}")

        if len(models) == 0:
            raise ValueError("No C2 models found!")

        return models

    def get_model_challenge_1(self):
        """Return ensemble model for C1"""
        def ensemble_predict(eeg_data):
            """Average predictions from all C1 models"""
            predictions = []

            with torch.no_grad():
                for model in self.models_c1:
                    pred = model(eeg_data.to(self.device))
                    predictions.append(pred)

            # Average predictions
            ensemble_pred = torch.mean(torch.stack(predictions), dim=0)
            return ensemble_pred

        return ensemble_predict

    def get_model_challenge_2(self):
        """Return ensemble model for C2"""
        def ensemble_predict(eeg_data):
            """Average predictions from all C2 models"""
            predictions = []

            with torch.no_grad():
                for model in self.models_c2:
                    pred = model(eeg_data.to(self.device))
                    predictions.append(pred)

            # Average predictions
            ensemble_pred = torch.mean(torch.stack(predictions), dim=0)
            return ensemble_pred

        return ensemble_predict
```

**This submission:**
- ✅ Loads multiple models automatically
- ✅ Averages their predictions
- ✅ Gracefully handles missing models (uses what's available)
- ✅ All in ONE submission.py file!

---

## 📦 Creating Ensemble Submission

### Modified create_submission.py:

```python
# create_ensemble_submission.py

import argparse
import zipfile
import shutil
from pathlib import Path
from datetime import import datetime

# ... SUBMISSION_TEMPLATE with ensemble code above ...

def create_ensemble_submission(c1_models, c2_models, output_name=None):
    """
    Create ensemble submission ZIP

    Args:
        c1_models: List of C1 model checkpoint paths
        c2_models: List of C2 model checkpoint paths
        output_name: Output ZIP filename
    """
    print("="*70)
    print("📦 Creating Ensemble Submission Package")
    print("="*70)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    if output_name is None:
        output_name = f"{timestamp}_ensemble_submission.zip"

    # Create temp directory
    temp_dir = Path(f"temp_ensemble_{timestamp}")
    temp_dir.mkdir(exist_ok=True)

    try:
        # Write submission.py (with ensemble code)
        submission_content = ENSEMBLE_SUBMISSION_TEMPLATE.format(timestamp=timestamp)
        with open(temp_dir / "submission.py", 'w') as f:
            f.write(submission_content)
        print(f"✅ Created submission.py")

        # Copy C1 models
        print(f"\n📊 Challenge 1 models:")
        for i, model_path in enumerate(c1_models, 1):
            if Path(model_path).exists():
                shutil.copy(model_path, temp_dir / f"c1_model{i}.pth")
                print(f"   ✅ C1 model {i}: {model_path}")
            else:
                print(f"   ⚠️  C1 model {i} not found: {model_path}")

        # Copy C2 models
        print(f"\n📊 Challenge 2 models:")
        for i, model_path in enumerate(c2_models, 1):
            if Path(model_path).exists():
                shutil.copy(model_path, temp_dir / f"c2_model{i}.pth")
                print(f"   ✅ C2 model {i}: {model_path}")
            else:
                print(f"   ⚠️  C2 model {i} not found: {model_path}")

        # Create ZIP
        with zipfile.ZipFile(output_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in temp_dir.rglob('*'):
                if file.is_file():
                    arcname = file.relative_to(temp_dir)
                    zipf.write(file, arcname)

        zip_size = Path(output_name).stat().st_size / 1024
        print(f"\n✅ Ensemble submission created: {output_name}")
        print(f"   Size: {zip_size:.1f} KB")
        print(f"   C1 models: {len(c1_models)}")
        print(f"   C2 models: {len(c2_models)}")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("="*70)
    print("🚀 Ready to submit ensemble!")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create ensemble submission")

    parser.add_argument('--c1_models', nargs='+', required=True,
                        help='List of C1 model checkpoints')
    parser.add_argument('--c2_models', nargs='+', required=True,
                        help='List of C2 model checkpoints')
    parser.add_argument('--output', type=str, default=None,
                        help='Output ZIP filename')

    args = parser.parse_args()

    create_ensemble_submission(args.c1_models, args.c2_models, args.output)
```

### Usage:

```bash
# Create ensemble submission with 3 models each
python create_ensemble_submission.py \
  --c1_models checkpoints/model1/c1_best.pth \
              checkpoints/model2/c1_best.pth \
              checkpoints/model3/c1_best.pth \
  --c2_models checkpoints/model1/c2_best.pth \
              checkpoints/model2/c2_best.pth \
              checkpoints/model3/c2_best.pth \
  --output ensemble_submission.zip
```

**Result:** ONE zip file with 6 model weights, ready to submit! ✅

---

## 🎯 Training Timeline for Ensemble

### Option 1: Sequential Training (Safe)

```bash
# Day 1: Train Model 1 (your "base" model)
python train.py --challenge 1 --use_official --max_subjects 150 --epochs 100 \
  --lr 0.001 --dropout 0.20 --seed 42 \
  --checkpoint_dir ./checkpoints/model1

python train.py --challenge 2 --use_official --max_subjects 150 --epochs 100 \
  --lr 0.001 --dropout 0.20 --seed 42 \
  --checkpoint_dir ./checkpoints/model1

# Submit Model 1 alone first → Get baseline

# Day 2: Train Model 2 (different hyperparams)
python train.py --challenge 1 --use_official --max_subjects 150 --epochs 100 \
  --lr 0.0005 --dropout 0.25 --seed 123 \
  --checkpoint_dir ./checkpoints/model2

python train.py --challenge 2 --use_official --max_subjects 150 --epochs 100 \
  --lr 0.0005 --dropout 0.25 --seed 123 \
  --checkpoint_dir ./checkpoints/model2

# Day 3: Train Model 3
python train.py --challenge 1 --use_official --max_subjects 150 --epochs 100 \
  --lr 0.0008 --dropout 0.15 --seed 999 \
  --checkpoint_dir ./checkpoints/model3

python train.py --challenge 2 --use_official --max_subjects 150 --epochs 100 \
  --lr 0.0008 --dropout 0.15 --seed 999 \
  --checkpoint_dir ./checkpoints/model3

# Day 4: Create ensemble submission
python create_ensemble_submission.py \
  --c1_models checkpoints/model*/c1_best.pth \
  --c2_models checkpoints/model*/c2_best.pth

# Submit ensemble → Should beat individual models!
```

---

### Option 2: Parallel Training (Fast, if you have GPUs)

```bash
# Run all 3 trainings in parallel (if you have 3 GPUs or patience)

# Terminal 1:
CUDA_VISIBLE_DEVICES=0 python train.py --challenge 1 ... --checkpoint_dir ./checkpoints/model1

# Terminal 2:
CUDA_VISIBLE_DEVICES=1 python train.py --challenge 1 ... --checkpoint_dir ./checkpoints/model2

# Terminal 3:
CUDA_VISIBLE_DEVICES=2 python train.py --challenge 1 ... --checkpoint_dir ./checkpoints/model3

# Done in 1 day instead of 3!
```

---

## 📊 Expected Ensemble Performance

### Research vs Reality:

**Paper claims:** "32% variance reduction"
**Reality:** ~5-15% error reduction (still good!)

**Example:**
- Single model: 0.95
- 3-model ensemble: 0.90-0.92
- 5-model ensemble: 0.88-0.91

**Diminishing returns:**
- 1→3 models: ~5-10% improvement ✅
- 3→5 models: ~2-4% improvement
- 5→10 models: ~1-2% improvement (not worth it)

**Sweet spot: 3-5 models**

---

## 💡 Practical Ensemble Strategies

### Strategy 1: Simple Average (Recommended)

```python
# Average predictions from all models
predictions = [model(x) for model in models]
ensemble_pred = np.mean(predictions)
```

**Pros:** Simple, works well
**Cons:** Treats all models equally

---

### Strategy 2: Weighted Average

```python
# Weight models by validation performance
# Model 1: Loss 0.004 → weight = 1/0.004 = 250
# Model 2: Loss 0.006 → weight = 1/0.006 = 167
# Model 3: Loss 0.005 → weight = 1/0.005 = 200

weights = [250, 167, 200]
weights = np.array(weights) / sum(weights)  # Normalize

predictions = [model(x) for model in models]
ensemble_pred = np.average(predictions, weights=weights)
```

**Pros:** Better models have more influence
**Cons:** Needs validation set to compute weights

---

### Strategy 3: Median Instead of Mean

```python
# Use median for robustness to outliers
predictions = [model(x) for model in models]
ensemble_pred = np.median(predictions)
```

**Pros:** Robust to one bad prediction
**Cons:** Requires odd number of models

---

## 🎯 Recommended Ensemble Approach

### For First Ensemble:

**3 Models, Different Seeds (Easiest)**

```bash
# Same hyperparams, different seeds
python train.py --challenge 1 --use_official --max_subjects 150 --epochs 100 --seed 42
python train.py --challenge 1 --use_official --max_subjects 150 --epochs 100 --seed 123
python train.py --challenge 1 --use_official --max_subjects 150 --epochs 100 --seed 999

# Create ensemble
python create_ensemble_submission.py --c1_models checkpoints/*/c1_best.pth --c2_models checkpoints/*/c2_best.pth
```

**Expected:**
- Single model: ~0.95
- 3-model ensemble: ~0.90
- **5-10% improvement!**

---

### For Advanced Ensemble:

**5 Models, Diverse Configurations**

1. Model 1: Base (LR=0.001, dropout=0.20)
2. Model 2: Low LR (LR=0.0005, dropout=0.20)
3. Model 3: High dropout (LR=0.001, dropout=0.30)
4. Model 4: Different output range (LR=0.001, range=[0.85, 1.15])
5. Model 5: More subjects (max_subjects=300)

**Expected:**
- ~8-12% improvement over single model
- Score: ~0.88-0.92

---

## ✅ Summary: Ensemble Questions Answered

### Q: "Do we need to train multiple models for ensemble?"

**A: YES** ✅
- Your Sub 6 (1.18) failed because models were untrained
- You need 3-5 TRAINED models with different configs/seeds

### Q: "How to make submission easy?"

**A: Package all weights in ONE zip!** ✅
- Use modified submission.py that loads multiple weights
- All models in one ZIP file
- Submission averages predictions automatically
- See create_ensemble_submission.py above

### Q: "Is it worth it?"

**A: YES, but start simple**
- 3 models → 5-10% improvement (worth it!)
- 5 models → 8-12% improvement (if you have time)
- 10+ models → diminishing returns (not worth it)

---

## 🚀 Quick Start: Your First Ensemble

### Day 1:
```bash
# Train 3 models with different seeds
for seed in 42 123 999; do
    python train.py --challenge 1 --use_official --max_subjects 100 --epochs 50 --seed $seed
    python train.py --challenge 2 --use_official --max_subjects 100 --epochs 50 --seed $seed
done
```

### Day 2:
```bash
# Create ensemble submission
python create_ensemble_submission.py \
  --c1_models checkpoints/*/c1_best.pth \
  --c2_models checkpoints/*/c2_best.pth

# Submit!
```

**Expected result:** ~5-10% better than single model!

---

## 📁 Files to Create

I can create these for you:

1. **create_ensemble_submission.py** - Creates ZIP with multiple models
2. **train_ensemble.sh** - Script to train multiple models automatically
3. **ensemble_submission_template.py** - Submission.py with ensemble logic

Let me know if you want these files created!
