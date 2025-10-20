# Complete Workflow - From Git Clone to Submission

## Step-by-Step Guide (Assuming Fresh Start)

---

## ✅ Step 1: Clone Repository

```bash
git clone https://github.com/owenk-git/2025EEGChallenge.git
cd 2025EEGChallenge
```

---

## ✅ Step 2: Setup Environment

### Option A: Using Conda (Recommended)

```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate eeg2025
```

### Option B: Using pip

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## ✅ Step 3: Download Dataset

You need HBN-EEG data. Choose one option:

### Option A: Mini Dataset (RECOMMENDED for testing - 20 subjects, fast)

1. Go to https://nemar.org
2. Search: **"HBN-EEG R1_mini_L100"**
3. Download the dataset
4. Extract to: `./data/R1_mini_L100/`

Your structure should look like:
```
data/
└── R1_mini_L100/
    └── sub-XXXXXXX/
        └── eeg/
            └── *.set or *.bdf files
```

### Option B: Full Dataset from S3 (100-250 GB, requires AWS CLI)

```bash
# Install AWS CLI
pip install awscli

# Download Release 1
aws s3 cp s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_R1 ./data/R1 --recursive --no-sign-request
```

---

## ✅ Step 4: Verify Setup

```bash
# Test model creation
python -c "from models.eegnet import create_model; print('Model OK')"

# Test dataset (if you have data downloaded)
python -c "from data.dataset import HBNEEGDataset; print('Dataset OK')"
```

---

## ✅ Step 5: Train Challenge 1 Model

```bash
python train.py \
  --challenge 1 \
  --data_path ./data/R1_mini_L100 \
  --epochs 50 \
  --batch_size 32 \
  --lr 0.001
```

**Expected output:**
- Training progress bar
- Loss decreasing each epoch
- Model saved to: `checkpoints/c1_best.pth`
- Training time: 2-4 hours (mini dataset)

**What it does:**
- Loads EEG data from `data_path`
- Trains EEGNeX model for 50 epochs
- Saves best model based on lowest loss
- Saves checkpoint every 10 epochs

---

## ✅ Step 6: Train Challenge 2 Model

```bash
python train.py \
  --challenge 2 \
  --data_path ./data/R1_mini_L100 \
  --epochs 50 \
  --batch_size 32 \
  --lr 0.001
```

**Expected output:**
- Similar to Challenge 1
- Model saved to: `checkpoints/c2_best.pth`

---

## ✅ Step 7: Create Submission Package

```bash
python create_submission.py \
  --model_c1 checkpoints/c1_best.pth \
  --model_c2 checkpoints/c2_best.pth
```

**Expected output:**
- Creates: `YYYYMMDD_HHMM_trained_submission.zip`
- File size: ~1-5 KB (just weights + code)

**What's in the ZIP:**
- `submission.py` - Inference code with model architecture
- `c1_weights.pth` - Trained Challenge 1 weights
- `c2_weights.pth` - Trained Challenge 2 weights

---

## ✅ Step 8: Submit to Codabench

1. Go to: https://www.codabench.org/competitions/9975/
2. Click "Participate" → "Submit"
3. Upload: `YYYYMMDD_HHMM_trained_submission.zip`
4. Wait for evaluation (1-2 hours)

**Expected results (first submission with mini dataset):**
- Overall: 1.2-1.3
- C1: 1.3-1.5
- C2: 1.0-1.1

---

## 🔄 Step 9: Iterate & Improve

### Strategy 1: Train Longer
```bash
python train.py --challenge 1 --data_path ./data/R1_mini_L100 --epochs 100
```

### Strategy 2: Use More Data
```bash
# Download R2, R3 mini datasets
# Train on combined data
python train.py --challenge 1 --data_path ./data/R1_mini_L100 --epochs 100
# Repeat for R2, R3 or combine them
```

### Strategy 3: Tune Hyperparameters
```bash
# Try different learning rates
python train.py --challenge 1 --data_path ./data/R1_mini_L100 --lr 0.0005

# Try different dropout
# Edit models/eegnet.py: dropout=0.25
```

### Strategy 4: Train Ensemble
```bash
# Train multiple models with different seeds
python train.py --challenge 1 --data_path ./data/R1_mini_L100 --epochs 50
# (Modify code to use different random seed)
# Train 3-5 models, then average predictions
```

---

## 📊 Expected Progression

```
First submission (mini R1):
├─ Overall: ~1.2-1.3
├─ C1: 1.3-1.5
└─ C2: 1.0-1.1

After 100 epochs (mini R1):
├─ Overall: ~1.1-1.2
├─ C1: 1.2-1.3
└─ C2: 1.0-1.05

Multiple releases (R1+R2+R3 mini):
├─ Overall: ~1.0-1.1
├─ C1: 1.1-1.2
└─ C2: 0.98-1.02

Full dataset training:
├─ Overall: ~0.95-1.0
├─ C1: 1.0-1.1
└─ C2: 0.97-1.00

Ensemble (3-5 trained models):
├─ Overall: <0.96 ✨
├─ C1: 0.94-0.98
└─ C2: 0.96-1.00
```

---

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'mne'"
```bash
conda activate eeg2025  # or: source venv/bin/activate
pip install -r requirements.txt
```

### "FileNotFoundError: data/R1_mini_L100 not found"
- Download dataset from https://nemar.org
- Extract to correct location
- Check structure: `data/R1_mini_L100/sub-*/eeg/*.set`

### "CUDA out of memory"
```bash
# Reduce batch size
python train.py --challenge 1 --data_path ./data/R1_mini_L100 --batch_size 16

# Or use CPU
CUDA_VISIBLE_DEVICES="" python train.py --challenge 1 ...
```

### "Training loss not decreasing"
- Check learning rate (try 0.0005 or 0.0001)
- Verify dataset loaded correctly
- Check for NaN values in data
- Try training longer (100 epochs)

### "Submission failed on server"
- Check ZIP contains: submission.py, c1_weights.pth, c2_weights.pth
- Verify weights file not corrupted
- Check file sizes (should be ~1-5 KB total)

---

## 💡 Tips for Best Results

1. **Start small:** Mini dataset (R1) → Quick iteration
2. **Monitor training:** Loss should decrease steadily
3. **Save checkpoints:** Don't lose training progress
4. **Use validation set:** Split data for better evaluation
5. **Try multiple seeds:** Train 3-5 models, average predictions
6. **Scale up gradually:** Mini → Multiple mini → Full dataset
7. **Track experiments:** Keep notes on what works

---

## 📝 File Checklist

After training, you should have:

```
✅ checkpoints/c1_best.pth      (Challenge 1 trained weights)
✅ checkpoints/c2_best.pth      (Challenge 2 trained weights)
✅ YYYYMMDD_HHMM_trained_submission.zip  (Ready to upload)
```

---

## 🎯 Your Path to Top 3

**Week 1:** Mini dataset training → Score ~1.1
**Week 2:** Multiple releases → Score ~1.0
**Week 3:** Full dataset → Score ~0.96
**Week 4:** Ensemble → Score <0.96 (Top 3!) 🏆

---

## 📧 Need Help?

- **GitHub Issues:** https://github.com/owenk-git/2025EEGChallenge/issues
- **Competition:** neurips2025-eeg-competition@googlegroups.com

---

**Next Step:** Go to Step 3 (Download Dataset) and start training! 🚀
