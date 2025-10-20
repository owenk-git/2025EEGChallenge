# Competition Requirements Verification ✅

## 🎯 Challenge Overview (Double-Checked)

### Challenge 1: Cross-Task Transfer Learning (30% of score)
- **Task:** Predict response time from EEG data
- **Data:** Contrast Change Detection (CCD) task
- **Metric:** Normalized RMSE
- **Approach:** Use passive tasks for pretraining, fine-tune for CCD
- **Output:** 1 value (response time)

### Challenge 2: Externalizing Factor Prediction (70% of score)
- **Task:** Predict psychopathology score from EEG data
- **Data:** EEG recordings from specific task
- **Metric:** Normalized RMSE
- **Approach:** Unsupervised/self-supervised pretraining
- **Output:** 1 value (externalizing factor)

**Overall Score:** 0.3 × C1_NRMSE + 0.7 × C2_NRMSE (lower is better)

---

## ✅ Dataset Verification

### Available Data (11 Releases + 1 Non-Commercial)

| Release | S3 URI | Subjects | Size | Status |
|---------|--------|----------|------|--------|
| R1 | `s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_R1` | 136 | 103 GB | ✅ Accessible |
| R2 | `s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_R2` | 152 | 120 GB | ✅ Accessible |
| R3 | `s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_R3` | 183 | 140 GB | ✅ Accessible |
| R4 | `s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_R4` | 324 | 230 GB | ✅ Accessible |
| R5 | `s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_R5` | 330 | 224 GB | ✅ Accessible |
| R6 | `s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_R6` | 134 | 91 GB | ✅ Accessible |
| R7 | `s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_R7` | 381 | 245 GB | ✅ Accessible |
| R8 | `s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_R8` | 257 | 157 GB | ✅ Accessible |
| R9 | `s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_R9` | 295 | 185 GB | ✅ Accessible |
| R10 | `s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_R10` | 295 | 160 GB | ✅ Accessible |
| R11 | `s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_R11` | 295 | 220 GB | ✅ Accessible |
| NC | `s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_NC` | 458 | 251 GB | ⚠️ Not for commercial |

**Total:** 3,000+ subjects, ~2 TB

**Mini Datasets (For Testing):**
- Each release has mini version: 20 subjects, 100 Hz, ~500 MB
- Available on NEMAR.org and Google Drive
- Format: BDF or SET (both supported by MNE-Python)

---

## ✅ Our Implementation Verification

### 1. S3 Streaming ✅ CORRECT
```python
# Our code:
data_path = "s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_R1"
--use_streaming

# Uses s3fs to stream files without download ✅
# Matches competition requirement: "Data available on Amazon S3" ✅
```

### 2. Data Format ✅ CORRECT
```python
# Our code supports both:
- .bdf files (via mne.io.read_raw_bdf) ✅
- .set files (via mne.io.read_raw_eeglab) ✅

# Competition provides: BDF and SET formats ✅
```

### 3. Sampling Rate ✅ CORRECT
```python
# Our code:
target_sfreq = 100  # Downsample to 100 Hz

# Competition requirement:
# "Data will be downsampled to 100 Hz for evaluation" ✅
```

### 4. Model Architecture ✅ CORRECT
```python
# Our EEGNeX:
- Input: 129 channels (HBN standard)
- Output: 1 value (response time or externalizing)
- Challenge 1: Uses sigmoid inside classifier ✅
- Challenge 2: Linear output ✅
```

---

## ⚠️ CRITICAL ISSUES TO FIX

### Issue 1: Behavioral Targets ❌ MISSING

**Problem:** We don't have the actual behavioral targets!

**What we need:**
- Challenge 1: Response time from CCD task
- Challenge 2: Externalizing factor scores

**Where they are:**
- HBN phenotype CSV files
- Should be on S3 or downloadable separately

**Current status:**
- Our `behavioral_streaming.py` uses **SYNTHETIC data** ❌
- This is only for TESTING
- Need to find real phenotype files

**Action required:**
```python
# Need to locate and load:
# - Response times from CCD task
# - Externalizing factor scores
# - Match by subject ID

# Possible locations:
# s3://fcp-indi/data/Projects/HBN/phenotype/...
```

---

### Issue 2: Task-Specific Data ⚠️ UNCLEAR

**Challenge 1 says:**
- "Use Contrast Change Detection (CCD) task"
- "Predict response time"

**Question:** Are all subjects in HBN-EEG doing CCD task?
- Competition mentions 6 cognitive tasks
- 3 passive, 3 active
- Which one is CCD?

**Our assumption:**
- Stream any EEG data from HBN
- Load response time from phenotype data

**Need to verify:** Is this correct approach?

---

### Issue 3: Submission Format ✅ MOSTLY CORRECT

**Competition requirements:**
1. ✅ Code submission (we have `create_submission.py`)
2. ✅ Use `load_model_path()` helper (we have this)
3. ✅ Include model weights (we package .pth files)
4. ⚠️ Must run on single GPU with 20 GB memory

**Our model size:**
```python
EEGNeX parameters: ~100K-200K
✅ Fits in 20 GB GPU easily
```

**Potential issue:**
- If we ensemble 5-10 models, need to load all in memory
- 10 models × 200K params = 2M params
- Still fits in 20 GB ✅

---

## ✅ Submission Requirements Checklist

### Required Files:
- [x] `submission.py` - Main inference code ✅
- [x] Model weights (.pth files) ✅
- [x] `load_model_path()` helper function ✅
- [ ] Extra dependencies (if needed) - Need to check base Docker image

### Base Docker Image:
**Image:** `sylvchev/codalab-eeg2025:v14`

**Likely includes:**
- Python 3.8+
- PyTorch
- MNE-Python
- NumPy, SciPy
- Basic ML libraries

**May need to package:**
- s3fs (if streaming during inference) ⚠️
- boto3 ⚠️
- Custom dependencies

**Note:** Inference happens on **evaluation server**, not S3 streaming!
- Evaluation data is provided locally on server
- We don't need S3 access during inference
- S3 streaming is only for TRAINING ✅

---

## 🎯 Challenge 2 Threshold Rule

**Important:** Challenge 2 only counts if:
- At least one team scores ≥ 0.990 on official evaluation
- Otherwise, only Challenge 1 counts

**Current SOTA:**
- Overall: 0.978
- C1: 0.928
- C2: 1.000

**Interpretation:**
- C2 threshold NOT met yet (need ≤ 0.990, current is 1.000)
- But C2 is still counting (70% of score)
- This is likely public leaderboard, final eval may differ

---

## 🔧 What We Need to Fix

### Priority 1: Get Real Behavioral Targets ❌ CRITICAL

**Current:** Using synthetic data
**Need:** Real response times and externalizing factors

**Options:**
1. Download phenotype CSV from HBN website
2. Find phenotype files on S3
3. Use competition-provided targets (check starter kit)

**Action:**
```bash
# Check starter kit on GitHub
# Look for phenotype data or behavioral targets
# This is CRITICAL for actual training
```

---

### Priority 2: Verify Task-Specific Data ⚠️

**Question:** Which EEG files correspond to CCD task?

**Possible answers:**
1. All HBN data includes CCD task (most likely)
2. Need to filter for specific task files
3. Competition provides subset

**Action:** Check starter kit documentation

---

### Priority 3: Test Inference Constraints ✅

**Requirements:**
- Single GPU with 20 GB memory
- Runs on evaluation server (not S3)
- Fast inference

**Our status:**
- Model fits easily ✅
- Inference code ready ✅
- No S3 dependency at inference ✅

---

## 📊 Corrected Strategy

### Training (Our S3 Streaming Approach) ✅ CORRECT

```bash
# This is fine for training:
python train.py \
  --data_path s3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_R1 \
  --use_streaming \
  --max_subjects 50
```

**But:** Need real behavioral targets, not synthetic!

---

### Inference (Submission) ✅ CORRECT

```python
# submission.py structure is correct:
class Submission:
    def __init__(self, SFREQ, DEVICE):
        self.model_path = load_model_path()  # ✅ Required
        # Load trained weights

    def get_model_challenge_1(self):
        # Return trained model ✅

    def get_model_challenge_2(self):
        # Return trained model ✅
```

**Server provides:** EEG data locally (not via S3)
**We provide:** Trained model weights + inference code

---

## 🎯 Immediate Actions Required

### 1. Find Real Behavioral Data ❌ BLOCKING

**Where to look:**
- HBN phenotype files on S3: `s3://fcp-indi/data/Projects/HBN/phenotype/`
- Competition starter kit on GitHub
- NEMAR.org download page

**What we need:**
- CSV with: subject_id, response_time, externalizing_factor
- Match with EEG subject IDs

---

### 2. Update Behavioral Streamer

**Current:**
```python
use_synthetic=True  # ⚠️ TESTING ONLY
```

**Need:**
```python
use_synthetic=False  # Use real phenotype data
phenotype_path="s3://fcp-indi/.../phenotype.csv"
```

---

### 3. Verify Starter Kit

**Check:** https://github.com/... (competition starter kit)

**Look for:**
- Sample code for loading targets
- Phenotype data location
- Task-specific filtering
- Baseline implementation

---

## ✅ What's Already Correct

1. ✅ S3 streaming architecture
2. ✅ Data format support (BDF/SET)
3. ✅ 100 Hz downsampling
4. ✅ Model architecture (EEGNeX)
5. ✅ Submission structure
6. ✅ Inference constraints (20 GB GPU)
7. ✅ Output format (1 value per challenge)

---

## 🚨 Summary

**Working:**
- S3 streaming infrastructure ✅
- Training pipeline ✅
- Model architecture ✅
- Submission format ✅

**Blocking:**
- Real behavioral targets ❌ CRITICAL
- Need actual response times
- Need actual externalizing factors

**Next steps:**
1. Find phenotype data location
2. Update `behavioral_streaming.py`
3. Test with real data
4. Then proceed with training plan

---

**Should I search for the HBN phenotype data location now?** 🔍
