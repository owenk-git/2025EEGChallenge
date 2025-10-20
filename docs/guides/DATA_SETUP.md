# Data Loading Setup - ALL RELEASES

## ✅ Default Configuration

**By default, the training pipeline now streams ALL competition data from S3:**

- **Total subjects**: 3,387
- **All releases**: R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, NC (Neurotypical Control)
- **No download required**: Streams directly from S3
- **BIDS format**: Uses official `EEGChallengeDataset` from `eegdash`

## 📊 All Available Releases

According to the competition website, here are ALL the releases included:

| Release | Description | Subjects | S3 URI |
|---------|-------------|----------|--------|
| R1 | Initial release | 323 | s3://nmdatasets/NeurIPS2025/R1 |
| R2 | Second release | 277 | s3://nmdatasets/NeurIPS2025/R2 |
| R3 | Third release | 364 | s3://nmdatasets/NeurIPS2025/R3 |
| R4 | Fourth release | 284 | s3://nmdatasets/NeurIPS2025/R4 |
| R5 | Fifth release | 306 | s3://nmdatasets/NeurIPS2025/R5 |
| R6 | Sixth release | 331 | s3://nmdatasets/NeurIPS2025/R6 |
| R7 | Seventh release | 249 | s3://nmdatasets/NeurIPS2025/R7 |
| R8 | Eighth release | 262 | s3://nmdatasets/NeurIPS2025/R8 |
| R9 | Ninth release | 239 | s3://nmdatasets/NeurIPS2025/R9 |
| R10 | Tenth release | 254 | s3://nmdatasets/NeurIPS2025/R10 |
| R11 | Eleventh release | 269 | s3://nmdatasets/NeurIPS2025/R11 |
| NC | Neurotypical Control | 229 | s3://nmdatasets/NeurIPS2025/NC |
| **TOTAL** | **All releases** | **3,387** | **All above** |

## 🚀 Quick Start

### 1. Quick Test (Mini Dataset)
Use mini dataset to test the pipeline (faster, small subset):

```bash
python train.py -c 1 -o -m --max 5 -e 3
```

- `-m` / `--official_mini`: Use mini subset for testing
- Will load small subset of data for quick validation

### 2. Full Training (ALL 3,387 Subjects)
Train on the complete competition dataset:

```bash
python train.py -c 1 -o --max 100 -e 50
```

**What happens**:
- `-o` / `--use_official`: Use official EEGChallengeDataset
- NO `-m` flag: Uses FULL dataset (not mini)
- `release="all"` (default): Loads ALL releases (R1-R11 + NC)
- Total: ~3,387 subjects across all releases
- Streams from S3 (no download!)

### 3. Full Training with Unlimited Subjects
Remove subject limit to use ALL available subjects:

```bash
python train.py -c 1 -o -e 100
```

**Important**: Don't specify `--max` to use all 3,387 subjects

## 📦 Data Loading Modes

### Mode 1: Official Dataset (RECOMMENDED ✅)

```bash
python train.py -c 1 -o --max 100 -e 50
```

**Pros**:
- Streams ALL releases automatically (R1-R11 + NC)
- Behavioral targets automatically loaded
- BIDS format handled correctly
- Official competition data loader
- No manual path configuration needed

**Cons**:
- Requires `eegdash` package: `pip install eegdash braindecode`

### Mode 2: Custom S3 Streaming (OLD - Deprecated)

```bash
python train.py -c 1 -d s3://nmdatasets/NeurIPS2025/R1_mini_L100_bdf -s --max 50 -e 50
```

**Issues**:
- Only loads ONE release at a time (not all R1-R11 + NC)
- Need to manually specify S3 path
- Behavioral targets need manual parsing
- More complex setup

**Use official mode instead!**

### Mode 3: Local Dataset (NOT RECOMMENDED)

```bash
python train.py -c 1 -d ./data/R1 -e 50
```

**Issues**:
- Requires downloading data locally (huge download!)
- No streaming
- Takes up disk space
- Only one release at a time

**Use official mode for S3 streaming instead!**

## 🔍 Verifying Data Loading

When you run training, you should see output like this:

```
📦 Loading EEGChallengeDataset
   Task: contrastChangeDetection
   Release: all (ALL RELEASES - 3,387 subjects)
   Mini: False 🌐 (FULL dataset)
✅ Loaded 67231 recordings
   Unique subjects: 3387
   Expected ~3387 subjects (full competition dataset)
```

**Key checks**:
1. ✅ "Release: all (ALL RELEASES)"
2. ✅ "Mini: False 🌐 (FULL dataset)"
3. ✅ "Unique subjects: 3387" (or close to it)

If you see different numbers, check your flags!

## ⚠️ Common Issues

### Issue 1: Only Loading Mini Dataset

**Symptom**: "Unique subjects: 10" (very small number)

**Cause**: Using `-m` flag

**Fix**: Remove `-m` flag to use full dataset

```bash
# ❌ Wrong (mini):
python train.py -c 1 -o -m -e 50

# ✅ Correct (full):
python train.py -c 1 -o -e 50
```

### Issue 2: Downloading Instead of Streaming

**Symptom**: See "Downloading..." messages

**Cause**: Official dataset caches metadata (but NOT raw data)

**Explanation**:
- Small metadata files (~MB) are cached locally
- Raw EEG data (~GB) is streamed from S3
- This is normal and expected behavior

### Issue 3: Only One Release

**Symptom**: "Unique subjects: 323" (just one release)

**Cause**: Using custom S3 streaming with single release path

**Fix**: Use official mode (`-o`) which loads all releases automatically

```bash
# ❌ Wrong (one release):
python train.py -c 1 -d s3://nmdatasets/NeurIPS2025/R1 -s -e 50

# ✅ Correct (all releases):
python train.py -c 1 -o -e 50
```

## 📝 Complete Training Commands

### Quick Test (5 minutes)
```bash
python train.py -c 1 -o -m --max 5 -e 3
```

### Medium Training (2 hours)
```bash
python train.py -c 1 -o --max 100 -e 50
```

### Full Training (12-24 hours)
```bash
python train.py -c 1 -o -e 100
```

All commands automatically:
- ✅ Stream from S3 (no download)
- ✅ Load ALL releases (R1-R11 + NC)
- ✅ Load behavioral targets
- ✅ Use subject-wise validation split (prevents data leakage)
- ✅ Save best model based on validation NRMSE

## 🎯 Summary

**Default setup now uses ALL data:**

- **Command**: `python train.py -c 1 -o -e 50`
- **Subjects**: 3,387 (all releases)
- **Streaming**: Yes (from S3)
- **Download**: Only small metadata cache (~MB)
- **Releases**: R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, NC

**To train on full dataset, just run**:

```bash
python train.py -c 1 -o -e 100
```

That's it! The official dataset loader handles everything automatically.
