# All Data Streaming - Summary & Answers

## ✅ Your Questions Answered

### Q1: Can you default setup `-d s3://...` S3 link?

**Answer**: YES! Even better - **you don't need to specify ANY S3 path now**.

Just use:
```bash
python train.py -c 1 -o -e 100
```

The `-o` flag uses the official `EEGChallengeDataset` which automatically:
- Streams from S3 (no download)
- Loads ALL releases (R1-R11 + NC)
- Handles 3,387 subjects
- Loads behavioral targets

**No manual S3 path needed!**

---

### Q2: Is it correctly streaming ALL data, or just one dataset?

**Answer**: With the NEW setup, it streams **ALL data** automatically!

**Before (OLD - Wrong ❌)**:
- Custom S3 streaming: Only loaded ONE release at a time
- `s3://nmdatasets/NeurIPS2025/R1_mini_L100_bdf` = Just R1 mini (tiny subset)
- Had to manually specify each release

**Now (NEW - Correct ✅)**:
- Official dataset with `release="all"` (default)
- Automatically loads ALL 12 releases: R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, NC
- Total: 3,387 subjects
- All behavioral targets included

---

## 📊 What Changed

### File: `data/official_dataset_example.py`

**Changed defaults**:
```python
# OLD ❌
release="R5"      # Only one release
mini=True         # Mini subset only

# NEW ✅
release="all"     # ALL releases (R1-R11 + NC)
mini=False        # Full dataset by default
```

**What this means**:
- Default behavior now loads ALL competition data
- Use `-m` flag only for quick testing
- No `-m` flag = full 3,387 subjects

---

## 🚀 Commands Comparison

### Quick Test (Mini - for testing pipeline only)
```bash
python train.py -c 1 -o -m --max 5 -e 3
```
**Loads**: ~10-20 subjects from mini subset
**Time**: 5 minutes
**Purpose**: Test that pipeline works

### Medium Training (Subset of full data)
```bash
python train.py -c 1 -o --max 100 -e 50
```
**Loads**: 100 subjects from full dataset (3,387 available)
**Time**: 2 hours
**Purpose**: Quick baseline with decent amount of data

### Full Training (ALL DATA ⭐)
```bash
python train.py -c 1 -o -e 100
```
**Loads**: ALL 3,387 subjects from R1-R11 + NC
**Time**: 12-24 hours
**Purpose**: Train on complete competition dataset

---

## 📦 Data Sources Explained

### The Competition Has Multiple Releases

```
📊 HBN-EEG Competition Dataset (Total: 3,387 subjects)

├── R1  (323 subjects)   - s3://nmdatasets/NeurIPS2025/R1
├── R2  (277 subjects)   - s3://nmdatasets/NeurIPS2025/R2
├── R3  (364 subjects)   - s3://nmdatasets/NeurIPS2025/R3
├── R4  (284 subjects)   - s3://nmdatasets/NeurIPS2025/R4
├── R5  (306 subjects)   - s3://nmdatasets/NeurIPS2025/R5
├── R6  (331 subjects)   - s3://nmdatasets/NeurIPS2025/R6
├── R7  (249 subjects)   - s3://nmdatasets/NeurIPS2025/R7
├── R8  (262 subjects)   - s3://nmdatasets/NeurIPS2025/R8
├── R9  (239 subjects)   - s3://nmdatasets/NeurIPS2025/R9
├── R10 (254 subjects)   - s3://nmdatasets/NeurIPS2025/R10
├── R11 (269 subjects)   - s3://nmdatasets/NeurIPS2025/R11
└── NC  (229 subjects)   - s3://nmdatasets/NeurIPS2025/NC
```

### Mini vs Full Dataset

**Mini Dataset** (`R1_mini_L100_bdf`):
- Small subset for testing
- ~10-20 subjects
- Good for pipeline validation
- NOT for final training

**Full Dataset** (R1-R11 + NC):
- Complete competition data
- 3,387 subjects
- What you need for good performance
- What top teams use

---

## ✅ Verification Checklist

When you run training, check the output:

```
📦 Loading EEGChallengeDataset
   Task: contrastChangeDetection
   Release: all (ALL RELEASES - 3,387 subjects)    ← ✅ Check this!
   Mini: False 🌐 (FULL dataset)                   ← ✅ Check this!
✅ Loaded 67231 recordings
   Unique subjects: 3387                            ← ✅ Check this!
   Expected ~3387 subjects (full competition dataset)
```

**Red flags** (wrong config):
- ❌ "Mini: True" = Only loading mini subset
- ❌ "Unique subjects: 10" = Not enough data
- ❌ "Release: R5" = Only one release

**Good signs** (correct config):
- ✅ "Release: all (ALL RELEASES - 3,387 subjects)"
- ✅ "Mini: False 🌐 (FULL dataset)"
- ✅ "Unique subjects: 3387"

---

## 🎯 Recommended Training Strategy

### Phase 1: Quick Test (5 minutes)
```bash
python train.py -c 1 -o -m --max 5 -e 3
```
**Goal**: Verify pipeline works
**Data**: Mini subset (~10 subjects)

### Phase 2: Baseline (2 hours)
```bash
python train.py -c 1 -o --max 100 -e 50
python train.py -c 2 -o --max 100 -e 50
```
**Goal**: Get first working submission
**Data**: 100 subjects from full dataset

### Phase 3: Full Training (12-24 hours)
```bash
python train.py -c 1 -o -e 100
python train.py -c 2 -o -e 100
```
**Goal**: Use ALL data for best performance
**Data**: ALL 3,387 subjects

### Phase 4: Create Submission
```bash
python create_submission.py \
  --model_c1 checkpoints/c1_best.pth \
  --model_c2 checkpoints/c2_best.pth
```

### Phase 5: Submit to Codabench
Upload the `.zip` file to: https://www.codabench.org/competitions/9975/

---

## 🔍 Technical Details

### How S3 Streaming Works

The official `EEGChallengeDataset` from `eegdash`:
1. Automatically discovers ALL releases from S3
2. Uses `s3fs` to stream data (no download)
3. Caches small metadata files locally (~MB)
4. Streams raw EEG data on-demand from S3 (~GB)
5. Parses BIDS format automatically
6. Loads behavioral targets from `participants.tsv`

### What Gets Cached vs Streamed

**Cached locally** (small, fast):
- `participants.tsv` (behavioral targets)
- Channel names
- Sampling rates
- Subject IDs
- Total: ~10-50 MB

**Streamed from S3** (large, on-demand):
- Raw EEG recordings (.bdf files)
- Only loaded when needed for training
- Total: ~100+ GB (but you never download it all)

### Why This is Better

**Old custom S3 streaming** (deprecated):
- Manual path specification: `s3://nmdatasets/NeurIPS2025/R1_mini_L100_bdf`
- Only one release at a time
- Manual BIDS parsing
- Manual behavioral target loading
- Prone to errors

**New official dataset** (recommended):
- Automatic: Just use `-o` flag
- All releases automatically
- BIDS parsing handled
- Behavioral targets auto-loaded
- Maintained by competition organizers

---

## 📝 Summary

### Your Original Concern
> "I need to use all data... I need to stream all data below [lists R1-R11 + NC releases]"

### The Solution
**You don't need to manually specify each release!**

The official dataset loader automatically handles ALL releases when you use:
```bash
python train.py -c 1 -o -e 100
```

### Key Points
1. ✅ Default now loads ALL 3,387 subjects (R1-R11 + NC)
2. ✅ Streams from S3 (no manual download)
3. ✅ Behavioral targets auto-loaded
4. ✅ Subject-wise validation split (prevents data leakage)
5. ✅ No manual S3 path configuration needed
6. ✅ Just use `-o` flag and you're done!

### Files Updated
- ✅ `data/official_dataset_example.py` - Changed defaults to `release="all"`, `mini=False`
- ✅ `README.md` - Updated commands and explanations
- ✅ `DATA_SETUP.md` - New comprehensive data guide (this file's sibling)

### Next Steps
1. Run quick test: `python train.py -c 1 -o -m --max 5 -e 3` (5 min)
2. Run full training: `python train.py -c 1 -o -e 100` (12-24 hours)
3. Create submission
4. Submit to Codabench
5. Beat that 1.14 score! 🚀

---

**You're now set up to train on ALL the competition data with a single simple command!** 🎉
