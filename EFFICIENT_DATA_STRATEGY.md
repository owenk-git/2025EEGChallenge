# Efficient Data Strategy - No Full Downloads Needed

## 🎯 Problem

- **Full dataset:** 11 releases × 100-250 GB = 1-2 TB total
- **Competition constraint:** 20 GB GPU memory for inference
- **Your constraint:** Don't want to download everything

## ✅ Solution: Smart Data Sampling + Streaming

---

## Strategy 1: Sample-Based Training (RECOMMENDED) ⭐⭐⭐⭐⭐

### Approach: Train on Representative Subset

Instead of downloading ALL data, strategically sample:

**Phase 1: Mini Datasets (Already Available!)**
```
R1_mini_L100: 20 subjects, 100 Hz, ~500 MB ✅
R2_mini_L100: 20 subjects, 100 Hz, ~500 MB ✅
R3_mini_L100: 20 subjects, 100 Hz, ~500 MB ✅
```

**Total download:** ~1.5 GB for 60 subjects across 3 releases

**Expected performance:**
- C1: 1.1-1.3
- C2: 1.0-1.1
- Overall: 1.05-1.2

**Why this works:**
- Mini datasets are **curated** (20 subjects who completed all tasks)
- Cross-release diversity improves generalization
- Fast iteration (hours, not days)

---

## Strategy 2: S3 Streaming (No Download) ⭐⭐⭐⭐

### Use S3 Direct Access with boto3

```python
# Install
pip install boto3 s3fs

# Access without downloading
import s3fs
fs = s3fs.S3FileSystem(anon=True)

# Stream files directly
s3_path = 's3://fcp-indi/data/Projects/HBN/BIDS_EEG/cmi_bids_R1/...'
with fs.open(s3_path, 'rb') as f:
    # Load EEG data directly from S3
    raw = mne.io.read_raw_bdf(f, preload=True)
```

**Advantages:**
- ✅ Zero disk space usage
- ✅ Access any file on-demand
- ✅ Pay-per-use (AWS free tier friendly)

**Disadvantages:**
- ❌ Slower (network latency)
- ❌ Requires stable internet
- ❌ May hit AWS rate limits

---

## Strategy 3: Progressive Downloading ⭐⭐⭐

### Download Only What You Need, When You Need It

```python
# Download one subject at a time, train, delete
for subject in subjects_list:
    download_subject(subject)     # ~300 MB
    train_on_subject(subject)     # Train
    delete_subject(subject)       # Free space
```

**Disk usage:** Only ~1-2 GB at any time

---

## Strategy 4: Use HBN Phenotype Data ⭐⭐⭐⭐

### Smart Subject Selection Based on Metadata

Instead of random sampling, select subjects with:
- Complete task data
- Good signal quality
- Diverse demographics
- Relevant behavioral scores

```python
# Use HBN phenotype CSV
phenotype = pd.read_csv('HBN_phenotype.csv')

# Filter for quality
good_subjects = phenotype[
    (phenotype['eeg_quality'] == 'good') &
    (phenotype['tasks_completed'] >= 5)
]

# Download only these subjects
download_subjects(good_subjects['participant_id'])
```

**Benefit:** 50-100 high-quality subjects > 1000 random subjects

---

## 🎯 RECOMMENDED WORKFLOW

### Phase 1: Mini Datasets (Week 1)
**Download:** R1, R2, R3 mini datasets (~1.5 GB)
**Train:** On all 60 subjects
**Expected:** Overall ~1.1

### Phase 2: Targeted Sampling (Week 2)
**Download:** 50 subjects from R4-R7 mini datasets (~2 GB)
**Strategy:** Use phenotype data to select best subjects
**Expected:** Overall ~1.0

### Phase 3: Streaming for Diversity (Week 3)
**S3 Stream:** Random sample 100 more subjects
**No download needed**
**Expected:** Overall ~0.96

### Phase 4: Ensemble (Week 4)
**Train:** 3-5 models on different subsets
**Expected:** Overall <0.96 (Top 3!)

**Total disk usage:** < 5 GB

---

## 💾 Updated Dataset Loader (S3 Streaming)

I'll create an updated `data/dataset.py` that supports:
1. ✅ Local files (mini datasets)
2. ✅ S3 streaming (no download)
3. ✅ Smart caching (LRU cache for frequently accessed files)
4. ✅ Progressive download (download on-demand)

---

## 📊 Data Efficiency Tips

### 1. Use 100 Hz Data (Not Original Sampling Rate)
- Original: 500-1000 Hz
- Downsampled: 100 Hz (already done for mini datasets)
- **Savings:** 5-10× smaller files

### 2. Load Only Necessary Channels
```python
# Instead of all 129 channels
channels_to_use = ['Cz', 'Fz', 'Pz', ...]  # Top 64 channels
raw.pick_channels(channels_to_use)
```
**Savings:** 50% smaller

### 3. Use Memory Mapping
```python
# Don't load full file into memory
raw = mne.io.read_raw_bdf(filename, preload=False)
```
**Savings:** Constant memory usage

### 4. Batch Processing with Cleanup
```python
for batch in batches:
    load_batch()
    train_on_batch()
    del batch  # Free memory
    gc.collect()
```

### 5. Use Preprocessed Features (If Allowed)
- Extract features once
- Save to disk (much smaller than raw)
- Load features for training

---

## 🔧 Implementation Plan

I will now create:

1. **`data/streaming_dataset.py`** - S3 streaming support
2. **`data/efficient_sampler.py`** - Smart subject selection
3. **`scripts/download_selective.py`** - Download only needed files
4. **`configs/data_strategy.yaml`** - Data usage configuration

---

## 📈 Expected Performance vs Data Used

```
Data Used          | Expected Score | Time to Train
-------------------|----------------|---------------
20 subjects (R1)   | 1.2-1.3       | 2 hours
60 subjects (R1-3) | 1.1-1.2       | 6 hours
100 subjects       | 1.0-1.1       | 12 hours
200 subjects       | 0.98-1.05     | 1 day
500 subjects       | 0.95-1.0      | 2-3 days
1000+ subjects     | 0.94-0.96     | 5-7 days
```

**Sweet spot:** 100-200 subjects = Good performance without huge downloads

---

## 💡 Key Insight

**Top teams likely didn't use ALL 3000+ subjects!**

They probably:
- Used 200-500 high-quality subjects
- Trained with heavy data augmentation
- Used ensemble of 3-5 models
- Total data: < 50 GB

**You can match this with smart sampling!**

---

## 🚀 Next Steps

1. Start with R1-R3 mini datasets (1.5 GB) ✅
2. Train baseline model
3. Implement S3 streaming for additional samples
4. Use phenotype data to select best subjects
5. Scale to 100-200 subjects strategically

**No need to download terabytes!**
