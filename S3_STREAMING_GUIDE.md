# S3 Streaming Guide - TRUE Cloud Streaming!

## What This Is

**Pure cloud streaming from AWS S3** - NO local caching at all!

- Data streams directly from: `s3://nmdatasets/NeurIPS2025/`
- Each batch is loaded on-demand from S3
- Zero local disk usage (except model checkpoints)
- Slowest option but truly zero-cache

---

## Prerequisites

Install two extra packages:

```bash
pip install boto3 mne
```

- **boto3**: AWS SDK for Python (to access S3)
- **mne**: MNE-Python (to read BDF files)

---

## Quick Start

### Test with Mini Dataset (Recommended First)

```bash
# Quick test (5 files, 10 epochs)
python3 train_domain_adaptation_s3.py \
    --challenge c1 \
    --epochs 10 \
    --batch_size 16 \
    --mini \
    --max_files 5
```

This will:
- ✅ Stream 5 files from S3
- ✅ No local caching
- ✅ Takes ~5-10 minutes
- ✅ Verifies S3 streaming works

### Full Training

```bash
# Full training with S3 streaming
python3 train_domain_adaptation_s3.py \
    --challenge c1 \
    --epochs 100 \
    --batch_size 32 \
    --release R11

# Challenge 2
python3 train_domain_adaptation_s3.py \
    --challenge c2 \
    --epochs 100 \
    --batch_size 32 \
    --release R11
```

---

## All Three Methods Compared

| Method | Command | Caching | Speed | Disk Usage |
|--------|---------|---------|-------|------------|
| **S3 Streaming** | `train_domain_adaptation_s3.py` | ❌ None | 🐌 Slowest | 0 MB (except checkpoints) |
| **Direct (eegdash)** | `train_domain_adaptation_direct.py` | ✅ Local | 🚀 Fast | ~500 MB |
| **Preprocessed** | `train_domain_adaptation.py` | ✅ numpy | ⚡ Fastest | ~1 GB |

---

## When to Use S3 Streaming

### ✅ Use S3 Streaming If:
- Zero local disk space available
- Testing on small subset of data
- Running on serverless/ephemeral compute
- Want truly stateless training

### ❌ Don't Use S3 Streaming If:
- Training full dataset (will be very slow)
- Local disk space available
- Need fastest training time
- Running multiple experiments

---

## S3 Streaming Details

### What Happens:
1. Script lists files from `s3://nmdatasets/NeurIPS2025/`
2. For each batch:
   - Downloads BDF file to memory (BytesIO)
   - Reads with MNE
   - Extracts EEG data
   - Passes to model
   - Discards from memory
3. No files saved to disk

### Performance:
- **Download per file**: ~1-5 seconds
- **Batch processing**: Similar to other methods
- **Overall**: ~2-3x slower than cached methods

### Network Requirements:
- Stable internet connection
- ~100-200 MB/hour data transfer
- AWS S3 access (no credentials needed - public bucket)

---

## Troubleshooting

### Error: "boto3 not installed"
```bash
pip install boto3
```

### Error: "mne not installed"
```bash
pip install mne
```

### Error: "Access Denied" from S3
The bucket is public with `--no-sign-request`. If you see this:
- Check your internet connection
- Verify bucket name: `nmdatasets`
- Verify prefix: `NeurIPS2025/`

### Slow Performance
S3 streaming is inherently slower. For faster training:
- Use mini dataset: `--mini`
- Reduce files: `--max_files 100`
- Or switch to cached methods

---

## Example Workflow

### 1. Test S3 Streaming Works
```bash
python3 train_domain_adaptation_s3.py \
    --challenge c1 \
    --epochs 5 \
    --batch_size 8 \
    --mini \
    --max_files 10
```

### 2. If It Works, Run Full Training
```bash
# WARNING: This will be slow (~6-8 hours)
python3 train_domain_adaptation_s3.py \
    --challenge c1 \
    --epochs 100 \
    --batch_size 32
```

### 3. Or Switch to Cached Method
```bash
# Much faster (~2-3 hours)
python3 train_domain_adaptation_direct.py \
    --challenge c1 \
    --epochs 100 \
    --batch_size 64
```

---

## Recommendation

**For most users**: Use `train_domain_adaptation_direct.py`
- ✅ Reasonable caching (~500 MB)
- ✅ Much faster
- ✅ Same as `train_official.py` approach

**For zero-disk scenarios**: Use `train_domain_adaptation_s3.py`
- ✅ True streaming
- ✅ Zero disk usage
- ❌ 2-3x slower

---

## Architecture

```
AWS S3 Bucket
s3://nmdatasets/NeurIPS2025/
         ↓
    boto3 download
         ↓
    BytesIO (memory)
         ↓
    MNE read_raw_bdf
         ↓
    EEG data (numpy)
         ↓
    PyTorch tensor
         ↓
    Model training
         ↓
    Discard from memory
```

No local files created except checkpoints!

---

## Summary

**YES, true S3 streaming is possible!**

```bash
# Install dependencies
pip install boto3 mne

# Test it works
python3 train_domain_adaptation_s3.py --challenge c1 --epochs 5 --mini --max_files 5

# Full training (slow but zero-cache)
python3 train_domain_adaptation_s3.py --challenge c1 --epochs 100 --batch_size 32
```

But for most use cases, the cached version is better:
```bash
python3 train_domain_adaptation_direct.py --challenge c1 --epochs 100
```
