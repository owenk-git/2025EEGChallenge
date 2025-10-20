# 🚀 Run with S3 Streaming (NO DOWNLOAD)

## ⚡ Quick Test (5 minutes)

```bash
python train.py \
  -c 1 \
  -d s3://nmdatasets/NeurIPS2025/R1_mini_L100_bdf \
  -s \
  --max 5 \
  -e 3 \
  --num 999
```

**What this does:**
- Streams data directly from S3 (no download!)
- Uses 5 subjects, 3 epochs
- Fast test to verify everything works

---

## 🎯 Baseline Experiments (S3 Streaming)

### Experiment 1: Challenge 1 (~1 hour)
```bash
python train.py \
  -c 1 \
  -d s3://nmdatasets/NeurIPS2025/R1_mini_L100_bdf \
  -s \
  --max 50 \
  -e 50 \
  --drop 0.2 \
  --lr 1e-3 \
  -b 32 \
  --num 1
```

### Experiment 2: Challenge 2 (~1 hour)
```bash
python train.py \
  -c 2 \
  -d s3://nmdatasets/NeurIPS2025/R1_mini_L100_bdf \
  -s \
  --max 50 \
  -e 50 \
  --drop 0.2 \
  --lr 1e-3 \
  -b 32 \
  --num 2
```

---

## 🔬 All Explorations (S3 Streaming)

```bash
# Run all 10 experiments with S3 streaming
./scripts/run_exploration_streaming.sh

# Or specify GPU:
./scripts/run_exploration_streaming.sh 1
```

---

## 📋 Key Flags for S3 Streaming

| Flag | Meaning | Example |
|------|---------|---------|
| `-c` | Challenge (1 or 2) | `-c 1` |
| `-d` | S3 data path | `-d s3://nmdatasets/NeurIPS2025/R1_mini_L100_bdf` |
| `-s` | Enable S3 streaming | `-s` |
| `--max` | Max subjects | `--max 50` |
| `-e` | Epochs | `-e 50` |
| `--drop` | Dropout rate | `--drop 0.2` |
| `--lr` | Learning rate | `--lr 1e-3` |
| `-b` | Batch size | `-b 32` |
| `--num` | Experiment number | `--num 1` |

**DO NOT USE:**
- ❌ `-o` or `--use_official` - This downloads!
- ❌ `-m` or `--official_mini` - Only for official dataset

---

## ⚠️ Important Notes

### S3 Streaming vs Official Dataset

**S3 Streaming (`-s`):**
- ✅ No download
- ✅ Saves disk space
- ❌ Slower (S3 latency each batch)
- ❌ Needs stable internet

**Official Dataset (`-o`):**
- ❌ Downloads once (~10-30 min)
- ✅ Much faster training (local disk)
- ✅ No internet needed after download
- ✅ Cached for future runs

### Performance Impact

With S3 streaming:
- First epoch: ~2-3 minutes
- Subsequent epochs: ~2-3 minutes each
- 50 epochs: ~2 hours

With cached official dataset:
- First epoch: ~2-3 minutes
- Subsequent epochs: ~1 minute each
- 50 epochs: ~1 hour

**S3 streaming is ~2x slower but requires no download.**

---

## 🚀 Your Next Command

**Start with quick test:**
```bash
python train.py -c 1 -d s3://nmdatasets/NeurIPS2025/R1_mini_L100_bdf -s --max 5 -e 3 --num 999
```

**Then run baselines:**
```bash
python train.py -c 1 -d s3://nmdatasets/NeurIPS2025/R1_mini_L100_bdf -s --max 50 -e 50 --num 1
python train.py -c 2 -d s3://nmdatasets/NeurIPS2025/R1_mini_L100_bdf -s --max 50 -e 50 --num 2
```

**No downloads! Pure streaming! 🚀**
