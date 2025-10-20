# Run Scripts Guide

All training scripts ready to use! 🚀

---

## 📋 Available Scripts

### 1. `run_test.sh` - Quick Test (⭐ START HERE!)
**What it does:** Verifies everything works (5 minutes)
**When to use:** First thing after setup
```bash
bash run_test.sh
```

### 2. `run_challenge_1.sh` - Train Challenge 1
**What it does:** Trains response time prediction model
**Time:** ~8-12 hours
**Output:** `checkpoints/c1_best.pth`
```bash
bash run_challenge_1.sh
```

### 3. `run_challenge_2.sh` - Train Challenge 2
**What it does:** Trains externalizing factor prediction model
**Time:** ~8-12 hours
**Output:** `checkpoints/c2_best.pth`
```bash
bash run_challenge_2.sh
```

### 4. `run_both_parallel.sh` - Train Both (if you have 2 GPUs)
**What it does:** Trains both challenges simultaneously
**Time:** ~8-12 hours total (not 16-24!)
**Output:** Both checkpoints
```bash
bash run_both_parallel.sh
```

### 5. `create_submission.sh` - Create Submission ZIP
**What it does:** Packages models into submission.zip
**Time:** <1 minute
**Output:** `[timestamp]_submission.zip`
```bash
bash create_submission.sh
```

### 6. `run_full_pipeline.sh` - Everything Automated
**What it does:** Test → Train C1 → Train C2 → Create submission
**Time:** ~24 hours
**Output:** Ready-to-submit ZIP
```bash
bash run_full_pipeline.sh
```

### 7. `run_large_scale.sh` - Maximum Performance
**What it does:** Trains on 200 subjects, 150 epochs each
**Time:** ~48-72 hours
**Output:** Best possible model
```bash
bash run_large_scale.sh
```

---

## 🎯 Recommended Workflow

### First Time (Day 1):

```bash
# 1. Quick test (5 min)
bash run_test.sh

# If test passes:

# 2. Train Challenge 1 (overnight)
bash run_challenge_1.sh

# 3. Train Challenge 2 (next night)
bash run_challenge_2.sh

# 4. Create submission
bash create_submission.sh

# 5. Upload to Codabench!
```

---

### If You Have 2 GPUs:

```bash
# 1. Test
bash run_test.sh

# 2. Train both at once (overnight)
bash run_both_parallel.sh

# 3. Create submission
bash create_submission.sh
```

---

### Fully Automated:

```bash
# Runs everything: test → train → submit
bash run_full_pipeline.sh

# Then upload the ZIP file!
```

---

## 📊 What Each Script Does

### run_test.sh
```bash
python train.py \
  --challenge 1 \
  --data_path dummy \
  --use_official \
  --official_mini \
  --max_subjects 5 \
  --epochs 3 \
  --batch_size 4
```

### run_challenge_1.sh
```bash
python train.py \
  --challenge 1 \
  --data_path dummy \
  --use_official \
  --max_subjects 100 \
  --epochs 100 \
  --batch_size 32 \
  --lr 0.001 \
  --dropout 0.20
```

### run_challenge_2.sh
```bash
python train.py \
  --challenge 2 \
  --data_path dummy \
  --use_official \
  --max_subjects 100 \
  --epochs 100 \
  --batch_size 32 \
  --lr 0.001 \
  --dropout 0.20
```

### create_submission.sh
```bash
python create_submission.py \
  --model_c1 checkpoints/c1_best.pth \
  --model_c2 checkpoints/c2_best.pth
```

---

## 🔧 Customization

### Change number of subjects:
Edit the script and change `--max_subjects 100` to desired number

### Change epochs:
Edit the script and change `--epochs 100` to desired number

### Change learning rate:
Edit the script and change `--lr 0.001` to desired value

### Change batch size (if out of memory):
Edit the script and change `--batch_size 32` to 16 or 8

---

## 📁 Output Files

After running scripts, you'll have:

```
BCI/
├── checkpoints/
│   ├── c1_best.pth          # Challenge 1 best model
│   ├── c2_best.pth          # Challenge 2 best model
│   ├── c1_epoch10.pth       # Checkpoint every 10 epochs
│   └── c2_epoch10.pth
│
├── logs/                     # If using run_both_parallel.sh
│   ├── challenge_1.log
│   └── challenge_2.log
│
└── 20251020_1234_submission.zip  # Ready to upload!
```

---

## 🚨 Troubleshooting

### "Permission denied"
```bash
chmod +x *.sh
```

### "CUDA out of memory"
Edit the script and reduce `--batch_size` to 16 or 8

### "Loss not decreasing"
- Check logs to verify data is loading
- Try lower learning rate: `--lr 0.0005`

### "Script stops unexpectedly"
Run in background:
```bash
nohup bash run_challenge_1.sh > train_c1.log 2>&1 &
tail -f train_c1.log  # Monitor progress
```

---

## 📊 Monitoring Training

### Check if training is running:
```bash
ps aux | grep python
nvidia-smi  # Check GPU usage
```

### Monitor progress:
```bash
# If using run_both_parallel.sh
tail -f logs/challenge_1.log
tail -f logs/challenge_2.log

# If running normally
# Just watch the terminal output
```

### Check checkpoints:
```bash
ls -lh checkpoints/
```

---

## ⏱️ Time Estimates

| Script | Time | When to Use |
|--------|------|-------------|
| run_test.sh | 5 min | First thing, verify setup |
| run_challenge_1.sh | 8-12 hours | Standard training |
| run_challenge_2.sh | 8-12 hours | Standard training |
| run_both_parallel.sh | 8-12 hours | If you have 2 GPUs |
| create_submission.sh | <1 min | After training |
| run_full_pipeline.sh | ~24 hours | Fully automated |
| run_large_scale.sh | 48-72 hours | Maximum performance |

---

## 🎯 Quick Reference

**Start here:**
```bash
bash run_test.sh
```

**Standard workflow:**
```bash
bash run_challenge_1.sh  # Overnight
bash run_challenge_2.sh  # Next night
bash create_submission.sh
```

**Fast workflow (2 GPUs):**
```bash
bash run_both_parallel.sh  # One night
bash create_submission.sh
```

**Lazy workflow:**
```bash
bash run_full_pipeline.sh  # Everything automated
```

---

**Ready to train!** Start with `bash run_test.sh` 🚀
