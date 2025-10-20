#!/bin/bash
# Quick Test Script - Verify everything works (5 minutes)

echo "🧪 Quick Test: Verifying setup and data loading..."
echo "Expected time: ~5 minutes"
echo ""

python train.py \
  --challenge 1 \
  --data_path dummy \
  --use_official \
  --official_mini \
  --max_subjects 5 \
  --epochs 3 \
  --batch_size 4 \
  --lr 0.001

echo ""
echo "✅ If test passed, run: bash run_challenge_1.sh"
