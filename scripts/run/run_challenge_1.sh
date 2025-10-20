#!/bin/bash
# Training Script for Challenge 1 (Response Time Prediction)
# Challenge 1 contributes 30% to overall score

echo "🚀 Training Challenge 1 (Response Time)"
echo "Expected time: ~8-12 hours"
echo "Checkpoint will be saved to: ./checkpoints/c1_best.pth"
echo ""

python train.py \
  --challenge 1 \
  --data_path dummy \
  --use_official \
  --max_subjects 100 \
  --epochs 100 \
  --batch_size 32 \
  --lr 0.001 \
  --dropout 0.20 \
  --checkpoint_dir ./checkpoints \
  --save_every 10

echo ""
echo "✅ Challenge 1 training complete!"
echo "📁 Model saved to: ./checkpoints/c1_best.pth"
echo ""
echo "Next: Run Challenge 2 with: bash run_challenge_2.sh"
