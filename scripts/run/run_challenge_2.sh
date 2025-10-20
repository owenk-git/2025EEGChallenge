#!/bin/bash
# Training Script for Challenge 2 (Externalizing Factor Prediction)
# Challenge 2 contributes 70% to overall score (MORE IMPORTANT!)

echo "🚀 Training Challenge 2 (Externalizing Factor)"
echo "Expected time: ~8-12 hours"
echo "Checkpoint will be saved to: ./checkpoints/c2_best.pth"
echo ""

python train.py \
  --challenge 2 \
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
echo "✅ Challenge 2 training complete!"
echo "📁 Model saved to: ./checkpoints/c2_best.pth"
echo ""
echo "Next: Create submission with: bash create_submission.sh"
