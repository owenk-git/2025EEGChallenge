#!/bin/bash
# Large Scale Training - More subjects for better performance

echo "🚀 Large Scale Training (200 subjects, 150 epochs)"
echo "Expected time: ~24-36 hours per challenge"
echo "⚠️  This will take a LONG time but should give better results!"
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
fi

# Challenge 1 - More subjects, more epochs
echo "Training Challenge 1..."
python train.py \
  --challenge 1 \
  --data_path dummy \
  --use_official \
  --max_subjects 200 \
  --epochs 150 \
  --batch_size 32 \
  --lr 0.001 \
  --dropout 0.20 \
  --checkpoint_dir ./checkpoints_large

# Challenge 2 - More subjects, more epochs
echo "Training Challenge 2..."
python train.py \
  --challenge 2 \
  --data_path dummy \
  --use_official \
  --max_subjects 200 \
  --epochs 150 \
  --batch_size 32 \
  --lr 0.001 \
  --dropout 0.20 \
  --checkpoint_dir ./checkpoints_large

echo ""
echo "✅ Large scale training complete!"
echo "Create submission:"
python create_submission.py \
  --model_c1 checkpoints_large/c1_best.pth \
  --model_c2 checkpoints_large/c2_best.pth \
  --output large_scale_submission.zip
