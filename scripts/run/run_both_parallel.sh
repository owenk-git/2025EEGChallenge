#!/bin/bash
# Train Both Challenges in Parallel (if you have 2 GPUs or multi-GPU)

echo "🚀 Training Both Challenges in Parallel"
echo "⚠️  Make sure you have enough GPU memory!"
echo ""

# Challenge 1 on GPU 0
CUDA_VISIBLE_DEVICES=0 python train.py \
  --challenge 1 \
  --data_path dummy \
  --use_official \
  --max_subjects 100 \
  --epochs 100 \
  --batch_size 32 \
  --lr 0.001 \
  --checkpoint_dir ./checkpoints \
  > logs/challenge_1.log 2>&1 &

PID1=$!

# Challenge 2 on GPU 1 (or same GPU if you have enough memory)
CUDA_VISIBLE_DEVICES=1 python train.py \
  --challenge 2 \
  --data_path dummy \
  --use_official \
  --max_subjects 100 \
  --epochs 100 \
  --batch_size 32 \
  --lr 0.001 \
  --checkpoint_dir ./checkpoints \
  > logs/challenge_2.log 2>&1 &

PID2=$!

echo "Training started!"
echo "Challenge 1 PID: $PID1 (log: logs/challenge_1.log)"
echo "Challenge 2 PID: $PID2 (log: logs/challenge_2.log)"
echo ""
echo "Monitor progress:"
echo "  tail -f logs/challenge_1.log"
echo "  tail -f logs/challenge_2.log"
echo ""
echo "Waiting for both to complete..."

# Wait for both processes
wait $PID1
wait $PID2

echo ""
echo "✅ Both challenges complete!"
echo "Next: bash create_submission.sh"
