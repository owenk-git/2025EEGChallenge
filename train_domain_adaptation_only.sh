#!/bin/bash

# Quick script to train ONLY Domain Adaptation model
# This is the most promising model and works with direct loading!

echo "================================"
echo "Training Domain Adaptation EEGNeX"
echo "No preprocessing needed!"
echo "================================"

# Set device
DEVICE="cuda"
EPOCHS=100
BATCH_SIZE=64

# Create checkpoints directory
mkdir -p checkpoints

echo ""
echo "Training Challenge 1..."
python3 train_domain_adaptation_direct.py \
    --challenge c1 \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr 1e-3 \
    --lambda_mmd 0.1 \
    --lambda_entropy 0.01 \
    --lambda_adv 0.1 \
    --device $DEVICE \
    --save_dir checkpoints

echo ""
echo "Training Challenge 2..."
python3 train_domain_adaptation_direct.py \
    --challenge c2 \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr 1e-3 \
    --lambda_mmd 0.1 \
    --lambda_entropy 0.01 \
    --lambda_adv 0.1 \
    --device $DEVICE \
    --save_dir checkpoints

echo ""
echo "Creating submission..."
python3 create_advanced_submission.py \
    --model domain_adaptation \
    --name domain_adaptation_v1 \
    --checkpoint_c1 checkpoints/domain_adaptation_c1_best.pt \
    --checkpoint_c2 checkpoints/domain_adaptation_c2_best.pt \
    --output_dir submissions

echo ""
echo "================================"
echo "Domain Adaptation Training Complete!"
echo "================================"
echo "Submission created: submissions/domain_adaptation_v1_submission.zip"
echo ""
echo "Expected performance:"
echo "  C1: 1.15-1.20 (vs current 1.31)"
echo "  C2: 1.00-1.05 (vs current 1.00)"
echo "  Overall: 1.05-1.10 (vs current 1.09)"
echo ""
echo "Ready to submit to competition!"
