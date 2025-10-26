#!/bin/bash

# Master script to train all 3 advanced models
# 1. Domain Adaptation EEGNeX
# 2. Cross-Task Pre-Training
# 3. Hybrid CNN-Transformer-DA

echo "================================"
echo "Training All Advanced Models"
echo "================================"

# Set device
DEVICE="cuda"
EPOCHS=100
BATCH_SIZE=64

# Create checkpoints directory
mkdir -p checkpoints

echo ""
echo "================================"
echo "Model 1: Domain Adaptation EEGNeX"
echo "================================"
echo "Training C1..."
python3 train_domain_adaptation.py \
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
echo "Training C2..."
python3 train_domain_adaptation.py \
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
echo "Model 2: Cross-Task Pre-Training"
echo "================================"
echo "Training C1..."
python3 train_cross_task.py \
    --challenge c1 \
    --epochs $EPOCHS \
    --pretrain_epochs 50 \
    --batch_size $BATCH_SIZE \
    --lr 1e-3 \
    --device $DEVICE \
    --save_dir checkpoints

echo ""
echo "Training C2..."
python3 train_cross_task.py \
    --challenge c2 \
    --epochs $EPOCHS \
    --pretrain_epochs 50 \
    --batch_size $BATCH_SIZE \
    --lr 1e-3 \
    --device $DEVICE \
    --save_dir checkpoints

echo ""
echo "Creating submission..."
python3 create_advanced_submission.py \
    --model cross_task \
    --name cross_task_pretrain_v1 \
    --checkpoint_c1 checkpoints/cross_task_c1_best.pt \
    --checkpoint_c2 checkpoints/cross_task_c2_best.pt \
    --output_dir submissions

echo ""
echo "================================"
echo "Model 3: Hybrid CNN-Transformer-DA"
echo "================================"
echo "Training C1..."
python3 train_hybrid.py \
    --challenge c1 \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr 1e-3 \
    --lambda_mmd 0.1 \
    --lambda_entropy 0.01 \
    --device $DEVICE \
    --save_dir checkpoints

echo ""
echo "Training C2..."
python3 train_hybrid.py \
    --challenge c2 \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr 1e-3 \
    --lambda_mmd 0.1 \
    --lambda_entropy 0.01 \
    --device $DEVICE \
    --save_dir checkpoints

echo ""
echo "Creating submission..."
python3 create_advanced_submission.py \
    --model hybrid \
    --name hybrid_cnn_transformer_v1 \
    --checkpoint_c1 checkpoints/hybrid_c1_best.pt \
    --checkpoint_c2 checkpoints/hybrid_c2_best.pt \
    --output_dir submissions

echo ""
echo "================================"
echo "All Models Trained!"
echo "================================"
echo "Submissions created:"
echo "  1. submissions/domain_adaptation_v1_submission.zip"
echo "  2. submissions/cross_task_pretrain_v1_submission.zip"
echo "  3. submissions/hybrid_cnn_transformer_v1_submission.zip"
echo ""
echo "Ready to submit to competition!"
