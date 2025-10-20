#!/bin/bash

# Run all exploration experiments with S3 STREAMING (NO DOWNLOAD)
# This version uses S3 streaming instead of official dataset

set -e

GPU=${1:-0}
S3_PATH="s3://nmdatasets/NeurIPS2025/R1_mini_L100_bdf"

echo "=========================================="
echo "🔬 EXPLORATION PHASE: S3 Streaming (No Download)"
echo "=========================================="
echo "GPU: $GPU"
echo "S3 Path: $S3_PATH"
echo "Started: $(date)"
echo ""

# Function to run experiment with streaming
run_exp() {
    local num=$1
    local challenge=$2
    local subjects=$3
    local epochs=$4
    local dropout=$5
    local lr=$6
    local batch=$7
    local desc=$8

    echo ""
    echo "=========================================="
    echo "🧪 Experiment $num: $desc"
    echo "=========================================="
    echo "Challenge: $challenge | Subjects: $subjects | Epochs: $epochs"
    echo "Dropout: $dropout | LR: $lr | Batch: $batch"
    echo "Started: $(date)"

    CUDA_VISIBLE_DEVICES=$GPU python train.py \
        -c $challenge \
        -d $S3_PATH \
        -s \
        --max $subjects \
        -e $epochs \
        --drop $dropout \
        --lr $lr \
        -b $batch \
        --num $num

    echo "✅ Experiment $num complete!"
    echo "Finished: $(date)"
}

# ===========================================
# Exp 1-2: BASELINE (Small & Fast)
# ===========================================
echo "📌 Group 1: BASELINE (50 subjects, 50 epochs)"
run_exp 1 1 50 50 0.2 1e-3 32 "C1 Baseline (S3 Streaming)"
run_exp 2 2 50 50 0.2 1e-3 32 "C2 Baseline (S3 Streaming)"

# ===========================================
# Exp 3-4: MORE DATA
# ===========================================
echo ""
echo "📌 Group 2: MORE DATA (200 subjects, 100 epochs)"
run_exp 3 1 200 100 0.2 1e-3 32 "C1 More Data"
run_exp 4 2 200 100 0.2 1e-3 32 "C2 More Data"

# ===========================================
# Exp 5-6: HIGHER DROPOUT
# ===========================================
echo ""
echo "📌 Group 3: HIGHER DROPOUT (100 subjects, 100 epochs, drop=0.4)"
run_exp 5 1 100 100 0.4 1e-3 32 "C1 High Dropout"
run_exp 6 2 100 100 0.4 1e-3 32 "C2 High Dropout"

# ===========================================
# Exp 7-8: LOWER LEARNING RATE
# ===========================================
echo ""
echo "📌 Group 4: LOWER LR (100 subjects, 150 epochs, lr=5e-4)"
run_exp 7 1 100 150 0.2 5e-4 32 "C1 Lower LR"
run_exp 8 2 100 150 0.2 5e-4 32 "C2 Lower LR"

# ===========================================
# Exp 9-10: LARGER BATCH SIZE
# ===========================================
echo ""
echo "📌 Group 5: LARGER BATCH (100 subjects, 100 epochs, batch=64)"
run_exp 9 1 100 100 0.2 2e-3 64 "C1 Large Batch"
run_exp 10 2 100 100 0.2 2e-3 64 "C2 Large Batch"

# ===========================================
# Summary
# ===========================================
echo ""
echo "=========================================="
echo "✅ ALL EXPLORATIONS COMPLETE (S3 Streaming)!"
echo "=========================================="
echo "Finished: $(date)"
echo ""
echo "Next steps:"
echo "1. Analyze results:"
echo "   python experiments/analyze_experiments.py"
echo ""
echo "2. Compare explorations:"
echo "   python scripts/compare_exploration.py"
echo ""
echo "3. Create best submissions:"
echo "   python create_submission.py --model_c1 checkpoints/c1_best.pth --model_c2 checkpoints/c2_best.pth"
echo ""
