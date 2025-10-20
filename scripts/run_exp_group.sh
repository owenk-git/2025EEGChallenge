#!/bin/bash

# Run specific experiment group
# Usage: ./run_exp_group.sh <group_number> [gpu_id]
#   Group 1: Baseline (Exp 1-2)
#   Group 2: More Data (Exp 3-4)
#   Group 3: Higher Dropout (Exp 5-6)
#   Group 4: Lower LR (Exp 7-8)
#   Group 5: Larger Batch (Exp 9-10)

set -e

GROUP=$1
GPU=${2:-0}

if [ -z "$GROUP" ]; then
    echo "Usage: ./run_exp_group.sh <group_number> [gpu_id]"
    echo ""
    echo "Groups:"
    echo "  1: Baseline (Exp 1-2) - 50 subjects, 50 epochs"
    echo "  2: More Data (Exp 3-4) - 200 subjects, 100 epochs"
    echo "  3: Higher Dropout (Exp 5-6) - 100 subjects, 100 epochs, drop=0.4"
    echo "  4: Lower LR (Exp 7-8) - 100 subjects, 150 epochs, lr=5e-4"
    echo "  5: Larger Batch (Exp 9-10) - 100 subjects, 100 epochs, batch=64"
    exit 1
fi

echo "Running Group $GROUP on GPU $GPU"
echo "Started: $(date)"

case $GROUP in
    1)
        echo "📌 Group 1: BASELINE"
        CUDA_VISIBLE_DEVICES=$GPU python train.py -c 1 -d dummy -o --max 50 -e 50 --drop 0.2 --lr 1e-3 -b 32 --num 1
        CUDA_VISIBLE_DEVICES=$GPU python train.py -c 2 -d dummy -o --max 50 -e 50 --drop 0.2 --lr 1e-3 -b 32 --num 2
        ;;
    2)
        echo "📌 Group 2: MORE DATA"
        CUDA_VISIBLE_DEVICES=$GPU python train.py -c 1 -d dummy -o --max 200 -e 100 --drop 0.2 --lr 1e-3 -b 32 --num 3
        CUDA_VISIBLE_DEVICES=$GPU python train.py -c 2 -d dummy -o --max 200 -e 100 --drop 0.2 --lr 1e-3 -b 32 --num 4
        ;;
    3)
        echo "📌 Group 3: HIGHER DROPOUT"
        CUDA_VISIBLE_DEVICES=$GPU python train.py -c 1 -d dummy -o --max 100 -e 100 --drop 0.4 --lr 1e-3 -b 32 --num 5
        CUDA_VISIBLE_DEVICES=$GPU python train.py -c 2 -d dummy -o --max 100 -e 100 --drop 0.4 --lr 1e-3 -b 32 --num 6
        ;;
    4)
        echo "📌 Group 4: LOWER LR"
        CUDA_VISIBLE_DEVICES=$GPU python train.py -c 1 -d dummy -o --max 100 -e 150 --drop 0.2 --lr 5e-4 -b 32 --num 7
        CUDA_VISIBLE_DEVICES=$GPU python train.py -c 2 -d dummy -o --max 100 -e 150 --drop 0.2 --lr 5e-4 -b 32 --num 8
        ;;
    5)
        echo "📌 Group 5: LARGER BATCH"
        CUDA_VISIBLE_DEVICES=$GPU python train.py -c 1 -d dummy -o --max 100 -e 100 --drop 0.2 --lr 2e-3 -b 64 --num 9
        CUDA_VISIBLE_DEVICES=$GPU python train.py -c 2 -d dummy -o --max 100 -e 100 --drop 0.2 --lr 2e-3 -b 64 --num 10
        ;;
    *)
        echo "Error: Invalid group number $GROUP"
        exit 1
        ;;
esac

echo "✅ Group $GROUP complete!"
echo "Finished: $(date)"
