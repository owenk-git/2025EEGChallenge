#!/bin/bash
#
# Run multiple aggressive training experiments in parallel
# Find the best configuration for C1 and C2
#

echo "=========================================="
echo "🚀 Running Aggressive Training Experiments"
echo "=========================================="

# Create logs directory
mkdir -p logs_experiments

# C1 Experiments - try different configurations
echo "Starting C1 experiments..."

# C1-1: Huber loss, median RT
python train_aggressive_c1.py \
    -e 50 -b 64 --lr 0.0001 --dropout 0.35 \
    --loss huber --early_stop 10 \
    --checkpoint_dir checkpoints_c1_huber_median \
    > logs_experiments/c1_huber_median.log 2>&1 &
PID1=$!

# C1-2: Huber loss, trimmed mean RT
python train_aggressive_c1.py \
    -e 50 -b 64 --lr 0.0001 --dropout 0.35 \
    --loss huber --early_stop 10 \
    --checkpoint_dir checkpoints_c1_huber_trimmed \
    > logs_experiments/c1_huber_trimmed.log 2>&1 &
PID2=$!

# C1-3: MAE loss, different output range
python train_aggressive_c1.py \
    -e 50 -b 64 --lr 0.0001 --dropout 0.40 \
    --loss mae --early_stop 10 \
    --output_min 0.4 --output_max 1.6 \
    --checkpoint_dir checkpoints_c1_mae_wide \
    > logs_experiments/c1_mae_wide.log 2>&1 &
PID3=$!

echo "C1 experiments started (PIDs: $PID1, $PID2, $PID3)"

# C2 Experiments
echo "Starting C2 experiments..."

# C2-1: EXTREME regularization
python train_aggressive_c2.py \
    -e 100 -b 32 --lr 0.00005 --dropout 0.45 \
    --loss huber --early_stop 15 \
    --checkpoint_dir checkpoints_c2_extreme \
    > logs_experiments/c2_extreme.log 2>&1 &
PID4=$!

# C2-2: MAE loss, slightly less regularization
python train_aggressive_c2.py \
    -e 100 -b 32 --lr 0.0001 --dropout 0.40 \
    --loss mae --early_stop 15 \
    --checkpoint_dir checkpoints_c2_mae \
    > logs_experiments/c2_mae.log 2>&1 &
PID5=$!

echo "C2 experiments started (PIDs: $PID4, $PID5)"

echo ""
echo "=========================================="
echo "All experiments running in background!"
echo "=========================================="
echo "Monitor progress:"
echo "  tail -f logs_experiments/c1_huber_median.log"
echo "  tail -f logs_experiments/c2_extreme.log"
echo ""
echo "Wait for all to complete:"
echo "  wait $PID1 $PID2 $PID3 $PID4 $PID5"
echo ""
echo "Check best results:"
echo "  python find_best_checkpoint.py"
echo "=========================================="

# Optional: wait for all to finish
# wait $PID1 $PID2 $PID3 $PID4 $PID5
# echo "All experiments completed!"
