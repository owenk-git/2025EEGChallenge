#!/bin/bash
#
# MASTER SUBMISSION SCRIPT - Creates submissions for both C1 and C2
#
# Based on complete analysis of both challenges:
# - C1: Trial-level approach (breakthrough to 0.97 NRMSE)
# - C2: Recording-level ensemble (expected 1.00-1.05 NRMSE)
#

set -e  # Exit on error

echo "================================================================================"
echo "CREATING FINAL SUBMISSIONS FOR BOTH CHALLENGES"
echo "================================================================================"
echo ""
echo "Challenge 1: Trial-level RT prediction"
echo "  Current best: 1.09 NRMSE"
echo "  Expected: 0.96-1.00 NRMSE (10% improvement)"
echo ""
echo "Challenge 2: Recording-level ensemble"
echo "  Current: 1.08 NRMSE"
echo "  Expected: 1.00-1.05 NRMSE"
echo ""
echo "================================================================================"
echo ""

# ============================================================================
# CHALLENGE 1: Trial-Level Submission
# ============================================================================

echo "Creating Challenge 1 submission..."
echo "--------------------------------------------------------------------------------"

if [ -f "checkpoints/trial_level_c1_best.pt" ]; then
    echo "✅ Found C1 trial-level model"
    python3 FINAL_C1_SUBMISSION.py \
        --model_path checkpoints/trial_level_c1_best.pt \
        --output_dir submissions \
        --device cuda

    echo ""
    echo "✅ C1 submission created!"
    echo ""
else
    echo "❌ C1 model not found: checkpoints/trial_level_c1_best.pt"
    echo "   Run: python3 train_trial_level.py --challenge c1 --epochs 100 --batch_size 32"
    echo ""
fi

# ============================================================================
# CHALLENGE 2: Recording-Level Ensemble Submission
# ============================================================================

echo "Creating Challenge 2 submission..."
echo "--------------------------------------------------------------------------------"

# Check which C2 models are available
C2_MODELS=0
if [ -f "checkpoints/domain_adaptation_c2_best.pt" ]; then
    echo "✅ Found domain_adaptation_c2_best.pt"
    ((C2_MODELS++))
fi

if [ -f "checkpoints/cross_task_c2_best.pt" ]; then
    echo "✅ Found cross_task_c2_best.pt"
    ((C2_MODELS++))
fi

if [ -f "checkpoints/hybrid_c2_best.pt" ]; then
    echo "✅ Found hybrid_c2_best.pt"
    ((C2_MODELS++))
fi

if [ $C2_MODELS -gt 0 ]; then
    echo ""
    echo "Creating C2 ensemble with $C2_MODELS models..."
    python3 FINAL_C2_SUBMISSION.py \
        --output_dir submissions \
        --device cuda

    echo ""
    echo "✅ C2 submission created!"
    echo ""
else
    echo ""
    echo "❌ No C2 models found"
    echo "   Run one or more of:"
    echo "     python3 train_domain_adaptation_direct.py --challenge c2 --epochs 100"
    echo "     python3 train_cross_task_direct.py --challenge c2 --epochs 100"
    echo "     python3 train_hybrid_direct.py --challenge c2 --epochs 100"
    echo ""
fi

# ============================================================================
# SUMMARY
# ============================================================================

echo "================================================================================"
echo "SUBMISSION SUMMARY"
echo "================================================================================"
echo ""

# List created submissions
echo "Created submissions:"
ls -lh submissions/*.zip 2>/dev/null | tail -5 || echo "  (none yet)"

echo ""
echo "Next steps:"
echo "  1. Upload submissions to: https://www.codabench.org/competitions/4145/"
echo "  2. Check test scores"
echo "  3. If C1 test NRMSE < 1.0 → SUCCESS! (breakthrough confirmed)"
echo "  4. If C2 test NRMSE < 1.05 → SUCCESS! (ensemble working)"
echo ""
echo "Remaining submissions:"
echo "  Today: 3 out of 5"
echo "  Total: 16 out of 35"
echo ""
echo "================================================================================"
