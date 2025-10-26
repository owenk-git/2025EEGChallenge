#!/bin/bash
#
# Master script to solve C1 prediction collapse
# Runs all solutions in recommended order
#

set -e  # Exit on error

echo "================================================================================"
echo "SOLVING C1 PREDICTION COLLAPSE"
echo "================================================================================"
echo ""

# Solution 1: Diagnose (2 mins)
echo "Step 1: Diagnosing training data distribution..."
echo "--------------------------------------------------------------------------------"
python3 diagnose_c1_distribution.py
echo ""
read -p "Press Enter to continue to temperature scaling solutions..."
echo ""

# Solution 2: Temperature Scaling T=1.3 (5 mins)
echo "Step 2: Creating submission with Temperature=1.3 (conservative)..."
echo "--------------------------------------------------------------------------------"
python3 C1_SUBMISSION_TEMPERATURE.py --temperature 1.3 --device cuda
echo ""
echo "✅ Created: submissions/c1_temperature_T1.3_*.zip"
echo ""

# Solution 3: Temperature Scaling T=1.5 (5 mins)
echo "Step 3: Creating submission with Temperature=1.5 (moderate - RECOMMENDED)..."
echo "--------------------------------------------------------------------------------"
python3 C1_SUBMISSION_TEMPERATURE.py --temperature 1.5 --device cuda
echo ""
echo "✅ Created: submissions/c1_temperature_T1.5_*.zip"
echo ""

# Solution 4: Temperature Scaling T=2.0 (5 mins)
echo "Step 4: Creating submission with Temperature=2.0 (aggressive)..."
echo "--------------------------------------------------------------------------------"
python3 C1_SUBMISSION_TEMPERATURE.py --temperature 2.0 --device cuda
echo ""
echo "✅ Created: submissions/c1_temperature_T2.0_*.zip"
echo ""

echo "================================================================================"
echo "QUICK FIXES COMPLETE"
echo "================================================================================"
echo ""
echo "Created 3 submissions with different temperature scaling:"
echo "  1. T=1.3 (conservative) - Expected NRMSE: 1.00-1.05"
echo "  2. T=1.5 (moderate)     - Expected NRMSE: 0.95-1.02 ⭐ RECOMMENDED"
echo "  3. T=2.0 (aggressive)   - Expected NRMSE: 0.92-1.00"
echo ""
echo "Next steps:"
echo "  1. Upload all 3 submissions to competition"
echo "  2. Compare scores"
echo "  3. If none beat 1.09, proceed to diversity loss retraining"
echo ""

read -p "Do you want to proceed with diversity loss retraining? (y/N) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Step 5: Retraining with diversity loss (4-5 hours)..."
    echo "--------------------------------------------------------------------------------"
    python3 train_trial_level_with_diversity.py \
        --challenge c1 \
        --epochs 100 \
        --batch_size 32 \
        --lambda_diversity 0.1 \
        --device cuda
    echo ""
    echo "✅ Retrained model saved: checkpoints/trial_level_c1_diversity_best.pt"
    echo ""
    echo "Creating submission with diversity-trained model..."
    python3 C1_SUBMISSION_TEMPERATURE.py \
        --temperature 1.0 \
        --checkpoint checkpoints/trial_level_c1_diversity_best.pt \
        --device cuda
    echo ""
    echo "✅ Created: submissions/c1_temperature_T1.0_*.zip (diversity-trained)"
fi

echo ""
echo "================================================================================"
echo "ALL SOLUTIONS COMPLETE"
echo "================================================================================"
echo ""
echo "Submissions ready in submissions/ directory"
echo "Upload to competition and compare results!"
echo ""
