#!/bin/bash
# Complete Pipeline: Test → Train → Submit

echo "🎯 Complete Training Pipeline"
echo "=============================="
echo ""

# Step 1: Quick test
echo "Step 1: Quick Test (5 min)"
bash run_test.sh

if [ $? -ne 0 ]; then
    echo "❌ Test failed! Fix issues before continuing."
    exit 1
fi

echo ""
echo "✅ Test passed! Continuing with training..."
echo ""

# Step 2: Train Challenge 1
echo "Step 2: Training Challenge 1 (~8-12 hours)"
bash run_challenge_1.sh

if [ $? -ne 0 ]; then
    echo "❌ Challenge 1 training failed!"
    exit 1
fi

# Step 3: Train Challenge 2
echo "Step 3: Training Challenge 2 (~8-12 hours)"
bash run_challenge_2.sh

if [ $? -ne 0 ]; then
    echo "❌ Challenge 2 training failed!"
    exit 1
fi

# Step 4: Create submission
echo "Step 4: Creating submission"
bash create_submission.sh

echo ""
echo "🎉 Complete pipeline finished!"
echo "Upload your submission ZIP to Codabench!"
