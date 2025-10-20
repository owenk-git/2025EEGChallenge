#!/bin/bash
# Create Submission ZIP file

echo "📦 Creating Submission Package"
echo ""

# Check if weights exist
if [ ! -f "checkpoints/c1_best.pth" ]; then
    echo "❌ Error: checkpoints/c1_best.pth not found!"
    echo "Run: bash run_challenge_1.sh first"
    exit 1
fi

if [ ! -f "checkpoints/c2_best.pth" ]; then
    echo "❌ Error: checkpoints/c2_best.pth not found!"
    echo "Run: bash run_challenge_2.sh first"
    exit 1
fi

# Create submission
python create_submission.py \
  --model_c1 checkpoints/c1_best.pth \
  --model_c2 checkpoints/c2_best.pth

echo ""
echo "✅ Submission created!"
echo ""
echo "📤 Upload to Codabench:"
echo "   1. Go to: https://www.codabench.org/competitions/9975/"
echo "   2. Click 'Participate' → 'Submit'"
echo "   3. Upload your ZIP file"
echo ""
ls -lh *_submission.zip 2>/dev/null | tail -1
