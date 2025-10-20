#!/bin/bash
# Setup script for EEG Challenge 2025

echo "🚀 Setting up EEG Challenge 2025 environment..."

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Miniconda or Anaconda first."
    echo "   Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create conda environment
echo "📦 Creating conda environment 'eeg2025'..."
conda env create -f environment.yml

if [ $? -ne 0 ]; then
    echo "❌ Failed to create conda environment."
    exit 1
fi

echo "✅ Environment created successfully!"
echo ""
echo "To activate the environment, run:"
echo "   conda activate eeg2025"
echo ""
echo "Next steps:"
echo "   1. Download mini dataset: python scripts/download_mini_data.py"
echo "   2. Train models: python train.py --challenge 1"
echo "   3. Create submission: python create_submission.py"
echo ""
echo "Happy training! 🧠"
