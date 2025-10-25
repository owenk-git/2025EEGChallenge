"""
Create submission using Transformer models

Usage:
    python create_transformer_submission.py
"""

import argparse
import zipfile
import shutil
from pathlib import Path
from datetime import datetime


# I'll use the same approach as create_submission.py but load transformer models
# The submission.py will be similar but with Transformer architecture instead of EEGNeX

print("="*70)
print("📦 Creating Transformer Submission")
print("="*70)
print("\nTo create transformer submission:")
print("1. Train models first:")
print("   python train_transformer.py -c 1")
print("   python train_transformer.py -c 2")
print("\n2. Then use create_submission.py with transformer checkpoints:")
print("   python create_submission.py \\")
print("       --model_c1 checkpoints_transformer/c1_transformer_best.pth \\")
print("       --model_c2 checkpoints_transformer/c2_transformer_best.pth \\")
print("       --output transformer_submission.zip")
print("\nNote: The model architecture is saved in the checkpoint,")
print("so create_submission.py will work with transformer models too!")
print("="*70)
