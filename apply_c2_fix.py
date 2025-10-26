#!/usr/bin/env python3
"""
Quick fix for DEBUG_C2_TRAINING.py unpacking error
Run this script to fix the issue without git pull
"""

import re

# Read the file
with open('DEBUG_C2_TRAINING.py', 'r') as f:
    content = f.read()

# Fix 1: Line ~231
content = re.sub(
    r'predictions, source_features = model\(source_X, alpha=alpha, return_features=True\)',
    'predictions, source_features, _ = model(source_X, alpha=alpha, return_features=True)',
    content
)

# Fix 2: Line ~232
content = re.sub(
    r'_, target_features = model\(target_X, alpha=alpha, return_features=True\)',
    '_, target_features, _ = model(target_X, alpha=alpha, return_features=True)',
    content
)

# Fix 3: Line ~242
content = re.sub(
    r'target_preds, _ = model\(target_X, alpha=alpha, return_features=True\)',
    'target_preds, _, _ = model(target_X, alpha=alpha, return_features=True)',
    content
)

# Write back
with open('DEBUG_C2_TRAINING.py', 'w') as f:
    f.write(content)

print("✅ Fixed DEBUG_C2_TRAINING.py")
print("   - Line ~231: Added third return value unpacking")
print("   - Line ~232: Added third return value unpacking")
print("   - Line ~242: Added third return value unpacking")
print("\nNow run: python3 DEBUG_C2_TRAINING.py --challenge c2 --epochs 100 --batch_size 64")
