#!/usr/bin/env python3
"""
Quick fix for DEBUG_C2_TRAINING.py - model returns 3 values, not 2
"""

file_path = 'DEBUG_C2_TRAINING.py'

with open(file_path, 'r') as f:
    content = f.read()

# Fix 1: Line 231 - unpack 3 values
old_line1 = "            predictions, source_features = model(source_X, alpha=alpha, return_features=True)"
new_line1 = "            predictions, source_features, _ = model(source_X, alpha=alpha, return_features=True)"

# Fix 2: Line 232 - unpack 3 values
old_line2 = "            _, target_features = model(target_X, alpha=alpha, return_features=True)"
new_line2 = "            _, target_features, _ = model(target_X, alpha=alpha, return_features=True)"

# Fix 3: Line 242 - unpack 3 values
old_line3 = "            target_preds, _ = model(target_X, alpha=alpha, return_features=True)"
new_line3 = "            target_preds, _, _ = model(target_X, alpha=alpha, return_features=True)"

if old_line1 in content:
    content = content.replace(old_line1, new_line1)
    print("✓ Fixed line 231")

if old_line2 in content:
    content = content.replace(old_line2, new_line2)
    print("✓ Fixed line 232")

if old_line3 in content:
    content = content.replace(old_line3, new_line3)
    print("✓ Fixed line 242")

with open(file_path, 'w') as f:
    f.write(content)

print(f"\n✅ Fixed {file_path}")
print("\nNow run:")
print("  python3 DEBUG_C2_TRAINING.py --challenge c2 --epochs 100 --batch_size 64")
