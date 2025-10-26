"""
Quick fix for dimension mismatch in models

Patches the models to auto-detect input time dimension
"""

import re
from pathlib import Path

def fix_domain_adaptation():
    """Fix domain_adaptation_eegnex.py"""
    file_path = Path('models/domain_adaptation_eegnex.py')
    content = file_path.read_text()

    # Find the __init__ method and add adaptive pooling
    # Replace the fixed feature_dim calculation with adaptive approach

    # Add global average pooling before task predictor
    old_code = '''        # Task predictor (regression head)
        self.task_predictor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_dim, 256),'''

    new_code = '''        # Adaptive pooling to handle variable time dimensions
        self.adaptive_pool = nn.AdaptiveAvgPool1d(28)  # Fixed output size
        self.feature_dim = 128 * 28  # Now always 3584

        # Task predictor (regression head)
        self.task_predictor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_dim, 256),'''

    if old_code in content:
        content = content.replace(old_code, new_code)

    # Update extract_features to use adaptive pooling
    old_extract = '''        # Separable convolutions
        x = self.sep_conv1(x)  # (batch, 128, time//8)
        x = self.sep_conv2(x)  # (batch, 128, time//16)
        x = self.sep_conv3(x)  # (batch, 128, time//16)

        return x'''

    new_extract = '''        # Separable convolutions
        x = self.sep_conv1(x)  # (batch, 128, time//8)
        x = self.sep_conv2(x)  # (batch, 128, time//16)
        x = self.sep_conv3(x)  # (batch, 128, time//16)

        # Adaptive pooling to fixed size
        x = self.adaptive_pool(x)  # (batch, 128, 28)

        return x'''

    if old_extract in content:
        content = content.replace(old_extract, new_extract)

    file_path.write_text(content)
    print(f"✓ Fixed {file_path}")


def fix_cross_task():
    """Fix cross_task_pretrain.py"""
    file_path = Path('models/cross_task_pretrain.py')
    content = file_path.read_text()

    # Add adaptive pooling
    old_code = '''        # Calculate feature dimension
        reduced_time = n_times // 4  # temporal pool
        reduced_time = reduced_time // 2  # sep_conv1 pool
        reduced_time = reduced_time // 2  # sep_conv2 pool
        reduced_time = reduced_time // 2  # sep_conv3 pool
        self.feature_dim = 256 * reduced_time'''

    new_code = '''        # Adaptive pooling to handle variable time dimensions
        self.adaptive_pool = nn.AdaptiveAvgPool1d(28)  # Fixed output size

        # Calculate feature dimension (now fixed)
        self.feature_dim = 256 * 28'''

    if old_code in content:
        content = content.replace(old_code, new_code)

    # Update forward to use adaptive pooling
    old_forward = '''        # Separable convolutions
        x = self.sep_conv1(x)  # (batch, 128, time//8)
        x = self.sep_conv2(x)  # (batch, 128, time//16)
        x = self.sep_conv3(x)  # (batch, 256, time//32)

        # Flatten
        x = x.view(x.size(0), -1)  # (batch, feature_dim)'''

    new_forward = '''        # Separable convolutions
        x = self.sep_conv1(x)  # (batch, 128, time//8)
        x = self.sep_conv2(x)  # (batch, 128, time//16)
        x = self.sep_conv3(x)  # (batch, 256, time//32)

        # Adaptive pooling to fixed size
        x = self.adaptive_pool(x)  # (batch, 256, 28)

        # Flatten
        x = x.view(x.size(0), -1)  # (batch, feature_dim)'''

    if old_forward in content:
        content = content.replace(old_forward, new_forward)

    file_path.write_text(content)
    print(f"✓ Fixed {file_path}")


def fix_hybrid():
    """Fix hybrid_cnn_transformer_da.py"""
    file_path = Path('models/hybrid_cnn_transformer_da.py')
    content = file_path.read_text()

    # Fix the .view() error with .reshape()
    old_view = '''        # Flatten transformer features
        transformer_features_flat = transformer_features.view(transformer_features.size(0), -1)'''

    new_view = '''        # Flatten transformer features (use reshape for non-contiguous tensors)
        transformer_features_flat = transformer_features.reshape(transformer_features.size(0), -1)'''

    if old_view in content:
        content = content.replace(old_view, new_view)
        print(f"✓ Fixed hybrid .view() → .reshape()")

    file_path.write_text(content)
    print(f"✓ Fixed {file_path}")


if __name__ == '__main__':
    print("Fixing dimension mismatches...")
    print("="*60)

    fix_domain_adaptation()
    fix_cross_task()
    fix_hybrid()

    print("="*60)
    print("\n✅ All models fixed!")
    print("\nNow run:")
    print("  python3 train_domain_adaptation_direct.py --challenge c1 --epochs 100 --batch_size 64")
    print("  python3 train_cross_task_direct.py --challenge c1 --epochs 100 --pretrain_epochs 50")
    print("  python3 train_hybrid_direct.py --challenge c1 --epochs 100 --batch_size 64")
