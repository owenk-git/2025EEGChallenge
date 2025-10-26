#!/usr/bin/env python3
"""
Apply all dimension mismatch fixes to the models.
Run this script on the remote server to patch all 3 models.
"""

def fix_domain_adaptation():
    """Fix domain_adaptation_eegnex.py"""
    file_path = 'models/domain_adaptation_eegnex.py'

    with open(file_path, 'r') as f:
        content = f.read()

    # Fix 1: Add adaptive pooling in __init__
    old_init = '''        # Calculate final feature dimension
        final_time = reduced_time // 4  # Two pooling layers (2, 2)
        self.feature_dim = 128 * final_time

        # Task predictor (regression head)'''

    new_init = '''        # Calculate final feature dimension
        final_time = reduced_time // 4  # Two pooling layers (2, 2)
        self.feature_dim = 128 * final_time

        # Adaptive pooling to handle variable time dimensions
        self.adaptive_pool = nn.AdaptiveAvgPool1d(28)  # Fixed output size
        self.feature_dim = 128 * 28  # Now always 3584

        # Task predictor (regression head)'''

    if old_init in content and 'self.adaptive_pool' not in content:
        content = content.replace(old_init, new_init)
        print("✓ Added adaptive_pool to __init__")

    # Fix 2: Add adaptive pooling in extract_features
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

    if old_extract in content and 'self.adaptive_pool(x)' not in content:
        content = content.replace(old_extract, new_extract)
        print("✓ Added adaptive_pool to extract_features")

    with open(file_path, 'w') as f:
        f.write(content)

    print(f"✓ Fixed {file_path}")


def fix_cross_task():
    """Fix cross_task_pretrain.py"""
    file_path = 'models/cross_task_pretrain.py'

    with open(file_path, 'r') as f:
        content = f.read()

    # Fix 1: Add adaptive pooling in __init__
    old_init = '''        # Calculate feature dimension after convolutions
        final_time = reduced_time // 4  # Two pooling layers
        self.feature_dim = 128 * final_time

        # Task-specific heads'''

    new_init = '''        # Calculate feature dimension after convolutions
        final_time = reduced_time // 4  # Two pooling layers
        self.feature_dim = 128 * final_time

        # Adaptive pooling to handle variable time dimensions
        self.adaptive_pool = nn.AdaptiveAvgPool1d(28)  # Fixed output size
        self.feature_dim = 128 * 28  # Now always 3584

        # Task-specific heads'''

    if old_init in content and 'self.adaptive_pool' not in content:
        content = content.replace(old_init, new_init)
        print("✓ Added adaptive_pool to __init__")

    # Fix 2: Add adaptive pooling in forward
    old_forward = '''        # Separable convolutions
        x = self.sep_conv1(x)
        x = self.sep_conv2(x)
        x = self.sep_conv3(x)

        # Flatten features'''

    new_forward = '''        # Separable convolutions
        x = self.sep_conv1(x)
        x = self.sep_conv2(x)
        x = self.sep_conv3(x)

        # Adaptive pooling to fixed size
        x = self.adaptive_pool(x)  # (batch, 128, 28)

        # Flatten features'''

    if old_forward in content and 'self.adaptive_pool(x)' not in content:
        content = content.replace(old_forward, new_forward)
        print("✓ Added adaptive_pool to forward")

    with open(file_path, 'w') as f:
        f.write(content)

    print(f"✓ Fixed {file_path}")


def fix_hybrid():
    """Fix hybrid_cnn_transformer_da.py"""
    file_path = 'models/hybrid_cnn_transformer_da.py'

    with open(file_path, 'r') as f:
        content = f.read()

    # Fix 1: Change .view() to .reshape()
    old_view = 'transformer_features_flat = transformer_features.view(transformer_features.size(0), -1)'
    new_reshape = 'transformer_features_flat = transformer_features.reshape(transformer_features.size(0), -1)'

    if old_view in content:
        content = content.replace(old_view, new_reshape)
        print("✓ Changed .view() to .reshape()")

    # Fix 2: Add adaptive pooling in CNNFeatureExtractor
    old_cnn = '''        # Separable convolutions
        x = self.sep_conv1(x)
        x = self.sep_conv2(x)
        x = self.sep_conv3(x)

        return x  # (batch, 128, time//16)'''

    new_cnn = '''        # Separable convolutions
        x = self.sep_conv1(x)
        x = self.sep_conv2(x)
        x = self.sep_conv3(x)

        # Adaptive pooling to fixed size
        x = self.adaptive_pool(x)  # (batch, 128, 28)

        return x  # (batch, 128, 28)'''

    if old_cnn in content and 'self.adaptive_pool(x)' not in content:
        content = content.replace(old_cnn, new_cnn)
        print("✓ Added adaptive_pool to CNNFeatureExtractor forward")

    # Fix 3: Add adaptive_pool in CNNFeatureExtractor __init__
    old_cnn_init = '''        self.sep_conv1 = EEGNeXBlock(128, kernel_size=3)
        self.sep_conv2 = EEGNeXBlock(128, kernel_size=3)
        self.sep_conv3 = EEGNeXBlock(128, kernel_size=5)'''

    new_cnn_init = '''        self.sep_conv1 = EEGNeXBlock(128, kernel_size=3)
        self.sep_conv2 = EEGNeXBlock(128, kernel_size=3)
        self.sep_conv3 = EEGNeXBlock(128, kernel_size=5)

        # Adaptive pooling to handle variable time dimensions
        self.adaptive_pool = nn.AdaptiveAvgPool1d(28)'''

    if old_cnn_init in content and 'self.adaptive_pool = nn.AdaptiveAvgPool1d(28)' not in content:
        content = content.replace(old_cnn_init, new_cnn_init)
        print("✓ Added adaptive_pool to CNNFeatureExtractor __init__")

    with open(file_path, 'w') as f:
        f.write(content)

    print(f"✓ Fixed {file_path}")


if __name__ == '__main__':
    print("Applying all fixes to models...")
    print("=" * 60)

    print("\n1. Fixing domain_adaptation_eegnex.py...")
    fix_domain_adaptation()

    print("\n2. Fixing cross_task_pretrain.py...")
    fix_cross_task()

    print("\n3. Fixing hybrid_cnn_transformer_da.py...")
    fix_hybrid()

    print("\n" + "=" * 60)
    print("✅ All fixes applied!")
    print("\nNow run:")
    print("  python3 train_domain_adaptation_direct.py --challenge c1 --epochs 100 --batch_size 64")
    print("  python3 train_cross_task_direct.py --challenge c1 --epochs 100 --pretrain_epochs 50")
    print("  python3 train_hybrid_direct.py --challenge c1 --epochs 100 --batch_size 64")
