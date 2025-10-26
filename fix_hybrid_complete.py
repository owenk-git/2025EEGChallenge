#!/usr/bin/env python3
"""
Complete fix for hybrid_cnn_transformer_da.py
This adds adaptive pooling to make feature dimensions constant.
"""

def fix_hybrid_model():
    """Fix hybrid_cnn_transformer_da.py with adaptive pooling"""
    file_path = 'models/hybrid_cnn_transformer_da.py'

    with open(file_path, 'r') as f:
        content = f.read()

    # Fix 1: Add adaptive_pool to CNNFeatureExtractor __init__
    old_init = '''        self.sep_conv2 = nn.Sequential(
            SeparableConv1d(128, 128, kernel_size=5),
            nn.ELU(),
            nn.AvgPool1d(2),
            nn.Dropout(0.3)
        )

        # Calculate output time dimension
        self.reduced_time = n_times // 4 // 2 // 2  # 4 from temporal, 2 from sep1, 2 from sep2'''

    new_init = '''        self.sep_conv2 = nn.Sequential(
            SeparableConv1d(128, 128, kernel_size=5),
            nn.ELU(),
            nn.AvgPool1d(2),
            nn.Dropout(0.3)
        )

        # Adaptive pooling to handle variable time dimensions
        self.adaptive_pool = nn.AdaptiveAvgPool1d(28)  # Fixed output size
        self.reduced_time = 28  # Now always 28

        # Calculate output time dimension (kept for backward compatibility, now constant)
        # self.reduced_time = n_times // 4 // 2 // 2  # 4 from temporal, 2 from sep1, 2 from sep2'''

    if old_init in content:
        content = content.replace(old_init, new_init)
        print("✓ Added adaptive_pool to CNNFeatureExtractor __init__")
    else:
        print("⚠ Pattern not found in __init__, trying alternative...")
        # Try simpler pattern
        old_simple = '''        # Calculate output time dimension
        self.reduced_time = n_times // 4 // 2 // 2  # 4 from temporal, 2 from sep1, 2 from sep2'''

        new_simple = '''        # Adaptive pooling to handle variable time dimensions
        self.adaptive_pool = nn.AdaptiveAvgPool1d(28)  # Fixed output size
        self.reduced_time = 28  # Now always 28'''

        if old_simple in content:
            content = content.replace(old_simple, new_simple)
            print("✓ Added adaptive_pool to CNNFeatureExtractor __init__ (alternative)")

    # Fix 2: Add adaptive_pool call in CNNFeatureExtractor forward
    old_forward = '''        # Separable convolutions
        x = self.sep_conv1(x)  # (batch, 128, time//8)
        x = self.sep_conv2(x)  # (batch, 128, time//16)

        return x'''

    new_forward = '''        # Separable convolutions
        x = self.sep_conv1(x)  # (batch, 128, time//8)
        x = self.sep_conv2(x)  # (batch, 128, time//16)

        # Adaptive pooling to fixed size
        x = self.adaptive_pool(x)  # (batch, 128, 28)

        return x'''

    if old_forward in content:
        content = content.replace(old_forward, new_forward)
        print("✓ Added adaptive_pool call in CNNFeatureExtractor forward")

    # Fix 3: Update feature dimension calculation in HybridCNNTransformerDA
    old_calc = '''        # Calculate feature dimensions
        transformer_time = self.cnn.reduced_time
        learned_feature_dim = d_model * transformer_time
        erp_feature_dim = 13

        total_feature_dim = learned_feature_dim + erp_feature_dim'''

    new_calc = '''        # Calculate feature dimensions (now constant due to adaptive pooling)
        transformer_time = 28  # Fixed by adaptive pooling in CNN
        learned_feature_dim = d_model * transformer_time  # 128 * 28 = 3584
        erp_feature_dim = 13

        total_feature_dim = learned_feature_dim + erp_feature_dim  # 3584 + 13 = 3597'''

    if old_calc in content:
        content = content.replace(old_calc, new_calc)
        print("✓ Updated feature dimension calculation to use constant")

    # Fix 4: Change .view() to .reshape() for non-contiguous tensors
    old_view = 'transformer_features_flat = transformer_features.view(transformer_features.size(0), -1)'
    new_reshape = 'transformer_features_flat = transformer_features.reshape(transformer_features.size(0), -1)'

    if old_view in content:
        content = content.replace(old_view, new_reshape)
        print("✓ Changed .view() to .reshape()")

    # Write fixed content
    with open(file_path, 'w') as f:
        f.write(content)

    print(f"\n✅ Fixed {file_path}")
    print("\nFeature dimensions:")
    print("  - CNN output: (batch, 128, 28)")
    print("  - Transformer output: (batch, 128, 28)")
    print("  - Flattened transformer: (batch, 3584)")
    print("  - ERP features: (batch, 13)")
    print("  - Total features: (batch, 3597)")
    print("  - Fusion input: 3597 → 512 → 256 → 128 → 1")


if __name__ == '__main__':
    print("Fixing hybrid_cnn_transformer_da.py...")
    print("=" * 60)
    fix_hybrid_model()
    print("=" * 60)
    print("\nNow run:")
    print("  python3 train_hybrid_direct.py --challenge c1 --epochs 100 --batch_size 64")
