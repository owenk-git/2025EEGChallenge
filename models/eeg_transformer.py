"""
EEG-Specific Transformer Architecture

Different from current CNN-based EEGNeX:
- Uses self-attention instead of convolutions
- Captures long-range temporal dependencies
- Proven effective in time-series tasks

Architecture:
    Input EEG (129 channels, 200 time points)
    → Patch Embedding (temporal chunks)
    → Positional Encoding
    → Multi-Head Self-Attention (6 layers)
    → Feed-Forward Network
    → Global Average Pooling
    → Classification Head
"""

import torch
import torch.nn as nn
import math


class PatchEmbedding(nn.Module):
    """
    Convert EEG time series into patches (similar to Vision Transformer)

    Args:
        n_channels: Number of EEG channels (129)
        n_times: Number of time points (200)
        patch_size: Size of each temporal patch (default: 10)
        embed_dim: Embedding dimension (default: 128)
    """

    def __init__(self, n_channels=129, n_times=200, patch_size=10, embed_dim=128):
        super().__init__()

        self.n_channels = n_channels
        self.n_times = n_times
        self.patch_size = patch_size
        self.n_patches = n_times // patch_size

        # Project each patch to embedding dimension
        self.projection = nn.Conv1d(
            n_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, n_channels, n_times)

        Returns:
            (batch_size, n_patches, embed_dim)
        """
        # Conv1d expects (batch, channels, time)
        x = self.projection(x)  # (batch, embed_dim, n_patches)
        x = x.transpose(1, 2)   # (batch, n_patches, embed_dim)
        return x


class PositionalEncoding(nn.Module):
    """Add positional information to patches"""

    def __init__(self, embed_dim, max_len=100):
        super().__init__()

        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))

        pe = torch.zeros(1, max_len, embed_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, embed_dim)
        """
        return x + self.pe[:, :x.size(1), :]


class TransformerBlock(nn.Module):
    """Single Transformer block with multi-head attention"""

    def __init__(self, embed_dim=128, n_heads=8, mlp_ratio=4, dropout=0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            n_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, embed_dim)
        """
        # Multi-head self-attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # MLP with residual
        x = x + self.mlp(self.norm2(x))

        return x


class EEGTransformer(nn.Module):
    """
    Transformer model for EEG classification

    Args:
        n_channels: Number of EEG channels (129)
        n_classes: Number of output classes (1 for regression)
        n_times: Number of time points (200)
        patch_size: Temporal patch size (10)
        embed_dim: Embedding dimension (128)
        n_layers: Number of transformer layers (6)
        n_heads: Number of attention heads (8)
        mlp_ratio: MLP hidden dim ratio (4)
        dropout: Dropout rate (0.1)
        challenge_name: 'c1' or 'c2'
    """

    def __init__(
        self,
        n_channels=129,
        n_classes=1,
        n_times=200,
        patch_size=10,
        embed_dim=128,
        n_layers=6,
        n_heads=8,
        mlp_ratio=4,
        dropout=0.1,
        challenge_name='c1',
        output_range=(0.5, 1.5)
    ):
        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.challenge_name = challenge_name
        self.is_c1 = (challenge_name == 'c1')
        self.output_min, self.output_max = output_range

        # Patch embedding
        self.patch_embed = PatchEmbedding(n_channels, n_times, patch_size, embed_dim)

        # Positional encoding
        n_patches = n_times // patch_size
        self.pos_encoding = PositionalEncoding(embed_dim, n_patches)

        # Dropout after embedding
        self.dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, mlp_ratio, dropout)
            for _ in range(n_layers)
        ])

        # Final normalization
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        if self.is_c1:
            self.head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim // 2, n_classes),
                nn.Sigmoid()  # For C1, output in [0, 1]
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim // 2, n_classes)
            )

    def forward(self, x):
        """
        Args:
            x: (batch_size, n_channels, n_times)

        Returns:
            (batch_size, n_classes)
        """
        # Patch embedding
        x = self.patch_embed(x)  # (batch, n_patches, embed_dim)

        # Add positional encoding
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Final normalization
        x = self.norm(x)

        # Global average pooling over patches
        x = x.mean(dim=1)  # (batch, embed_dim)

        # Classification head
        x = self.head(x)

        # Scale output for C1
        if self.is_c1:
            x = self.output_min + x * (self.output_max - self.output_min)

        return x


def create_transformer(challenge='c1', device='cuda', **kwargs):
    """
    Factory function to create Transformer model

    Args:
        challenge: 'c1' or 'c2'
        device: 'cuda' or 'cpu'
        **kwargs: additional model arguments

    Returns:
        model: EEGTransformer model
    """
    model = EEGTransformer(challenge_name=challenge, **kwargs)
    model = model.to(device)
    return model


if __name__ == "__main__":
    # Test model
    print("Testing EEG Transformer...")

    model = create_transformer(challenge='c1', device='cpu')

    # Test forward pass
    x = torch.randn(4, 129, 200)  # batch=4, channels=129, time=200
    y = model(x)

    print(f"✅ Input shape: {x.shape}")
    print(f"✅ Output shape: {y.shape}")
    print(f"✅ Output range: [{y.min():.3f}, {y.max():.3f}]")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Parameters: {n_params:,}")

    print("\n🎉 Transformer model test passed!")
