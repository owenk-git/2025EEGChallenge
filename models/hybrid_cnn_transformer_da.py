"""
Hybrid CNN-Transformer with Domain Adaptation

Combines:
- CNN (EEGNet-style) for local spatial-temporal features
- Transformer for global temporal dependencies
- ERP features (P300, N200, alpha, beta)
- Domain adaptation (MMD, entropy minimization)

Pure PyTorch implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SeparableConv1d(nn.Module):
    """Depthwise Separable Convolution"""
    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size,
                                   groups=in_channels, padding=kernel_size//2, bias=bias)
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        return x + self.pe[:, :x.size(1), :]


class CNNFeatureExtractor(nn.Module):
    """
    CNN for local feature extraction (EEGNet-style)
    """
    def __init__(self, n_channels=129, n_times=900):
        super().__init__()
        self.n_channels = n_channels
        self.n_times = n_times

        # Spatial filtering
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(1, 32, (n_channels, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU()
        )

        # Temporal filtering
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(32, 64, (1, 25), padding=(0, 12), bias=False),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.3)
        )

        # Separable convolutions
        self.sep_conv1 = nn.Sequential(
            SeparableConv1d(64, 128, kernel_size=5),
            nn.ELU(),
            nn.AvgPool1d(2),
            nn.Dropout(0.3)
        )

        self.sep_conv2 = nn.Sequential(
            SeparableConv1d(128, 128, kernel_size=5),
            nn.ELU(),
            nn.AvgPool1d(2),
            nn.Dropout(0.3)
        )

        # Adaptive pooling to handle variable time dimensions
        self.adaptive_pool = nn.AdaptiveAvgPool1d(28)  # Fixed output size
        self.reduced_time = 28  # Now always 28

        # Calculate output time dimension (kept for backward compatibility, now constant)
        # self.reduced_time = n_times // 4 // 2 // 2  # 4 from temporal, 2 from sep1, 2 from sep2

    def forward(self, x):
        """
        Args:
            x: (batch, channels, time)

        Returns:
            features: (batch, 128, reduced_time)
        """
        # Add channel dimension for Conv2d
        x = x.unsqueeze(1)  # (batch, 1, channels, time)

        # Spatial filtering
        x = self.spatial_conv(x)  # (batch, 32, 1, time)

        # Temporal filtering
        x = self.temporal_conv(x)  # (batch, 64, 1, time//4)

        # Remove spatial dimension
        x = x.squeeze(2)  # (batch, 64, time//4)

        # Separable convolutions
        x = self.sep_conv1(x)  # (batch, 128, time//8)
        x = self.sep_conv2(x)  # (batch, 128, time//16)

        # Adaptive pooling to fixed size
        x = self.adaptive_pool(x)  # (batch, 128, 28)

        return x


class TransformerEncoder(nn.Module):
    """
    Transformer encoder for global temporal dependencies
    """
    def __init__(self, d_model=128, nhead=8, num_layers=4, dim_feedforward=512, dropout=0.3):
        super().__init__()
        self.d_model = d_model

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=500)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (batch, channels, time)

        Returns:
            features: (batch, channels, time)
        """
        # Transpose to (batch, time, channels) for transformer
        x = x.transpose(1, 2)  # (batch, time, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Apply transformer
        x = self.transformer_encoder(x)  # (batch, time, d_model)

        # Transpose back to (batch, channels, time)
        x = x.transpose(1, 2)  # (batch, d_model, time)

        return x


class ERPFeatureExtractor(nn.Module):
    """
    Extract ERP features (P300, N200, alpha, beta)
    """
    def __init__(self, sfreq=100):
        super().__init__()
        self.sfreq = sfreq

    def forward(self, eeg_data):
        """
        Extract ERP features from EEG

        Args:
            eeg_data: (batch, channels, time)

        Returns:
            erp_features: (batch, 13)
        """
        batch_size, n_channels, n_times = eeg_data.shape
        device = eeg_data.device

        features_list = []

        for i in range(batch_size):
            signal = eeg_data[i]  # (channels, time)

            # P300 features (300-600ms window)
            p300_start = n_times // 3
            p300_end = 2 * n_times // 3
            p300_window = signal[:, p300_start:p300_end]  # (channels, window)

            p300_latencies = []
            p300_amplitudes = []
            for ch in range(n_channels):
                ch_window = p300_window[ch]
                peak_idx = ch_window.argmax()
                latency = (p300_start + peak_idx.item()) / self.sfreq
                amplitude = ch_window[peak_idx]
                p300_latencies.append(latency)
                p300_amplitudes.append(amplitude.item())

            p300_latencies_t = torch.tensor(p300_latencies, device=device)
            p300_amplitudes_t = torch.tensor(p300_amplitudes, device=device)

            p300_features = torch.stack([
                p300_latencies_t.mean(),
                p300_latencies_t.std(),
                p300_amplitudes_t.mean(),
                p300_amplitudes_t.max(),
                (p300_amplitudes_t * p300_latencies_t).mean()
            ])

            # N200 features (200-350ms window)
            n2_start = int(0.2 * n_times)
            n2_end = int(0.35 * n_times)
            n2_window = signal[:, n2_start:n2_end]

            n2_amplitudes = n2_window.min(dim=1)[0]  # Negative deflection
            n2_features = torch.stack([
                n2_amplitudes.mean(),
                n2_amplitudes.std(),
                n2_amplitudes.min()
            ])

            # Pre-stimulus alpha (8-13 Hz)
            # Simplified: just use mean power in first third
            prestim_window = signal[:, :n_times//3]
            alpha_power = prestim_window.pow(2).mean(dim=1)
            alpha_features = torch.stack([
                alpha_power.mean(),
                alpha_power.std(),
                alpha_power.max()
            ])

            # Motor beta (13-30 Hz)
            # Simplified: use power in middle third
            motor_window = signal[:, n_times//3:2*n_times//3]
            beta_power = motor_window.pow(2).mean(dim=1)
            beta_features = torch.stack([
                beta_power.mean(),
                beta_power.std()
            ])

            # Combine all features
            sample_features = torch.cat([
                p300_features,  # 5
                n2_features,    # 3
                alpha_features, # 3
                beta_features   # 2
            ])  # Total: 13 features

            features_list.append(sample_features)

        # Stack into batch
        erp_features = torch.stack(features_list)  # (batch, 13)

        return erp_features


class HybridCNNTransformerDA(nn.Module):
    """
    Hybrid CNN-Transformer with Domain Adaptation

    Architecture:
    1. CNN: Extract local spatial-temporal features
    2. Transformer: Model global temporal dependencies
    3. ERP: Extract neuroscience-based features
    4. Fusion: Combine learned and handcrafted features
    5. Domain Adaptation: MMD + entropy minimization
    """
    def __init__(self, n_channels=129, n_times=900, challenge='c1', output_range=None,
                 d_model=128, nhead=8, num_transformer_layers=4):
        super().__init__()
        self.n_channels = n_channels
        self.n_times = n_times
        self.challenge = challenge

        # Output range
        if output_range is None:
            self.output_range = (0.5, 1.5) if challenge == 'c1' else (-3, 3)
        else:
            self.output_range = output_range

        # CNN feature extractor
        self.cnn = CNNFeatureExtractor(n_channels, n_times)

        # Transformer encoder
        self.transformer = TransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_transformer_layers,
            dim_feedforward=512,
            dropout=0.3
        )

        # ERP feature extractor
        self.erp_extractor = ERPFeatureExtractor(sfreq=100)

        # Calculate feature dimensions (now constant due to adaptive pooling)
        transformer_time = 28  # Fixed by adaptive pooling in CNN
        learned_feature_dim = d_model * transformer_time  # 128 * 28 = 3584
        erp_feature_dim = 13

        total_feature_dim = learned_feature_dim + erp_feature_dim  # 3584 + 13 = 3597

        # Fusion and prediction head
        self.fusion = nn.Sequential(
            nn.Linear(total_feature_dim, 512),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1)
        )

    def extract_features(self, x):
        """Extract combined features"""
        # CNN features
        cnn_features = self.cnn(x)  # (batch, d_model, time)

        # Transformer features
        transformer_features = self.transformer(cnn_features)  # (batch, d_model, time)

        # Flatten transformer features (use reshape for non-contiguous tensors)
        transformer_features_flat = transformer_features.reshape(transformer_features.size(0), -1)

        # ERP features
        erp_features = self.erp_extractor(x)  # (batch, 13)

        # Concatenate learned and handcrafted features
        combined_features = torch.cat([transformer_features_flat, erp_features], dim=1)

        return combined_features

    def forward(self, x, return_features=False):
        """
        Forward pass

        Args:
            x: Input EEG (batch, channels, time)
            return_features: If True, also return features

        Returns:
            predictions: (batch,)
            features (optional): (batch, feature_dim)
        """
        # Extract features
        features = self.extract_features(x)

        # Predict
        predictions = self.fusion(features).squeeze(-1)

        # Clip to valid range
        predictions = torch.clamp(predictions, self.output_range[0], self.output_range[1])

        if return_features:
            return predictions, features
        else:
            return predictions


def create_hybrid_cnn_transformer_da(challenge='c1', device='cuda'):
    """
    Create Hybrid CNN-Transformer with Domain Adaptation

    Args:
        challenge: 'c1' or 'c2'
        device: 'cuda' or 'cpu'

    Returns:
        model
    """
    if challenge == 'c1':
        output_range = (0.5, 1.5)
    else:
        output_range = (-3, 3)

    model = HybridCNNTransformerDA(
        n_channels=129,
        n_times=900,
        challenge=challenge,
        output_range=output_range,
        d_model=128,
        nhead=8,
        num_transformer_layers=4
    )

    model = model.to(device)
    return model


def compute_mmd_loss(source_features, target_features, kernel='rbf', bandwidth=None):
    """Compute Maximum Mean Discrepancy loss"""
    if kernel == 'linear':
        source_mean = source_features.mean(dim=0)
        target_mean = target_features.mean(dim=0)
        mmd = torch.sum((source_mean - target_mean) ** 2)
    elif kernel == 'rbf':
        if bandwidth is None:
            all_features = torch.cat([source_features, target_features], dim=0)
            pairwise_distances = torch.cdist(all_features, all_features, p=2)
            bandwidth = torch.median(pairwise_distances[pairwise_distances > 0])
            if bandwidth == 0:
                bandwidth = 1.0

        def rbf_kernel(x, y, bandwidth):
            pairwise_sq_dists = torch.cdist(x, y, p=2) ** 2
            return torch.exp(-pairwise_sq_dists / (2 * bandwidth ** 2))

        batch_size_s = source_features.size(0)
        batch_size_t = target_features.size(0)

        K_ss = rbf_kernel(source_features, source_features, bandwidth)
        K_tt = rbf_kernel(target_features, target_features, bandwidth)
        K_st = rbf_kernel(source_features, target_features, bandwidth)

        mmd = K_ss.sum() / (batch_size_s * batch_size_s) + \
              K_tt.sum() / (batch_size_t * batch_size_t) - \
              2 * K_st.sum() / (batch_size_s * batch_size_t)

    return mmd


def compute_entropy_loss(predictions):
    """Entropy minimization (prediction variance)"""
    return torch.var(predictions)


class HybridLoss(nn.Module):
    """
    Combined loss for hybrid model:
    - Task loss (MSE)
    - MMD loss (distribution alignment)
    - Entropy loss (confident predictions)
    """
    def __init__(self, lambda_mmd=0.1, lambda_entropy=0.01):
        super().__init__()
        self.lambda_mmd = lambda_mmd
        self.lambda_entropy = lambda_entropy

    def forward(self, predictions, targets, source_features=None, target_features=None):
        """Compute combined loss"""
        # Task loss
        task_loss = F.mse_loss(predictions, targets)

        loss_dict = {'task_loss': task_loss.item()}
        total_loss = task_loss

        # MMD loss
        if source_features is not None and target_features is not None:
            mmd_loss = compute_mmd_loss(source_features, target_features, kernel='rbf')
            total_loss = total_loss + self.lambda_mmd * mmd_loss
            loss_dict['mmd_loss'] = mmd_loss.item()

        # Entropy loss
        entropy_loss = compute_entropy_loss(predictions)
        total_loss = total_loss + self.lambda_entropy * entropy_loss
        loss_dict['entropy_loss'] = entropy_loss.item()

        loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict


if __name__ == '__main__':
    # Test model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Testing Hybrid CNN-Transformer with Domain Adaptation...")
    model = create_hybrid_cnn_transformer_da(challenge='c1', device=device)

    # Test forward pass
    batch_size = 8
    x = torch.randn(batch_size, 129, 900).to(device)

    predictions = model(x)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions range: [{predictions.min():.2f}, {predictions.max():.2f}]")

    predictions, features = model(x, return_features=True)
    print(f"Features shape: {features.shape}")

    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Hybrid CNN-Transformer-DA model created successfully!")
