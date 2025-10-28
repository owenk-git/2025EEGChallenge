"""
Domain Adaptation EEGNeX with MMD, Entropy Minimization, and Subject-Adversarial Training

Pure PyTorch implementation (no external dependencies)
Based on literature review recommendations for cross-subject generalization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GradientReversalLayer(torch.autograd.Function):
    """
    Gradient Reversal Layer for adversarial domain adaptation
    Passes forward normally, but reverses gradients during backprop
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


def gradient_reversal(x, lambda_=1.0):
    """Apply gradient reversal"""
    return GradientReversalLayer.apply(x, lambda_)


class SubjectDiscriminator(nn.Module):
    """
    Subject discriminator for adversarial training
    Tries to predict which subject the features come from
    Feature extractor learns to fool this discriminator
    """
    def __init__(self, input_dim, num_subjects):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_subjects)
        )

    def forward(self, x):
        return self.discriminator(x)


class SeparableConv1d(nn.Module):
    """Depthwise Separable Convolution for EEG"""
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


class EEGNeXBlock(nn.Module):
    """EEGNeX Block with residual connection"""
    def __init__(self, channels, kernel_size=5):
        super().__init__()
        self.sep_conv = SeparableConv1d(channels, channels, kernel_size)
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        residual = x
        x = self.sep_conv(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x + residual


class DomainAdaptationEEGNeX(nn.Module):
    """
    EEGNeX with Domain Adaptation:
    - Feature extractor
    - Task predictor (RT or psychopathology)
    - Subject discriminator (adversarial)

    Training uses:
    - Task loss (NRMSE)
    - MMD loss (align distributions)
    - Entropy minimization (confident predictions)
    - Adversarial loss (subject-invariant features)
    """
    def __init__(self, n_channels=129, n_times=900, challenge='c1', num_subjects=100, output_range=None):
        super().__init__()
        self.n_channels = n_channels
        self.n_times = n_times
        self.challenge = challenge
        self.num_subjects = num_subjects

        # Output range for clipping predictions
        if output_range is None:
            self.output_range = (0.5, 1.5) if challenge == 'c1' else (-3, 3)
        else:
            self.output_range = output_range

        # Initial spatial filtering (across channels)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(1, 32, (n_channels, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU()
        )

        # Temporal convolution
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(32, 64, (1, 25), padding=(0, 12), bias=False),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.3)
        )

        # Flatten to 1D for separable convs
        # After spatial: (batch, 32, 1, time)
        # After temporal + pool: (batch, 64, 1, time//4)
        reduced_time = n_times // 4

        # Separable conv blocks (feature extractor)
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

        self.sep_conv3 = EEGNeXBlock(128, kernel_size=5)

        # Calculate final feature dimension
        final_time = reduced_time // 4  # Two pooling layers (2, 2)
        self.feature_dim = 128 * final_time

        # Adaptive pooling to handle variable time dimensions
        self.adaptive_pool = nn.AdaptiveAvgPool1d(28)  # Fixed output size
        self.feature_dim = 128 * 28  # Now always 3584

        # Task predictor (regression head)
        self.task_predictor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_dim, 256),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Linear(64, 1)
        )

        # Subject discriminator (for adversarial training)
        self.subject_discriminator = SubjectDiscriminator(self.feature_dim, num_subjects)

    def extract_features(self, x):
        """Extract features from EEG (for domain adaptation)"""
        # x: (batch, channels, time)
        batch_size = x.size(0)

        # Add channel dimension for Conv2d
        x = x.unsqueeze(1)  # (batch, 1, channels, time)

        # Spatial filtering
        x = self.spatial_conv(x)  # (batch, 32, 1, time)

        # Temporal filtering
        x = self.temporal_conv(x)  # (batch, 64, 1, time//4)

        # Remove spatial dimension and convert to 1D
        x = x.squeeze(2)  # (batch, 64, time//4)

        # Separable convolutions
        x = self.sep_conv1(x)  # (batch, 128, time//8)
        x = self.sep_conv2(x)  # (batch, 128, time//16)
        x = self.sep_conv3(x)  # (batch, 128, time//16)

        # Adaptive pooling to fixed size
        x = self.adaptive_pool(x)  # (batch, 128, 28)

        return x

    def forward(self, x, alpha=0.0, return_features=False):
        """
        Forward pass

        Args:
            x: Input EEG (batch, channels, time)
            alpha: Gradient reversal strength for adversarial training
            return_features: If True, return features for domain adaptation

        Returns:
            If return_features=False: predictions
            If return_features=True: (predictions, features, subject_logits)
        """
        # Extract features
        features = self.extract_features(x)  # (batch, 128, time//16)

        # Task prediction
        predictions = self.task_predictor(features)  # (batch, 1)
        # KEEP (batch, 1) shape - competition expects this format
        # predictions = predictions.squeeze(-1)  # REMOVED - was causing shape error

        # Clip predictions to valid range
        predictions = torch.clamp(predictions, self.output_range[0], self.output_range[1])

        if return_features:
            # Flatten features for discriminator
            features_flat = features.view(features.size(0), -1)  # (batch, feature_dim)

            # Apply gradient reversal
            reversed_features = gradient_reversal(features_flat, alpha)

            # Subject discrimination
            subject_logits = self.subject_discriminator(reversed_features)

            return predictions, features_flat, subject_logits
        else:
            return predictions


def compute_mmd_loss(source_features, target_features, kernel='rbf', bandwidth=None):
    """
    Compute Maximum Mean Discrepancy (MMD) loss
    Measures difference between two distributions

    Args:
        source_features: Features from source subjects (batch_s, feature_dim)
        target_features: Features from target subjects (batch_t, feature_dim)
        kernel: 'rbf' or 'linear'
        bandwidth: RBF kernel bandwidth (auto-computed if None)

    Returns:
        MMD loss (scalar)
    """
    batch_size_s = source_features.size(0)
    batch_size_t = target_features.size(0)

    if kernel == 'linear':
        # Linear MMD
        source_mean = source_features.mean(dim=0)
        target_mean = target_features.mean(dim=0)
        mmd = torch.sum((source_mean - target_mean) ** 2)

    elif kernel == 'rbf':
        # RBF (Gaussian) kernel MMD
        if bandwidth is None:
            # Auto-compute bandwidth as median pairwise distance
            all_features = torch.cat([source_features, target_features], dim=0)
            pairwise_distances = torch.cdist(all_features, all_features, p=2)
            bandwidth = torch.median(pairwise_distances[pairwise_distances > 0])
            if bandwidth == 0:
                bandwidth = 1.0

        def rbf_kernel(x, y, bandwidth):
            """RBF kernel k(x,y) = exp(-||x-y||^2 / (2*bandwidth^2))"""
            pairwise_sq_dists = torch.cdist(x, y, p=2) ** 2
            return torch.exp(-pairwise_sq_dists / (2 * bandwidth ** 2))

        # K(source, source)
        K_ss = rbf_kernel(source_features, source_features, bandwidth)
        # K(target, target)
        K_tt = rbf_kernel(target_features, target_features, bandwidth)
        # K(source, target)
        K_st = rbf_kernel(source_features, target_features, bandwidth)

        # MMD^2 = E[K(s,s)] + E[K(t,t)] - 2*E[K(s,t)]
        mmd = K_ss.sum() / (batch_size_s * batch_size_s) + \
              K_tt.sum() / (batch_size_t * batch_size_t) - \
              2 * K_st.sum() / (batch_size_s * batch_size_t)

    return mmd


def compute_entropy_loss(predictions):
    """
    Entropy minimization loss
    Encourages confident predictions (low entropy)

    For regression, we use prediction variance as proxy for entropy
    Lower variance = more confident
    """
    # Compute variance of predictions
    pred_variance = torch.var(predictions)
    return pred_variance


def create_domain_adaptation_eegnex(challenge='c1', num_subjects=100, device='cuda'):
    """
    Create Domain Adaptation EEGNeX model

    Args:
        challenge: 'c1' or 'c2'
        num_subjects: Number of subjects in training set
        device: 'cuda' or 'cpu'

    Returns:
        model: DomainAdaptationEEGNeX model
    """
    if challenge == 'c1':
        output_range = (0.5, 1.5)
    else:
        output_range = (-3, 3)

    model = DomainAdaptationEEGNeX(
        n_channels=129,
        n_times=900,
        challenge=challenge,
        num_subjects=num_subjects,
        output_range=output_range
    )

    model = model.to(device)
    return model


class DomainAdaptationLoss(nn.Module):
    """
    Combined loss for domain adaptation:
    - Task loss (NRMSE)
    - MMD loss (distribution alignment)
    - Entropy loss (confident predictions)
    - Adversarial loss (subject invariance)
    """
    def __init__(self, lambda_mmd=0.1, lambda_entropy=0.01, lambda_adv=0.1):
        super().__init__()
        self.lambda_mmd = lambda_mmd
        self.lambda_entropy = lambda_entropy
        self.lambda_adv = lambda_adv

    def forward(self, predictions, targets, source_features=None, target_features=None,
                subject_logits=None, subject_labels=None):
        """
        Compute combined loss

        Args:
            predictions: Model predictions (batch,)
            targets: Ground truth (batch,)
            source_features: Features from source subjects (optional)
            target_features: Features from target subjects (optional)
            subject_logits: Subject classification logits (optional)
            subject_labels: True subject IDs (optional)

        Returns:
            total_loss, loss_dict
        """
        # Task loss (MSE for regression)
        task_loss = F.mse_loss(predictions, targets)

        loss_dict = {'task_loss': task_loss.item()}
        total_loss = task_loss

        # MMD loss (if source and target features provided)
        if source_features is not None and target_features is not None:
            mmd_loss = compute_mmd_loss(source_features, target_features, kernel='rbf')
            total_loss = total_loss + self.lambda_mmd * mmd_loss
            loss_dict['mmd_loss'] = mmd_loss.item()

        # Entropy loss
        entropy_loss = compute_entropy_loss(predictions)
        total_loss = total_loss + self.lambda_entropy * entropy_loss
        loss_dict['entropy_loss'] = entropy_loss.item()

        # Adversarial loss (subject classification)
        if subject_logits is not None and subject_labels is not None:
            adv_loss = F.cross_entropy(subject_logits, subject_labels)
            total_loss = total_loss + self.lambda_adv * adv_loss
            loss_dict['adv_loss'] = adv_loss.item()

        loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict


if __name__ == '__main__':
    # Test model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Testing Domain Adaptation EEGNeX...")
    model = create_domain_adaptation_eegnex(challenge='c1', num_subjects=50, device=device)

    # Test forward pass
    batch_size = 8
    x = torch.randn(batch_size, 129, 900).to(device)

    # Without domain adaptation
    predictions = model(x)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions range: [{predictions.min():.2f}, {predictions.max():.2f}]")

    # With domain adaptation
    predictions, features, subject_logits = model(x, alpha=1.0, return_features=True)
    print(f"Features shape: {features.shape}")
    print(f"Subject logits shape: {subject_logits.shape}")

    # Test MMD loss
    source_features = torch.randn(8, features.size(1)).to(device)
    target_features = torch.randn(8, features.size(1)).to(device)
    mmd = compute_mmd_loss(source_features, target_features)
    print(f"MMD loss: {mmd.item():.4f}")

    # Test combined loss
    loss_fn = DomainAdaptationLoss(lambda_mmd=0.1, lambda_entropy=0.01, lambda_adv=0.1)
    targets = torch.randn(batch_size).to(device)
    subject_labels = torch.randint(0, 50, (batch_size,)).to(device)

    total_loss, loss_dict = loss_fn(
        predictions, targets,
        source_features=source_features,
        target_features=target_features,
        subject_logits=subject_logits,
        subject_labels=subject_labels
    )

    print(f"\nLoss breakdown:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.4f}")

    print("\nDomain Adaptation EEGNeX model created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
