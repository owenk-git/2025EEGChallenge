"""
Cross-Task Pre-Training Model

Strategy:
1. Pre-train on passive tasks (resting state, video watching)
2. Fine-tune on active cognitive task (CCD)
3. Multi-task learning across all 6 tasks

Pure PyTorch implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class SharedFeatureExtractor(nn.Module):
    """
    Shared feature extractor for all tasks
    Learns general EEG representations
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

        self.sep_conv3 = nn.Sequential(
            SeparableConv1d(128, 256, kernel_size=5),
            nn.ELU(),
            nn.AvgPool1d(2),
            nn.Dropout(0.3)
        )

        # Adaptive pooling to handle variable time dimensions
        self.adaptive_pool = nn.AdaptiveAvgPool1d(28)  # Fixed output size

        # Calculate feature dimension (now fixed)
        self.feature_dim = 256 * 28

    def forward(self, x):
        """
        Extract features from EEG

        Args:
            x: (batch, channels, time)

        Returns:
            features: (batch, feature_dim)
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
        x = self.sep_conv3(x)  # (batch, 256, time//32)

        # Adaptive pooling to fixed size
        x = self.adaptive_pool(x)  # (batch, 256, 28)

        # Flatten
        x = x.view(x.size(0), -1)  # (batch, feature_dim)

        return x


class TaskSpecificHead(nn.Module):
    """
    Task-specific prediction head
    """
    def __init__(self, feature_dim, output_range=None):
        super().__init__()
        self.output_range = output_range

        self.head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Linear(64, 1)
        )

    def forward(self, features):
        """
        Predict from features

        Args:
            features: (batch, feature_dim)

        Returns:
            predictions: (batch,)
        """
        predictions = self.head(features).squeeze(-1)

        if self.output_range is not None:
            predictions = torch.clamp(predictions, self.output_range[0], self.output_range[1])

        return predictions


class CrossTaskPretrainModel(nn.Module):
    """
    Cross-Task Pre-Training Model

    Can be trained in two modes:
    1. Multi-task pre-training: Train on all 6 tasks simultaneously
    2. Fine-tuning: Fine-tune on specific task (CCD)
    """
    def __init__(self, n_channels=129, n_times=900, num_tasks=6, task_names=None,
                 output_ranges=None):
        super().__init__()
        self.n_channels = n_channels
        self.n_times = n_times
        self.num_tasks = num_tasks
        self.task_names = task_names or [f'task_{i}' for i in range(num_tasks)]

        # Shared feature extractor
        self.feature_extractor = SharedFeatureExtractor(n_channels, n_times)

        # Task-specific heads
        self.task_heads = nn.ModuleDict()
        for i, task_name in enumerate(self.task_names):
            if output_ranges and i < len(output_ranges):
                output_range = output_ranges[i]
            else:
                output_range = None
            self.task_heads[task_name] = TaskSpecificHead(
                self.feature_extractor.feature_dim,
                output_range=output_range
            )

    def forward(self, x, task_name=None, return_features=False):
        """
        Forward pass

        Args:
            x: Input EEG (batch, channels, time)
            task_name: Name of task to predict (if None, predict all tasks)
            return_features: If True, also return features

        Returns:
            If task_name is specified: predictions for that task
            If task_name is None: dict of predictions for all tasks
            If return_features=True: (predictions, features)
        """
        # Extract features
        features = self.feature_extractor(x)

        if task_name is not None:
            # Single task prediction
            predictions = self.task_heads[task_name](features)

            if return_features:
                return predictions, features
            else:
                return predictions
        else:
            # Multi-task prediction
            predictions = {}
            for name, head in self.task_heads.items():
                predictions[name] = head(features)

            if return_features:
                return predictions, features
            else:
                return predictions

    def freeze_feature_extractor(self):
        """Freeze feature extractor for fine-tuning"""
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def unfreeze_feature_extractor(self):
        """Unfreeze feature extractor"""
        for param in self.feature_extractor.parameters():
            param.requires_grad = True


def create_cross_task_model(challenge='c1', device='cuda'):
    """
    Create Cross-Task Pre-Training model for specific challenge

    Args:
        challenge: 'c1' or 'c2'
        device: 'cuda' or 'cpu'

    Returns:
        model: CrossTaskPretrainModel
    """
    # Define task names (6 tasks in HBN dataset)
    task_names = [
        'resting_state',
        'video_watching',
        'reading',
        'contrast_change_detection',  # CCD - our target task
        'task_5',
        'task_6'
    ]

    # Define output ranges for each task (if known)
    # For now, use generic ranges
    output_ranges = [
        None,  # resting state
        None,  # video watching
        None,  # reading
        (0.5, 1.5) if challenge == 'c1' else (-3, 3),  # CCD
        None,
        None
    ]

    model = CrossTaskPretrainModel(
        n_channels=129,
        n_times=900,
        num_tasks=6,
        task_names=task_names,
        output_ranges=output_ranges
    )

    model = model.to(device)
    return model


class CrossTaskLoss(nn.Module):
    """
    Multi-task loss for pre-training

    Can weight tasks differently
    """
    def __init__(self, task_weights=None, num_tasks=6):
        super().__init__()
        if task_weights is None:
            # Equal weights by default
            self.task_weights = torch.ones(num_tasks) / num_tasks
        else:
            self.task_weights = torch.tensor(task_weights)

    def forward(self, predictions_dict, targets_dict):
        """
        Compute multi-task loss

        Args:
            predictions_dict: Dict of {task_name: predictions}
            targets_dict: Dict of {task_name: targets}

        Returns:
            total_loss, loss_dict
        """
        loss_dict = {}
        total_loss = 0.0

        for i, (task_name, predictions) in enumerate(predictions_dict.items()):
            if task_name in targets_dict:
                targets = targets_dict[task_name]
                task_loss = F.mse_loss(predictions, targets)

                weight = self.task_weights[i]
                total_loss = total_loss + weight * task_loss

                loss_dict[f'{task_name}_loss'] = task_loss.item()

        loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict


if __name__ == '__main__':
    # Test model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Testing Cross-Task Pre-Training Model...")
    model = create_cross_task_model(challenge='c1', device=device)

    # Test forward pass
    batch_size = 8
    x = torch.randn(batch_size, 129, 900).to(device)

    # Multi-task prediction
    predictions = model(x)
    print(f"Multi-task predictions:")
    for task_name, preds in predictions.items():
        print(f"  {task_name}: {preds.shape}")

    # Single task prediction
    single_pred = model(x, task_name='contrast_change_detection')
    print(f"\nSingle task prediction shape: {single_pred.shape}")

    # With features
    single_pred, features = model(x, task_name='contrast_change_detection', return_features=True)
    print(f"Features shape: {features.shape}")

    # Test multi-task loss
    targets_dict = {
        'contrast_change_detection': torch.randn(batch_size).to(device),
        'resting_state': torch.randn(batch_size).to(device),
        'video_watching': torch.randn(batch_size).to(device)
    }

    loss_fn = CrossTaskLoss(num_tasks=6)
    total_loss, loss_dict = loss_fn(predictions, targets_dict)

    print(f"\nLoss breakdown:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.4f}")

    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Cross-Task Pre-Training model created successfully!")
