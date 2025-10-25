"""
Pretrained CNN for EEG (Practical Alternative to BENDR)

Since BENDR requires specific library, use pretrained EfficientNet/ResNet:
1. Adapt first layer for EEG (129 channels instead of 3 RGB)
2. Use pretrained ImageNet features
3. Fine-tune on EEG data

This works because:
- Lower layers learn general features (edges, patterns)
- Can transfer to EEG temporal patterns
- Much better than training from scratch

Usage:
    from models.pretrained_eeg import create_pretrained_model
    model = create_pretrained_model('efficientnet_b0', challenge='c1')
"""

import torch
import torch.nn as nn
try:
    from torchvision import models
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    print("⚠️ torchvision not available, pretrained models won't work")


class EEGPretrainedCNN(nn.Module):
    """
    Adapt pretrained CNN for EEG

    Strategy:
    - Use pretrained ResNet/EfficientNet backbone
    - Replace first conv layer (3→129 channels)
    - Replace final layer for regression
    - Fine-tune entire network
    """

    def __init__(
        self,
        backbone='resnet18',
        n_channels=129,
        n_times=200,
        n_classes=1,
        challenge_name='c1',
        output_range=(0.5, 1.5),
        pretrained=True
    ):
        super().__init__()

        if not TORCHVISION_AVAILABLE:
            raise ImportError("torchvision required for pretrained models")

        self.challenge_name = challenge_name
        self.is_c1 = (challenge_name == 'c1')
        self.output_min, self.output_max = output_range

        # Load pretrained backbone
        if backbone == 'resnet18':
            base_model = models.resnet18(pretrained=pretrained)
            feature_dim = 512
        elif backbone == 'resnet34':
            base_model = models.resnet34(pretrained=pretrained)
            feature_dim = 512
        elif backbone == 'resnet50':
            base_model = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        elif backbone == 'efficientnet_b0':
            base_model = models.efficientnet_b0(pretrained=pretrained)
            feature_dim = 1280
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        print(f"✅ Loaded pretrained {backbone}")

        # Adapt first layer for EEG (129 channels instead of 3)
        if 'resnet' in backbone:
            original_conv = base_model.conv1
            # Create new conv with same params but 129 input channels
            self.conv1 = nn.Conv2d(
                1,  # We'll reshape EEG as (1, 129, 200)
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False
            )

            # Initialize new conv by averaging pretrained weights across input channels
            with torch.no_grad():
                self.conv1.weight[:, 0, :, :] = original_conv.weight.mean(dim=1)

            # Keep rest of ResNet
            self.bn1 = base_model.bn1
            self.relu = base_model.relu
            self.maxpool = base_model.maxpool
            self.layer1 = base_model.layer1
            self.layer2 = base_model.layer2
            self.layer3 = base_model.layer3
            self.layer4 = base_model.layer4
            self.avgpool = base_model.avgpool

        elif 'efficientnet' in backbone:
            # EfficientNet adaptation
            original_conv = base_model.features[0][0]
            self.features = base_model.features
            self.features[0][0] = nn.Conv2d(
                1,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False
            )
            with torch.no_grad():
                self.features[0][0].weight[:, 0, :, :] = original_conv.weight.mean(dim=1)

            self.avgpool = base_model.avgpool

        self.backbone = backbone

        # New classification head
        if self.is_c1:
            self.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(feature_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, n_classes),
                nn.Sigmoid()
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(feature_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, n_classes)
            )

    def forward(self, x):
        """
        Args:
            x: (batch, 129, 200) EEG data

        Returns:
            (batch, 1) predictions
        """
        # Reshape EEG to image-like: (batch, 1, 129, 200)
        # Treat channels as height, time as width
        x = x.unsqueeze(1)  # (batch, 1, 129, 200)

        # Pass through backbone
        if 'resnet' in self.backbone:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = torch.flatten(x, 1)

        elif 'efficientnet' in self.backbone:
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)

        # Classification
        x = self.classifier(x)

        # Scale for C1
        if self.is_c1:
            x = self.output_min + x * (self.output_max - self.output_min)

        return x


def create_pretrained_model(
    backbone='resnet18',
    challenge='c1',
    device='cuda',
    **kwargs
):
    """
    Factory function to create pretrained model

    Args:
        backbone: 'resnet18', 'resnet34', 'resnet50', 'efficientnet_b0'
        challenge: 'c1' or 'c2'
        device: 'cuda' or 'cpu'

    Returns:
        model: Pretrained CNN adapted for EEG
    """
    model = EEGPretrainedCNN(
        backbone=backbone,
        challenge_name=challenge,
        **kwargs
    )
    model = model.to(device)
    return model


if __name__ == "__main__":
    print("Testing Pretrained EEG CNN...")

    # Test ResNet
    model = create_pretrained_model('resnet18', 'c1', 'cpu')

    x = torch.randn(4, 129, 200)
    y = model(x)

    print(f"✅ Input: {x.shape}")
    print(f"✅ Output: {y.shape}, range [{y.min():.3f}, {y.max():.3f}]")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Parameters: {n_params:,}")

    print("\n🎉 Pretrained model test passed!")
