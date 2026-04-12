"""ResNet implementation for CIFAR-10.

Used as the teacher / base model throughout HW3: robustness evaluation,
AugMix fine-tuning, PGD adversarial testing, and knowledge distillation.

Reference:
    He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for
    image recognition. CVPR 2016.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Residual block with two 3×3 convolutions (used in ResNet-18/34).

    Args:
        in_channels: Number of input channels.
        channels: Number of output channels.
        stride: Stride for the first convolution. Default: 1.
        norm: Normalisation layer constructor. Default: ``nn.BatchNorm2d``.

    Attributes:
        expansion: Channel expansion factor (1 for BasicBlock).

    Shape:
        Input:  (N, in_channels, H, W)
        Output: (N, channels, H/stride, W/stride)
    """

    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        channels: int,
        stride: int = 1,
        norm=nn.BatchNorm2d,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = norm(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = norm(channels)

        self.shortcut: nn.Module = nn.Sequential()
        if stride != 1 or in_channels != channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, channels, 1, stride=stride, bias=False),
                norm(channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the residual block."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + self.shortcut(x))
        return out


class ResNet(nn.Module):
    """ResNet for CIFAR-10 (32×32 input, 3-channel RGB).

    The initial conv layer uses a 3×3 kernel with stride 1 (no max-pool),
    matching the CIFAR-adapted design from the original paper.

    Args:
        block: Residual block class (e.g. :class:`BasicBlock`).
        num_blocks: List of block counts per stage, e.g. ``[2, 2, 2, 2]``
            for ResNet-18.
        norm: Normalisation layer constructor. Default: ``nn.BatchNorm2d``.
        num_classes: Number of output classes. Default: 10.

    Shape:
        Input:  (N, 3, 32, 32)
        Output: (N, num_classes)

    Example:
        >>> model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)
        >>> logits = model(torch.randn(8, 3, 32, 32))   # (8, 10)
    """

    def __init__(
        self,
        block,
        num_blocks,
        norm=nn.BatchNorm2d,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm(64)
        self.layer1 = self._make_layer(block, 64,  num_blocks[0], norm=norm, stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], norm=norm, stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], norm=norm, stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], norm=norm, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(
        self, block, channels: int, num_blocks: int, norm, stride: int
    ) -> nn.Sequential:
        """Build one residual stage."""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, channels, s, norm))
            self.in_channels = channels * block.expansion
        return nn.Sequential(*layers)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract the pooled feature vector (before the linear head).

        Useful for T-SNE visualisation of clean vs. adversarial embeddings.

        Args:
            x: Input tensor (N, 3, 32, 32).

        Returns:
            Feature tensor (N, 512).
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        return out.view(out.size(0), -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (N, 3, 32, 32).

        Returns:
            Class logits (N, num_classes).
        """
        return self.linear(self.get_features(x))
