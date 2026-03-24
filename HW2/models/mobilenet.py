"""MobileNetV2 for CIFAR-10, used as the student in modified knowledge distillation.

Reference:
    Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018).
    MobileNetV2: Inverted residuals and linear bottlenecks. CVPR 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InvertedResidual(nn.Module):
    """Inverted residual block (expand → depthwise → project).

    Args:
        in_planes: Number of input channels.
        out_planes: Number of output channels.
        expansion: Channel expansion factor for the intermediate layer.
        stride: Stride for the depthwise convolution.

    Shape:
        Input:  (N, in_planes, H, W)
        Output: (N, out_planes, H/stride, W/stride)
    """

    def __init__(
        self, in_planes: int, out_planes: int, expansion: int, stride: int
    ) -> None:
        super().__init__()
        self.stride = stride
        planes = expansion * in_planes

        self.conv1 = nn.Conv2d(in_planes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut: nn.Module = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, 1, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.stride == 1:
            out = out + self.shortcut(x)
        return out


class MobileNetV2(nn.Module):
    """MobileNetV2 adapted for CIFAR-10 (32×32 input).

    Strides in stages 2 and the initial conv are changed from 2→1 to keep
    spatial resolution compatible with 32×32 images.

    Args:
        num_classes: Number of output classes. Default: 10.

    Shape:
        Input:  (N, 3, 32, 32)
        Output: (N, num_classes)

    Example:
        >>> model = MobileNetV2(num_classes=10)
        >>> logits = model(torch.randn(4, 3, 32, 32))   # (4, 10)
    """

    # (expansion, out_planes, num_blocks, stride)
    _cfg = [
        (1,  16, 1, 1),
        (6,  24, 2, 1),   # stride 2→1 for CIFAR-10
        (6,  32, 3, 2),
        (6,  64, 4, 2),
        (6,  96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1),
    ]

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False)  # stride 2→1
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)

    def _make_layers(self, in_planes: int) -> nn.Sequential:
        layers = []
        for expansion, out_planes, num_blocks, stride in self._cfg:
            strides = [stride] + [1] * (num_blocks - 1)
            for s in strides:
                layers.append(InvertedResidual(in_planes, out_planes, expansion, s))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (N, 3, 32, 32).

        Returns:
            Class logits (N, num_classes).
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out, 4)   # kernel=4 for CIFAR-10
        out = out.view(out.size(0), -1)
        return self.linear(out)
