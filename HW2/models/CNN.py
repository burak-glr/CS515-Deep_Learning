"""Simple CNN architecture used as a student model in knowledge distillation."""

import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """Lightweight CNN for CIFAR-10, used as the student in distillation.

    Architecture:
        Conv(3→32, 3×3, pad=1) → ReLU → MaxPool(2)
        Conv(32→64, 3×3, pad=1) → ReLU → MaxPool(2)
        Flatten → FC(64×8×8→128) → ReLU → FC(128→num_classes)

    Kaiming (He) initialization is applied to all Conv and Linear layers.

    Args:
        num_classes: Number of output classes. Default: 10.

    Shape:
        Input:  (N, 3, 32, 32)
        Output: (N, num_classes)

    Example:
        >>> model = SimpleCNN(num_classes=10)
        >>> x = torch.randn(4, 3, 32, 32)
        >>> logits = model(x)   # shape (4, 10)
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        """Apply Kaiming normal initialization to Conv2d and Linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input tensor of shape (N, 3, 32, 32).

        Returns:
            Logits tensor of shape (N, num_classes).
        """
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)   # 32→16
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)   # 16→8
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
