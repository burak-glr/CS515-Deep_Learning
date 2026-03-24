"""Transfer learning model builders for CIFAR-10 classification.

Provides two adaptation strategies for pretrained ImageNet models:
  Option 1 — Resize CIFAR-10 images to 224×224, freeze all feature layers,
              and fine-tune only the classification head.
  Option 2 — Keep 32×32 input, replace the first conv with a smaller-stride
              variant, remove the max-pool, and fine-tune the full network.
"""

from dataclasses import dataclass
from typing import Tuple

import torch.nn as nn
from torchvision import models


@dataclass
class TransferConfig:
    """Configuration for building a transfer-learning model.

    Attributes:
        arch: Backbone architecture, one of ``"resnet18"`` or ``"vgg16"``.
        option: Adaptation strategy.
            ``"1"`` — resize input + freeze early layers.
            ``"2"`` — change first conv for 32×32 + fine-tune all.
        num_classes: Number of output classes (10 for CIFAR-10).
    """

    arch: str = "resnet18"
    option: str = "1"
    num_classes: int = 10


def build_transfer_model(cfg: TransferConfig) -> Tuple[nn.Module, int]:
    """Build a pretrained model adapted for CIFAR-10.

    Args:
        cfg: :class:`TransferConfig` specifying the backbone, option, and
            number of target classes.

    Returns:
        A tuple ``(model, image_size)`` where *image_size* is the spatial
        dimension the input images should be resized to before feeding into
        the model (224 for option 1, 32 for option 2).

    Raises:
        ValueError: If *cfg.arch* is not one of ``"resnet18"`` or ``"vgg16"``.

    Examples:
        >>> cfg = TransferConfig(arch="resnet18", option="1", num_classes=10)
        >>> model, img_size = build_transfer_model(cfg)
        >>> img_size
        224
    """
    if cfg.arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, cfg.num_classes)

        if cfg.option == "1":
            # Freeze every parameter except the new FC head.
            for name, param in model.named_parameters():
                if "fc" not in name:
                    param.requires_grad = False
            return model, 224

        # Option 2: adapt first conv + remove max-pool so 32×32 flows through.
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()  # type: ignore[assignment]
        return model, 32

    if cfg.arch == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        model.classifier[6] = nn.Linear(4096, cfg.num_classes)

        if cfg.option == "1":
            # Freeze feature extractor; only the classifier is trainable.
            for name, param in model.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False
            return model, 224

        # Option 2: shrink first conv, use adaptive pool for small feature maps.
        # VGG-16 has 5 max-pools: 32→16→8→4→2→1, so features end at 1×1×512.
        model.features[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Replace the heavy three-layer classifier with a single linear layer.
        model.classifier = nn.Linear(512, cfg.num_classes)
        return model, 32

    raise ValueError(f"Unknown architecture: {cfg.arch!r}. Choose 'resnet18' or 'vgg16'.")


def count_trainable_params(model: nn.Module) -> int:
    """Return the number of trainable parameters in *model*.

    Args:
        model: A PyTorch :class:`~torch.nn.Module`.

    Returns:
        Total count of parameters with ``requires_grad=True``.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
