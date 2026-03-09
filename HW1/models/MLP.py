from typing import List

import torch
import torch.nn as nn

from parameters import ModelParams


# Supported activation functions
ACTIVATIONS: dict = {
    "relu":      nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
    "gelu":      nn.GELU,
}


class MLP(nn.Module):
    """Multi-Layer Perceptron for image classification.

    Each hidden block consists of: Linear → BatchNorm1d → Activation → Dropout,
    stored in a nn.ModuleList. The output layer is a single nn.Linear.
    Input is flattened automatically using nn.Flatten.

    Args:
        params: ModelParams dataclass containing architecture configuration.
    """

    def __init__(self, params: ModelParams) -> None:
        super().__init__()

        act_cls = ACTIVATIONS.get(params.activation.lower())
        if act_cls is None:
            raise ValueError(
                f"Unknown activation '{params.activation}'. "
                f"Choose from {list(ACTIVATIONS)}."
            )

        self.flatten = nn.Flatten()

        # Each hidden block is a Sequential; all blocks stored in a ModuleList
        in_dim: int = params.input_size
        hidden_blocks: List[nn.Module] = []

        for h in params.hidden_sizes:
            if params.bn_after_act:
                block = nn.Sequential(
                    nn.Linear(in_dim, h),
                    act_cls(),
                    nn.BatchNorm1d(h),
                    nn.Dropout(params.dropout),
                )
            else:
                block = nn.Sequential(
                    nn.Linear(in_dim, h),
                    nn.BatchNorm1d(h),
                    act_cls(),
                    nn.Dropout(params.dropout),
                )
            hidden_blocks.append(block)
            in_dim = h

        self.hidden = nn.ModuleList(hidden_blocks)
        self.output = nn.Linear(in_dim, params.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (B, 1, 28, 28).

        Returns:
            Logits tensor of shape (B, num_classes).
        """
        x = self.flatten(x)
        for block in self.hidden:
            x = block(x)
        return self.output(x)
