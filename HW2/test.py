"""Evaluation on the CIFAR-10 test split.

Loads the best checkpoint from ``params["save_path"]``, runs inference, and
prints overall accuracy as well as per-class accuracy.
"""

from typing import Dict, Any, List

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from train import get_transforms

CIFAR10_CLASSES: List[str] = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


@torch.no_grad()
def run_test(model: torch.nn.Module, params: Dict[str, Any], device: torch.device) -> None:
    """Evaluate *model* on the CIFAR-10 test set.

    Loads the saved checkpoint from ``params["save_path"]``, sets the model
    to evaluation mode, and prints overall and per-class accuracies.

    Args:
        model: Model architecture (weights loaded inside this function).
        params: Configuration dict from :func:`parameters.get_params`.
        device: Compute device.
    """
    tf = get_transforms(params, train=False)
    test_ds = datasets.CIFAR10(params["data_dir"], train=False, download=True, transform=tf)
    loader  = DataLoader(
        test_ds, batch_size=params["batch_size"],
        shuffle=False, num_workers=params["num_workers"],
    )

    checkpoint = torch.load(params["save_path"], map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    correct, n = 0, 0
    class_correct = [0] * params["num_classes"]
    class_total   = [0] * params["num_classes"]

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(1)
        correct += preds.eq(labels).sum().item()
        n       += labels.size(0)
        for p, t in zip(preds.tolist(), labels.tolist()):
            class_correct[t] += int(p == t)
            class_total[t]   += 1

    print("\n=== Test Results ===")
    print(f"Overall accuracy: {correct / n:.4f}  ({correct}/{n})\n")
    for i, name in enumerate(CIFAR10_CLASSES):
        acc = class_correct[i] / class_total[i]
        print(f"  {name:12s}: {acc:.4f}  ({class_correct[i]}/{class_total[i]})")
