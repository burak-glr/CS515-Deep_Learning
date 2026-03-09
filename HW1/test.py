import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets

from parameters import DataParams, TrainParams
from train import get_transforms


@torch.no_grad()
def run_test(
    model: nn.Module,
    data_params: DataParams,
    train_params: TrainParams,
    device: torch.device,
) -> None:
    """Evaluate the model on the test set and print per-class accuracy.

    Loads the best saved weights from disk before evaluation.

    Args:
        model: The neural network model.
        data_params: Dataset parameters.
        train_params: Training parameters (batch size, save path).
        device: Device to run computations on.
    """
    tf = get_transforms(data_params, train=False)

    if data_params.dataset == "mnist":
        test_ds = datasets.MNIST(data_params.data_dir, train=False, download=True, transform=tf)
    else:  # cifar10
        test_ds = datasets.CIFAR10(data_params.data_dir, train=False, download=True, transform=tf)

    loader = DataLoader(
        test_ds,
        batch_size=train_params.batch_size,
        shuffle=False,
        num_workers=data_params.num_workers,
    )

    model.load_state_dict(torch.load(train_params.save_path, map_location=device))
    model.eval()

    correct, n = 0, 0
    class_correct: list = [0] * data_params.num_classes
    class_total: list   = [0] * data_params.num_classes

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(1)
        correct += preds.eq(labels).sum().item()
        n       += imgs.size(0)
        for p, t in zip(preds, labels):
            class_correct[t] += int(p == t)
            class_total[t]   += 1

    print("\n=== Test Results ===")
    print(f"Overall accuracy: {correct/n:.4f}  ({correct}/{n})\n")
    for i in range(data_params.num_classes):
        acc = class_correct[i] / class_total[i]
        print(f"  Class {i}: {acc:.4f}  ({class_correct[i]}/{class_total[i]})")
