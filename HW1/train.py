import copy
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from parameters import DataParams, TrainParams


def get_transforms(data_params: DataParams, train: bool = True) -> transforms.Compose:
    """Build the torchvision transform pipeline for a given dataset split.

    Args:
        data_params: Dataset parameters (dataset name, mean, std).
        train: If True, apply training augmentations (CIFAR-10 only).

    Returns:
        A composed transform pipeline.
    """
    mean, std = data_params.mean, data_params.std

    if data_params.dataset == "mnist":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:  # cifar10
        if train:
            return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])


def get_loaders(
    data_params: DataParams,
    train_params: TrainParams,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders.

    Args:
        data_params: Dataset parameters.
        train_params: Training parameters (batch size, num workers).

    Returns:
        Tuple of (train_loader, val_loader).
    """
    train_tf = get_transforms(data_params, train=True)
    val_tf   = get_transforms(data_params, train=False)

    if data_params.dataset == "mnist":
        train_ds = datasets.MNIST(data_params.data_dir, train=True,  download=True, transform=train_tf)
        val_ds   = datasets.MNIST(data_params.data_dir, train=False, download=True, transform=val_tf)
    else:  # cifar10
        train_ds = datasets.CIFAR10(data_params.data_dir, train=True,  download=True, transform=train_tf)
        val_ds   = datasets.CIFAR10(data_params.data_dir, train=False, download=True, transform=val_tf)

    train_loader = DataLoader(
        train_ds, batch_size=train_params.batch_size,
        shuffle=True, num_workers=data_params.num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_params.batch_size,
        shuffle=False, num_workers=data_params.num_workers,
    )
    return train_loader, val_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    log_interval: int,
    regularizer: str = "l2",
    reg_coeff: float = 0.0,
) -> Tuple[float, float]:
    """Run one full training epoch.

    Args:
        model: The neural network model.
        loader: Training DataLoader.
        optimizer: Optimizer instance.
        criterion: Loss function.
        device: Device to run computations on.
        log_interval: How often (in batches) to print progress.
        regularizer: Regularization type ('l1' or 'l2').
        reg_coeff: Regularization coefficient (used for L1).

    Returns:
        Tuple of (average loss, accuracy) over the epoch.
    """
    model.train()
    total_loss, correct, n = 0.0, 0, 0

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        if regularizer == "l1":
            l1 = sum(p.abs().sum() for p in model.parameters())
            loss = loss + reg_coeff * l1
        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item() * imgs.size(0)
        correct    += out.argmax(1).eq(labels).sum().item()
        n          += imgs.size(0)

        if (batch_idx + 1) % log_interval == 0:
            print(f"  [{batch_idx+1}/{len(loader)}] "
                  f"loss: {total_loss/n:.4f}  acc: {correct/n:.4f}", flush=True)

    return total_loss / n, correct / n


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate the model on a validation or test set.

    Args:
        model: The neural network model.
        loader: Validation DataLoader.
        criterion: Loss function.
        device: Device to run computations on.

    Returns:
        Tuple of (average loss, accuracy).
    """
    model.eval()
    total_loss, correct, n = 0.0, 0, 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out  = model(imgs)
            loss = criterion(out, labels)
            total_loss += loss.detach().item() * imgs.size(0)
            correct    += out.argmax(1).eq(labels).sum().item()
            n          += imgs.size(0)

    return total_loss / n, correct / n


def run_training(
    model: nn.Module,
    data_params: DataParams,
    train_params: TrainParams,
    device: torch.device,
) -> Tuple[list, list, list, list]:
    """Full training loop with validation, early stopping, and model checkpointing.

    Args:
        model: The neural network model.
        data_params: Dataset parameters.
        train_params: Training hyperparameters.
        device: Device to run computations on.

    Returns:
        Tuple of (train_losses, val_losses, train_accs, val_accs) per epoch.
    """
    train_loader, val_loader = get_loaders(data_params, train_params)
    criterion = nn.CrossEntropyLoss()
    wd = train_params.weight_decay if train_params.regularizer == "l2" else 0.0
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_params.learning_rate,
        weight_decay=wd,
    )
    if train_params.scheduler == "steplr":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    elif train_params.scheduler == "reducelronplateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3)
    else:  # cosineannealinglr
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_params.epochs)

    best_val_loss  = float("inf")
    best_weights   = None
    patience_count = 0

    train_losses: list = []
    val_losses:   list = []
    train_accs:   list = []
    val_accs:     list = []

    for epoch in range(1, train_params.epochs + 1):
        print(f"\nEpoch {epoch}/{train_params.epochs}", flush=True)

        tr_loss, tr_acc   = train_one_epoch(
            model, train_loader, optimizer, criterion, device, train_params.log_interval,
            regularizer=train_params.regularizer,
            reg_coeff=train_params.weight_decay,
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        if train_params.scheduler == "reducelronplateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()

        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        train_accs.append(tr_acc)
        val_accs.append(val_acc)

        print(f"  Train loss: {tr_loss:.4f}  acc: {tr_acc:.4f}", flush=True)
        print(f"  Val   loss: {val_loss:.4f}  acc: {val_acc:.4f}", flush=True)

        # Checkpoint based on best validation loss
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_weights   = copy.deepcopy(model.state_dict())
            patience_count = 0
            torch.save(best_weights, train_params.save_path)
            print(f"  Saved best model (val_loss={best_val_loss:.4f})", flush=True)
        else:
            patience_count += 1

        # Early stopping
        if train_params.early_stop_patience > 0:
            if patience_count >= train_params.early_stop_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs.")
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)
    print(f"\nTraining done. Best val loss: {best_val_loss:.4f}")

    return train_losses, val_losses, train_accs, val_accs
