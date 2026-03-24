"""Training utilities for HW2.

Provides:
  - Data-loading helpers (with optional resize for transfer learning).
  - :class:`LabelSmoothingLoss` — soft-label cross-entropy.
  - :func:`distillation_loss` — Hinton et al. (2015) KD loss.
  - :func:`modified_distillation_loss` — teacher assigns prob. to true class
    only; other classes share the remainder equally.
  - :func:`run_training` — standard training loop (with optional label smoothing).
  - :func:`run_distillation_training` — distillation training loop.
"""

import copy
import os
from typing import Dict, Any, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ─────────────────────────── Data loading ────────────────────────────────────

def get_transforms(params: Dict[str, Any], train: bool = True) -> transforms.Compose:
    """Build the torchvision transform pipeline for CIFAR-10.

    Args:
        params: Configuration dict; must contain ``"mean"``, ``"std"``, and
            optionally ``"resize"`` (int) to resize images before other
            augmentations (used for transfer-learning option 1).
        train: If ``True`` apply random augmentation; otherwise only
            deterministic transforms.

    Returns:
        A :class:`~torchvision.transforms.Compose` pipeline.
    """
    mean, std = params["mean"], params["std"]
    resize: int = params.get("resize", 0)   # 0 means no resize

    ops = []
    if resize:
        if train:
            ops += [transforms.RandomResizedCrop(resize), transforms.RandomHorizontalFlip()]
        else:
            ops += [transforms.Resize(256), transforms.CenterCrop(resize)]
    else:
        if train:
            ops += [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]

    ops += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(ops)


def get_loaders(params: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """Create CIFAR-10 train and validation data loaders.

    Args:
        params: Configuration dict; forwarded to :func:`get_transforms`.

    Returns:
        Tuple of ``(train_loader, val_loader)``.
    """
    train_tf = get_transforms(params, train=True)
    val_tf   = get_transforms(params, train=False)

    train_ds = datasets.CIFAR10(params["data_dir"], train=True,  download=True, transform=train_tf)
    val_ds   = datasets.CIFAR10(params["data_dir"], train=False, download=True, transform=val_tf)

    train_loader = DataLoader(
        train_ds, batch_size=params["batch_size"], shuffle=True,
        num_workers=params["num_workers"], pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=params["batch_size"], shuffle=False,
        num_workers=params["num_workers"], pin_memory=True,
    )
    return train_loader, val_loader


# ─────────────────────────── Loss functions ──────────────────────────────────

class LabelSmoothingLoss(nn.Module):
    """Cross-entropy loss with label smoothing.

    The ground-truth distribution is a mixture of a one-hot encoding and a
    uniform distribution::

        q(k | x) = (1 − ε) * 1[k == y]  +  ε / C

    where ε is the smoothing factor and C is the number of classes.

    Args:
        num_classes: Total number of classes C.
        smoothing: Smoothing factor ε ∈ [0, 1). Default: ``0.1``.

    Shape:
        pred:   (N, C) — raw logits.
        target: (N,)  — integer class indices.
        output: scalar loss.

    Example:
        >>> criterion = LabelSmoothingLoss(num_classes=10, smoothing=0.1)
        >>> loss = criterion(torch.randn(8, 10), torch.randint(10, (8,)))
    """

    def __init__(self, num_classes: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute label-smoothed cross-entropy loss.

        Args:
            pred:   Logits of shape (N, C).
            target: Ground-truth labels of shape (N,).

        Returns:
            Scalar loss tensor.
        """
        confidence = 1.0 - self.smoothing
        smooth_val = self.smoothing / (self.num_classes - 1)

        one_hot = torch.zeros_like(pred).scatter_(1, target.unsqueeze(1), 1.0)
        smooth_one_hot = one_hot * confidence + (1.0 - one_hot) * smooth_val

        log_prob = F.log_softmax(pred, dim=1)
        return -(smooth_one_hot * log_prob).sum(dim=1).mean()


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    alpha: float,
) -> torch.Tensor:
    """Hinton et al. (2015) knowledge-distillation loss.

    Combines a soft-target KL-divergence term (scaled by T²) with a
    standard hard-label cross-entropy term::

        L = α * CE(student, labels) + (1 − α) * T² * KL(student_soft ‖ teacher_soft)

    Args:
        student_logits: Raw logits from the student, shape (N, C).
        teacher_logits: Raw logits from the teacher, shape (N, C).
        labels: Ground-truth integer labels, shape (N,).
        temperature: Distillation temperature T > 0.
        alpha: Weight for the hard-label loss (1−alpha for soft-label loss).

    Returns:
        Scalar loss tensor.
    """
    soft_targets = F.softmax(teacher_logits / temperature, dim=1)
    soft_preds   = F.log_softmax(student_logits / temperature, dim=1)
    kd_loss = F.kl_div(soft_preds, soft_targets, reduction="batchmean") * (temperature ** 2)
    ce_loss = F.cross_entropy(student_logits, labels)
    return alpha * ce_loss + (1.0 - alpha) * kd_loss


def modified_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Modified distillation loss using teacher confidence on the true class.

    The soft target assigns the teacher's predicted probability for the true
    class to that class, then distributes the remainder equally among all
    other classes::

        P(k = y | x) = teacher_prob[y]
        P(k ≠ y | x) = (1 − teacher_prob[y]) / (C − 1)

    This encodes example-level difficulty: easy examples (high teacher
    confidence) produce near-one-hot targets; hard examples produce more
    uniform targets.

    Args:
        student_logits: Raw logits from the student, shape (N, C).
        teacher_logits: Raw logits from the teacher, shape (N, C).
        labels: Ground-truth integer labels, shape (N,).

    Returns:
        Scalar loss tensor.
    """
    num_classes = student_logits.size(1)
    teacher_probs = F.softmax(teacher_logits, dim=1)                   # (N, C)
    true_prob = teacher_probs.gather(1, labels.unsqueeze(1))           # (N, 1)

    other_prob = (1.0 - true_prob) / (num_classes - 1)                # (N, 1)
    soft_targets = other_prob.expand(-1, num_classes).clone()          # (N, C)
    soft_targets.scatter_(1, labels.unsqueeze(1), true_prob)           # set true class

    log_probs = F.log_softmax(student_logits, dim=1)
    return -(soft_targets * log_probs).sum(dim=1).mean()


# ─────────────────────────── Visualization helpers ───────────────────────────

CIFAR10_CLASSES: List[str] = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

COLORS: List[str] = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
]


def _save_training_plot(history: Dict[str, List[float]], save_path: str) -> None:
    """Save loss and accuracy curves to a PNG file next to the checkpoint.

    Args:
        history: Dict with keys ``train_loss``, ``train_acc``,
            ``val_loss``, ``val_acc`` — each a list of per-epoch values.
        save_path: Checkpoint path; the plot is saved with the same stem
            and a ``_curves.png`` suffix.
    """
    import matplotlib.pyplot as plt

    stem = os.path.splitext(save_path)[0]
    plot_path = f"{stem}_curves.png"
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, history["train_loss"], label="Train loss", color="#4363d8")
    ax1.plot(epochs, history["val_loss"],   label="Val loss",   color="#e6194b")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], label="Train acc", color="#4363d8")
    ax2.plot(epochs, history["val_acc"],   label="Val acc",   color="#e6194b")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy curves")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    print(f"[Plot] Training curves saved → {plot_path}")
    plt.close(fig)


def plot_tsne(n_samples: int = 1000, save_path: str = "tsne_cifar10.png") -> None:
    """Run t-SNE on raw CIFAR-10 pixels and save a scatter plot.

    Args:
        n_samples: Number of CIFAR-10 test samples to embed.
        save_path: Output PNG file path.
    """
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    import numpy as np

    print(f"Loading {n_samples} CIFAR-10 samples …")
    tf = transforms.Compose([transforms.ToTensor()])
    ds = datasets.CIFAR10("./data", train=False, download=True, transform=tf)
    indices = np.random.choice(len(ds), n_samples, replace=False)
    X, y = [], []
    for i in indices:
        img, label = ds[i]
        X.append(img.numpy().flatten())
        y.append(label)
    X, y = np.array(X), np.array(y)

    print("Running t-SNE (may take a minute on CPU) …")
    X_2d = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000).fit_transform(X)

    fig, ax = plt.subplots(figsize=(10, 8))
    for cls_idx, cls_name in enumerate(CIFAR10_CLASSES):
        mask = y == cls_idx
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=COLORS[cls_idx], label=cls_name, s=10, alpha=0.7)
    ax.set_title("t-SNE of CIFAR-10 raw pixels")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.legend(markerscale=2, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"[Plot] t-SNE saved → {save_path}")
    plt.show()


def plot_flops_comparison(
    model_a: nn.Module,
    model_b: nn.Module,
    name_a: str,
    name_b: str,
    acc_a: float,
    acc_b: float,
    save_path: str = "flops_comparison.png",
) -> None:
    """Bar chart comparing FLOPs and accuracy for two models.

    Args:
        model_a: First model (e.g. teacher ResNet).
        model_b: Second model (e.g. distilled student).
        name_a: Display name for model A.
        name_b: Display name for model B.
        acc_a: Test accuracy of model A (0–1).
        acc_b: Test accuracy of model B (0–1).
        save_path: Output PNG file path.
    """
    import matplotlib.pyplot as plt

    def _get_flops(model: nn.Module) -> str:
        try:
            from ptflops import get_model_complexity_info
            macs, _ = get_model_complexity_info(
                model, (3, 32, 32), as_strings=True,
                print_per_layer_stat=False, verbose=False,
            )
            return macs
        except ImportError:
            return "N/A"

    flops_a = _get_flops(model_a)
    flops_b = _get_flops(model_b)
    print(f"{name_a}  FLOPs: {flops_a}  Acc: {acc_a*100:.2f}%")
    print(f"{name_b}  FLOPs: {flops_b}  Acc: {acc_b*100:.2f}%")

    names = [name_a, name_b]
    accs  = [acc_a * 100, acc_b * 100]
    colors = ["#4363d8", "#e6194b"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    bars = ax1.bar(names, accs, color=colors, width=0.4)
    for bar, val in zip(bars, accs):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                 f"{val:.2f}%", ha="center", va="bottom", fontweight="bold")
    ax1.set_ylabel("Test Accuracy (%)")
    ax1.set_title("Accuracy comparison")
    ax1.set_ylim(0, 100)
    ax1.grid(axis="y", alpha=0.3)

    ax2.axis("off")
    flops_text = f"{name_a}\n  FLOPs: {flops_a}\n\n{name_b}\n  FLOPs: {flops_b}"
    ax2.text(0.1, 0.5, flops_text, transform=ax2.transAxes,
             fontsize=12, verticalalignment="center",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax2.set_title("FLOPs (ptflops)")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"[Plot] FLOPs comparison saved → {save_path}")
    plt.show()


# ─────────────────────────── Training loops ──────────────────────────────────

def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    log_interval: int,
    epoch: int,
    total_epochs: int,
) -> Tuple[float, float]:
    """Run one training epoch.

    Args:
        model: The model to train.
        loader: Training data loader.
        optimizer: Optimiser instance.
        criterion: Loss function.
        device: Compute device.
        log_interval: Print progress every this many batches.
        epoch: Current epoch number (1-based).
        total_epochs: Total number of epochs.

    Returns:
        Tuple ``(avg_loss, accuracy)`` over the full epoch.
    """
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item() * imgs.size(0)
        correct    += out.argmax(1).eq(labels).sum().item()
        n          += imgs.size(0)

        if (batch_idx + 1) % log_interval == 0:
            print(f"  Epoch {epoch}/{total_epochs} "
                  f"({batch_idx + 1}/{len(loader)})  "
                  f"loss: {total_loss / n:.4f}  acc: {correct / n:.4f}")
    return total_loss / n, correct / n


def _validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate the model on a validation split.

    Args:
        model: The model to evaluate.
        loader: Validation data loader.
        criterion: Loss function.
        device: Compute device.

    Returns:
        Tuple ``(avg_loss, accuracy)``.
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
    params: Dict[str, Any],
    device: torch.device,
) -> None:
    """Standard training loop with optional label smoothing.

    Saves the best checkpoint (by validation accuracy) to ``params["save_path"]``.
    Uses a StepLR scheduler that halves the LR every 20 epochs.

    Args:
        model: Model to train (modified in-place; best weights loaded at end).
        params: Configuration dict from :func:`parameters.get_params`.
        device: Compute device.
    """
    train_loader, val_loader = get_loaders(params)

    if params["label_smoothing"] > 0:
        criterion: nn.Module = LabelSmoothingLoss(
            num_classes=params["num_classes"],
            smoothing=params["label_smoothing"],
        )
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params["learning_rate"],
        weight_decay=params["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    best_acc     = 0.0
    best_weights = None
    history: Dict[str, List[float]] = {
        "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []
    }

    total_epochs = params["epochs"]
    print(f"\n>>> Starting training: {total_epochs} epochs, "
          f"{len(train_loader)} batches/epoch")

    for epoch in range(1, total_epochs + 1):
        print(f"\n--- Epoch {epoch}/{total_epochs}  lr={scheduler.get_last_lr()[0]:.5f} ---")
        tr_loss,  tr_acc  = _train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            params["log_interval"], epoch, total_epochs,
        )
        val_loss, val_acc = _validate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"  Train loss: {tr_loss:.4f}  acc: {tr_acc:.4f}")
        print(f"  Val   loss: {val_loss:.4f}  acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc     = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, params["save_path"])
            print(f"  *** Saved best (val_acc={best_acc:.4f}) → {params['save_path']}")

    if best_weights is not None:
        model.load_state_dict(best_weights)
    print(f"\nTraining done. Best val accuracy: {best_acc:.4f}")

    if params.get("save_plots"):
        _save_training_plot(history, params["save_path"])


def run_distillation_training(
    student: nn.Module,
    teacher: nn.Module,
    params: Dict[str, Any],
    device: torch.device,
) -> None:
    """Knowledge-distillation training loop.

    Supports two distillation modes controlled by ``params["distill_mode"]``:

    * ``"standard"`` — Hinton et al. (2015): soft-target KD with temperature.
    * ``"modified"`` — teacher probability assigned to true class only;
      remaining probability distributed uniformly over other classes.

    The teacher is kept in evaluation mode and its gradients are frozen.
    The best student checkpoint is saved to ``params["save_path"]``.

    Args:
        student: Student model to train.
        teacher: Pre-trained teacher model (weights already loaded).
        params: Configuration dict from :func:`parameters.get_params`.
        device: Compute device.
    """
    train_loader, val_loader = get_loaders(params)

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(
        student.parameters(),
        lr=params["learning_rate"],
        weight_decay=params["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    ce_criterion = nn.CrossEntropyLoss()
    mode = params["distill_mode"]

    best_acc     = 0.0
    best_weights = None
    total_epochs = params["epochs"]

    print(f"\n>>> Starting distillation ({mode}): {total_epochs} epochs, "
          f"{len(train_loader)} batches/epoch")

    for epoch in range(1, total_epochs + 1):
        student.train()
        total_loss, correct, n = 0.0, 0, 0
        print(f"\n--- Epoch {epoch}/{total_epochs}  mode={mode}  lr={scheduler.get_last_lr()[0]:.5f} ---")

        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            student_logits = student(imgs)
            with torch.no_grad():
                teacher_logits = teacher(imgs)

            if mode == "standard":
                loss = distillation_loss(
                    student_logits, teacher_logits, labels,
                    temperature=params["distill_temp"],
                    alpha=params["distill_alpha"],
                )
            else:  # "modified"
                loss = modified_distillation_loss(student_logits, teacher_logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.detach().item() * imgs.size(0)
            correct    += student_logits.argmax(1).eq(labels).sum().item()
            n          += imgs.size(0)

            if (batch_idx + 1) % params["log_interval"] == 0:
                print(f"  Epoch {epoch}/{total_epochs} "
                      f"({batch_idx + 1}/{len(train_loader)})  "
                      f"loss: {total_loss / n:.4f}  acc: {correct / n:.4f}")

        scheduler.step()

        # Validate with plain CE so the number is comparable across runs.
        val_loss, val_acc = _validate(student, val_loader, ce_criterion, device)
        print(f"  Train loss: {total_loss / n:.4f}  acc: {correct / n:.4f}")
        print(f"  Val   loss: {val_loss:.4f}  acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc     = val_acc
            best_weights = copy.deepcopy(student.state_dict())
            torch.save(best_weights, params["save_path"])
            print(f"  *** Saved best (val_acc={best_acc:.4f}) → {params['save_path']}")

    if best_weights is not None:
        student.load_state_dict(best_weights)
    print(f"\nDistillation done. Best val accuracy: {best_acc:.4f}")
