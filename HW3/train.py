"""Training utilities for HW3: Data Augmentation and Adversarial Samples.

Provides:
  - :func:`get_loaders` — standard CIFAR-10 DataLoaders.
  - :func:`get_augmix_loaders` — DataLoaders whose training set returns three
    views per sample (clean + two AugMix-augmented) for JSD consistency training.
  - :class:`AugMixDataset` — Dataset wrapper that generates AugMix views.
  - :func:`augment_and_mix` — core AugMix mixing function (PIL-based).
  - :class:`LabelSmoothingLoss` — soft-label cross-entropy.
  - :func:`distillation_loss` / :func:`modified_distillation_loss` — KD losses.
  - :func:`jsd_consistency_loss` — Jensen-Shannon divergence consistency term.
  - :func:`run_training` — standard training loop.
  - :func:`run_augmix_training` — AugMix training loop with JSD consistency.
  - :func:`run_distillation_training` — knowledge-distillation training loop.
"""

import copy
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from parameters import AugMixParams, DataParams, DistillParams, TrainingParams


# ──────────────────────────── AugMix primitives ──────────────────────────────

def _int_param(level: float, maxval: int) -> int:
    """Convert a normalised level in [0, 1] to an integer parameter.

    Args:
        level: Normalised magnitude in [0, 1].
        maxval: Maximum integer value.

    Returns:
        ``int(level * maxval)``.
    """
    return int(level * maxval)


def _float_param(level: float, maxval: float) -> float:
    """Convert a normalised level in [0, 1] to a float parameter.

    Args:
        level: Normalised magnitude in [0, 1].
        maxval: Maximum float value.

    Returns:
        ``level * maxval``.
    """
    return level * maxval


def _autocontrast(img: Image.Image, _level: float) -> Image.Image:
    """Apply PIL autocontrast (level unused)."""
    return ImageOps.autocontrast(img)


def _equalize(img: Image.Image, _level: float) -> Image.Image:
    """Apply PIL histogram equalisation (level unused)."""
    return ImageOps.equalize(img)


def _posterize(img: Image.Image, level: float) -> Image.Image:
    """Reduce the number of bits for each colour channel.

    Args:
        img: Input PIL image.
        level: Normalised magnitude; maps to 2–4 bits.
    """
    bits = max(1, 4 - _int_param(level, 4))
    return ImageOps.posterize(img, bits)


def _rotate(img: Image.Image, level: float) -> Image.Image:
    """Rotate the image by up to ±30 degrees.

    Args:
        img: Input PIL image.
        level: Normalised magnitude.
    """
    degrees = _float_param(level, 30.0)
    if np.random.random() < 0.5:
        degrees = -degrees
    return img.rotate(degrees, resample=Image.BILINEAR, fillcolor=(128, 128, 128))


def _solarize(img: Image.Image, level: float) -> Image.Image:
    """Invert pixels above a threshold.

    Args:
        img: Input PIL image.
        level: Normalised magnitude; higher → lower threshold.
    """
    threshold = 256 - _int_param(level, 256)
    return ImageOps.solarize(img, threshold)


def _shear_x(img: Image.Image, level: float) -> Image.Image:
    """Apply horizontal shear of up to ±0.3.

    Args:
        img: Input PIL image.
        level: Normalised magnitude.
    """
    shear = _float_param(level, 0.3)
    if np.random.random() < 0.5:
        shear = -shear
    return img.transform(
        img.size, Image.AFFINE, (1, shear, 0, 0, 1, 0),
        resample=Image.BILINEAR, fillcolor=(128, 128, 128),
    )


def _shear_y(img: Image.Image, level: float) -> Image.Image:
    """Apply vertical shear of up to ±0.3.

    Args:
        img: Input PIL image.
        level: Normalised magnitude.
    """
    shear = _float_param(level, 0.3)
    if np.random.random() < 0.5:
        shear = -shear
    return img.transform(
        img.size, Image.AFFINE, (1, 0, 0, shear, 1, 0),
        resample=Image.BILINEAR, fillcolor=(128, 128, 128),
    )


def _translate_x(img: Image.Image, level: float) -> Image.Image:
    """Translate the image horizontally by up to ±10 pixels.

    Args:
        img: Input PIL image.
        level: Normalised magnitude.
    """
    pixels = _int_param(level, 10)
    if np.random.random() < 0.5:
        pixels = -pixels
    return img.transform(
        img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0),
        resample=Image.BILINEAR, fillcolor=(128, 128, 128),
    )


def _translate_y(img: Image.Image, level: float) -> Image.Image:
    """Translate the image vertically by up to ±10 pixels.

    Args:
        img: Input PIL image.
        level: Normalised magnitude.
    """
    pixels = _int_param(level, 10)
    if np.random.random() < 0.5:
        pixels = -pixels
    return img.transform(
        img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels),
        resample=Image.BILINEAR, fillcolor=(128, 128, 128),
    )


def _color(img: Image.Image, level: float) -> Image.Image:
    """Randomly adjust colour saturation.

    Args:
        img: Input PIL image.
        level: Normalised magnitude; factor in [0.1, 1.9].
    """
    factor = 1.0 + _float_param(level, 0.9)
    if np.random.random() < 0.5:
        factor = 2.0 - factor
    return ImageEnhance.Color(img).enhance(max(0.1, factor))


def _contrast(img: Image.Image, level: float) -> Image.Image:
    """Randomly adjust contrast.

    Args:
        img: Input PIL image.
        level: Normalised magnitude; factor in [0.1, 1.9].
    """
    factor = 1.0 + _float_param(level, 0.9)
    if np.random.random() < 0.5:
        factor = 2.0 - factor
    return ImageEnhance.Contrast(img).enhance(max(0.1, factor))


def _brightness(img: Image.Image, level: float) -> Image.Image:
    """Randomly adjust brightness.

    Args:
        img: Input PIL image.
        level: Normalised magnitude; factor in [0.1, 1.9].
    """
    factor = 1.0 + _float_param(level, 0.9)
    if np.random.random() < 0.5:
        factor = 2.0 - factor
    return ImageEnhance.Brightness(img).enhance(max(0.1, factor))


def _sharpness(img: Image.Image, level: float) -> Image.Image:
    """Randomly adjust sharpness.

    Args:
        img: Input PIL image.
        level: Normalised magnitude; factor in [0.1, 1.9].
    """
    factor = 1.0 + _float_param(level, 0.9)
    if np.random.random() < 0.5:
        factor = 2.0 - factor
    return ImageEnhance.Sharpness(img).enhance(max(0.1, factor))


#: All augmentation operations available to AugMix.
_AUGMENTATIONS = [
    _autocontrast, _equalize, _posterize, _rotate, _solarize,
    _shear_x, _shear_y, _translate_x, _translate_y,
    _color, _contrast, _brightness, _sharpness,
]


def augment_and_mix(
    img: Image.Image,
    severity: int = 3,
    width: int = 3,
    depth: int = -1,
    alpha: float = 1.0,
) -> np.ndarray:
    """Apply AugMix to a PIL image and return a float32 array in [0, 1].

    Generates *width* independent augmentation chains of random depth
    (unless *depth* is fixed), mixes them using Dirichlet weights, and
    blends the mixture with the original image using a Beta weight.

    Args:
        img: Input PIL image (uint8, H×W×3).
        severity: Maximum augmentation magnitude (1–10).
        width: Number of parallel augmentation chains.
        depth: Fixed chain depth; ``-1`` samples uniformly from 1–3.
        alpha: Concentration parameter for Dirichlet and Beta distributions.

    Returns:
        Mixed image as a float32 numpy array of shape (H, W, 3) in [0, 1].
    """
    ws = np.float32(np.random.dirichlet([alpha] * width))
    m  = np.float32(np.random.beta(alpha, alpha))

    orig = np.array(img, dtype=np.float32) / 255.0
    mix  = np.zeros_like(orig)

    for i in range(width):
        img_aug = img.copy()
        chain_depth = depth if depth > 0 else np.random.randint(1, 4)
        for _ in range(chain_depth):
            op    = np.random.choice(_AUGMENTATIONS)
            level = np.random.uniform(0.1, severity / 10.0)
            img_aug = op(img_aug, level)
        mix += ws[i] * (np.array(img_aug, dtype=np.float32) / 255.0)

    mixed = (1.0 - m) * orig + m * mix
    return np.clip(mixed, 0.0, 1.0)


# ──────────────────────────── AugMix Dataset ─────────────────────────────────


class AugMixDataset(Dataset):
    """Dataset wrapper that produces three views per sample for AugMix training.

    Wraps a CIFAR-10 ``Dataset`` (with ``transform=None``) and returns a
    tuple ``(x_clean, x_aug1, x_aug2, label)`` where:

    * ``x_clean`` is the image with standard augmentation (random crop + flip)
      followed by normalisation.
    * ``x_aug1`` and ``x_aug2`` are two independent AugMix versions of the
      *un-augmented* image, also normalised.

    Args:
        base_dataset: A :class:`~torchvision.datasets.CIFAR10` dataset loaded
            **without** any transform (``transform=None``).
        augmix_params: :class:`~parameters.AugMixParams` controlling the
            AugMix augmentation chain.
        mean: Per-channel mean for normalisation.
        std: Per-channel std for normalisation.

    Returns (per item):
        Tuple ``(x_clean, x_aug1, x_aug2, label)`` of three tensors (C, H, W)
        and an integer label.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        augmix_params: AugMixParams,
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
    ) -> None:
        self.base_dataset  = base_dataset
        self.augmix_params = augmix_params
        self._normalize    = transforms.Normalize(mean, std)
        self._to_tensor    = transforms.ToTensor()
        self._basic_aug    = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Return (x_clean, x_aug1, x_aug2, label) for sample *idx*."""
        img, label = self.base_dataset[idx]    # PIL image, int

        # Clean view: standard spatial augmentation then normalise
        x_clean = self._normalize(self._to_tensor(self._basic_aug(img)))

        # AugMix views: mix in PIL space then normalise
        x_aug1 = self._normalize(self._augmix_to_tensor(img))
        x_aug2 = self._normalize(self._augmix_to_tensor(img))

        return x_clean, x_aug1, x_aug2, label

    def _augmix_to_tensor(self, img: Image.Image) -> torch.Tensor:
        """Apply AugMix and convert the result to a (C, H, W) float tensor.

        Args:
            img: Source PIL image.

        Returns:
            Float32 tensor of shape (3, 32, 32) in [0, 1] before normalisation.
        """
        mixed = augment_and_mix(
            img,
            severity=self.augmix_params.severity,
            width=self.augmix_params.width,
            depth=self.augmix_params.depth,
            alpha=self.augmix_params.alpha,
        )
        # mixed is (H, W, 3) float32 in [0, 1]
        return torch.from_numpy(mixed).permute(2, 0, 1)


# ──────────────────────────── Data loading ───────────────────────────────────


def get_transforms(
    data_params: DataParams,
    train: bool = True,
) -> transforms.Compose:
    """Build the standard CIFAR-10 transform pipeline.

    Args:
        data_params: :class:`~parameters.DataParams` with mean/std.
        train: If ``True`` apply random augmentation; otherwise deterministic.

    Returns:
        A :class:`~torchvision.transforms.Compose` pipeline.
    """
    mean, std = data_params.mean, data_params.std
    ops = []
    if train:
        ops += [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    ops += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return transforms.Compose(ops)


def get_loaders(data_params: DataParams) -> Tuple[DataLoader, DataLoader]:
    """Create standard CIFAR-10 train and validation DataLoaders.

    Args:
        data_params: :class:`~parameters.DataParams`.

    Returns:
        Tuple ``(train_loader, val_loader)``.
    """
    train_tf = get_transforms(data_params, train=True)
    val_tf   = get_transforms(data_params, train=False)

    train_ds = datasets.CIFAR10(data_params.data_dir, train=True,  download=True, transform=train_tf)
    val_ds   = datasets.CIFAR10(data_params.data_dir, train=False, download=True, transform=val_tf)

    train_loader = DataLoader(
        train_ds, batch_size=data_params.batch_size, shuffle=True,
        num_workers=data_params.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=data_params.batch_size, shuffle=False,
        num_workers=data_params.num_workers, pin_memory=True,
    )
    return train_loader, val_loader


def get_augmix_loaders(
    data_params: DataParams,
    augmix_params: AugMixParams,
) -> Tuple[DataLoader, DataLoader]:
    """Create CIFAR-10 DataLoaders with AugMix training augmentation.

    The training loader returns ``(x_clean, x_aug1, x_aug2, label)`` per
    batch.  The validation loader returns ``(x, label)`` as usual.

    Args:
        data_params: :class:`~parameters.DataParams`.
        augmix_params: :class:`~parameters.AugMixParams`.

    Returns:
        Tuple ``(train_loader, val_loader)``.
    """
    # Load without transform to keep PIL images for AugMix
    raw_train_ds = datasets.CIFAR10(data_params.data_dir, train=True,
                                    download=True, transform=None)
    val_tf = get_transforms(data_params, train=False)
    val_ds = datasets.CIFAR10(data_params.data_dir, train=False,
                               download=True, transform=val_tf)

    augmix_ds = AugMixDataset(raw_train_ds, augmix_params,
                               data_params.mean, data_params.std)

    train_loader = DataLoader(
        augmix_ds, batch_size=data_params.batch_size, shuffle=True,
        num_workers=data_params.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=data_params.batch_size, shuffle=False,
        num_workers=data_params.num_workers, pin_memory=True,
    )
    return train_loader, val_loader


# ──────────────────────────── Loss functions ─────────────────────────────────


class LabelSmoothingLoss(nn.Module):
    """Cross-entropy loss with label smoothing.

    The ground-truth distribution is a mixture of a one-hot encoding and a
    uniform distribution::

        q(k | x) = (1 − ε) * 1[k == y]  +  ε / C

    Args:
        num_classes: Total number of classes C.
        smoothing: Smoothing factor ε ∈ [0, 1). Default: ``0.1``.

    Shape:
        pred:   (N, C) — raw logits.
        target: (N,)  — integer class indices.
        output: scalar loss.
    """

    def __init__(self, num_classes: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smoothing   = smoothing

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
        one_hot    = torch.zeros_like(pred).scatter_(1, target.unsqueeze(1), 1.0)
        smooth_lbl = one_hot * confidence + (1.0 - one_hot) * smooth_val
        log_prob   = F.log_softmax(pred, dim=1)
        return -(smooth_lbl * log_prob).sum(dim=1).mean()


def jsd_consistency_loss(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    logits_c: torch.Tensor,
) -> torch.Tensor:
    """Jensen-Shannon divergence consistency loss for three views.

    Encourages the model to produce similar predictions across augmented views
    of the same image::

        JSD(p_a, p_b, p_c) = (1/3) * [KL(p_a ‖ m) + KL(p_b ‖ m) + KL(p_c ‖ m)]

    where m = (p_a + p_b + p_c) / 3.

    Args:
        logits_a: Logits for the clean / first view, shape (N, C).
        logits_b: Logits for the first AugMix view, shape (N, C).
        logits_c: Logits for the second AugMix view, shape (N, C).

    Returns:
        Scalar JSD loss (non-negative).
    """
    p_a = F.softmax(logits_a, dim=1)
    p_b = F.softmax(logits_b, dim=1)
    p_c = F.softmax(logits_c, dim=1)

    m     = ((p_a + p_b + p_c) / 3.0).clamp(1e-8)
    log_m = m.log()

    # F.kl_div(log_m, p_x) = KL(p_x ‖ m) = Σ p_x * (log p_x − log m)
    loss = (
        F.kl_div(log_m, p_a, reduction="batchmean")
        + F.kl_div(log_m, p_b, reduction="batchmean")
        + F.kl_div(log_m, p_c, reduction="batchmean")
    ) / 3.0
    return loss


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
    class to that class, then distributes the remainder equally::

        P(k = y | x) = teacher_prob[y]
        P(k ≠ y | x) = (1 − teacher_prob[y]) / (C − 1)

    Args:
        student_logits: Raw logits from the student, shape (N, C).
        teacher_logits: Raw logits from the teacher, shape (N, C).
        labels: Ground-truth integer labels, shape (N,).

    Returns:
        Scalar loss tensor.
    """
    num_classes   = student_logits.size(1)
    teacher_probs = F.softmax(teacher_logits, dim=1)
    true_prob     = teacher_probs.gather(1, labels.unsqueeze(1))
    other_prob    = (1.0 - true_prob) / (num_classes - 1)
    soft_targets  = other_prob.expand(-1, num_classes).clone()
    soft_targets.scatter_(1, labels.unsqueeze(1), true_prob)
    log_probs = F.log_softmax(student_logits, dim=1)
    return -(soft_targets * log_probs).sum(dim=1).mean()


# ──────────────────────────── Helpers ────────────────────────────────────────


def _save_training_plot(
    history: Dict[str, List[float]],
    save_path: str,
) -> None:
    """Save loss and accuracy curves to a PNG file.

    Args:
        history: Dict with keys ``train_loss``, ``train_acc``,
            ``val_loss``, ``val_acc`` — each a list of per-epoch values.
        save_path: Checkpoint path; the plot is saved with a ``_curves.png``
            suffix derived from the checkpoint stem.
    """
    import matplotlib.pyplot as plt

    stem      = os.path.splitext(save_path)[0]
    plot_path = f"{stem}_curves.png"
    epochs    = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, history["train_loss"], label="Train loss", color="#4363d8")
    ax1.plot(epochs, history["val_loss"],   label="Val loss",   color="#e6194b")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Loss curves"); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], label="Train acc", color="#4363d8")
    ax2.plot(epochs, history["val_acc"],   label="Val acc",   color="#e6194b")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy curves"); ax2.legend(); ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    print(f"[Plot] Training curves saved → {plot_path}")
    plt.close(fig)


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
    """Run one standard training epoch.

    Args:
        model: The model to train.
        loader: Training DataLoader yielding ``(images, labels)``.
        optimizer: Optimizer instance.
        criterion: Loss function.
        device: Compute device.
        log_interval: Print progress every this many batches.
        epoch: Current epoch (1-based).
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
                  f"({batch_idx+1}/{len(loader)})  "
                  f"loss: {total_loss/n:.4f}  acc: {correct/n:.4f}")
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
        loader: Validation DataLoader.
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


# ──────────────────────────── Training loops ─────────────────────────────────


def run_training(
    model: nn.Module,
    data_params: DataParams,
    training_params: TrainingParams,
    device: torch.device,
) -> None:
    """Standard training loop (no AugMix).

    Saves the best checkpoint (by validation accuracy) to
    ``training_params.save_path``.  Uses an Adam optimizer with StepLR
    scheduler that halves the LR every 20 epochs.

    Args:
        model: Model to train (best weights loaded at end).
        data_params: :class:`~parameters.DataParams`.
        training_params: :class:`~parameters.TrainingParams`.
        device: Compute device.
    """
    train_loader, val_loader = get_loaders(data_params)
    criterion  = nn.CrossEntropyLoss()
    optimizer  = torch.optim.Adam(
        model.parameters(), lr=training_params.lr, weight_decay=training_params.weight_decay
    )
    scheduler  = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    best_acc   = 0.0
    best_state = None
    history: Dict[str, List[float]] = {
        "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []
    }

    print(f"\n>>> Standard training: {training_params.epochs} epochs")
    for epoch in range(1, training_params.epochs + 1):
        print(f"\n--- Epoch {epoch}/{training_params.epochs}  "
              f"lr={scheduler.get_last_lr()[0]:.5f} ---")
        tr_loss, tr_acc   = _train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            training_params.log_interval, epoch, training_params.epochs,
        )
        val_loss, val_acc = _validate(model, val_loader, criterion, device)
        scheduler.step()

        for k, v in [("train_loss", tr_loss), ("train_acc", tr_acc),
                     ("val_loss", val_loss), ("val_acc", val_acc)]:
            history[k].append(v)

        print(f"  Train → loss: {tr_loss:.4f}  acc: {tr_acc:.4f}")
        print(f"  Val   → loss: {val_loss:.4f}  acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc   = val_acc
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, training_params.save_path)
            print(f"  *** Best saved (val_acc={best_acc:.4f}) → {training_params.save_path}")

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"\nTraining done. Best val accuracy: {best_acc:.4f}")
    if training_params.save_plots:
        _save_training_plot(history, training_params.save_path)


def run_augmix_training(
    model: nn.Module,
    data_params: DataParams,
    training_params: TrainingParams,
    augmix_params: AugMixParams,
    device: torch.device,
) -> None:
    """AugMix training loop with JSD consistency regularisation.

    For each mini-batch three views are produced (clean, aug1, aug2).  The
    loss is::

        L = CE(model(x_clean), y) + λ * JSD(p_clean, p_aug1, p_aug2)

    Saves the best checkpoint to ``training_params.save_path``.

    Args:
        model: Model to train.
        data_params: :class:`~parameters.DataParams`.
        training_params: :class:`~parameters.TrainingParams`.
        augmix_params: :class:`~parameters.AugMixParams`.
        device: Compute device.
    """
    train_loader, val_loader = get_augmix_loaders(data_params, augmix_params)
    ce_criterion = nn.CrossEntropyLoss()
    optimizer    = torch.optim.Adam(
        model.parameters(), lr=training_params.lr, weight_decay=training_params.weight_decay
    )
    scheduler    = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    best_acc     = 0.0
    best_state   = None
    history: Dict[str, List[float]] = {
        "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []
    }
    lam = augmix_params.jsd_lambda

    print(f"\n>>> AugMix training: {training_params.epochs} epochs  λ_jsd={lam}")
    for epoch in range(1, training_params.epochs + 1):
        model.train()
        total_loss, correct, n = 0.0, 0, 0
        print(f"\n--- Epoch {epoch}/{training_params.epochs}  "
              f"lr={scheduler.get_last_lr()[0]:.5f} ---")

        for batch_idx, (x_clean, x_aug1, x_aug2, labels) in enumerate(train_loader):
            x_clean = x_clean.to(device)
            x_aug1  = x_aug1.to(device)
            x_aug2  = x_aug2.to(device)
            labels  = labels.to(device)

            optimizer.zero_grad()
            logits_clean = model(x_clean)
            logits_aug1  = model(x_aug1)
            logits_aug2  = model(x_aug2)

            ce_loss  = ce_criterion(logits_clean, labels)
            jsd_loss = jsd_consistency_loss(logits_clean, logits_aug1, logits_aug2)
            loss     = ce_loss + lam * jsd_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.detach().item() * x_clean.size(0)
            correct    += logits_clean.argmax(1).eq(labels).sum().item()
            n          += x_clean.size(0)

            if (batch_idx + 1) % training_params.log_interval == 0:
                print(f"  Epoch {epoch}/{training_params.epochs} "
                      f"({batch_idx+1}/{len(train_loader)})  "
                      f"loss: {total_loss/n:.4f}  acc: {correct/n:.4f}")

        scheduler.step()
        val_loss, val_acc = _validate(model, val_loader, ce_criterion, device)
        for k, v in [("train_loss", total_loss/n), ("train_acc", correct/n),
                     ("val_loss", val_loss), ("val_acc", val_acc)]:
            history[k].append(v)

        print(f"  Train → loss: {total_loss/n:.4f}  acc: {correct/n:.4f}")
        print(f"  Val   → loss: {val_loss:.4f}        acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc   = val_acc
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, training_params.save_path)
            print(f"  *** Best saved (val_acc={best_acc:.4f}) → {training_params.save_path}")

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"\nAugMix training done. Best val accuracy: {best_acc:.4f}")
    if training_params.save_plots:
        _save_training_plot(history, training_params.save_path)


def run_distillation_training(
    student: nn.Module,
    teacher: nn.Module,
    data_params: DataParams,
    training_params: TrainingParams,
    distill_params: DistillParams,
    device: torch.device,
) -> None:
    """Knowledge-distillation training loop.

    The teacher is kept in evaluation mode with frozen gradients.  Two modes
    are supported:

    * ``"standard"`` — Hinton et al. (2015) soft-target KD with temperature.
    * ``"modified"`` — teacher probability assigned to true class only;
      remainder distributed uniformly over other classes.

    Args:
        student: Student model to train.
        teacher: Pre-trained teacher model (already loaded).
        data_params: :class:`~parameters.DataParams`.
        training_params: :class:`~parameters.TrainingParams`.
        distill_params: :class:`~parameters.DistillParams`.
        device: Compute device.
    """
    train_loader, val_loader = get_loaders(data_params)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    optimizer    = torch.optim.Adam(
        student.parameters(), lr=training_params.lr, weight_decay=training_params.weight_decay
    )
    scheduler    = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    ce_criterion = nn.CrossEntropyLoss()
    mode         = distill_params.mode
    best_acc     = 0.0
    best_state   = None

    print(f"\n>>> Distillation ({mode}): {training_params.epochs} epochs")
    for epoch in range(1, training_params.epochs + 1):
        student.train()
        total_loss, correct, n = 0.0, 0, 0
        print(f"\n--- Epoch {epoch}/{training_params.epochs}  "
              f"mode={mode}  lr={scheduler.get_last_lr()[0]:.5f} ---")

        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            s_logits = student(imgs)
            with torch.no_grad():
                t_logits = teacher(imgs)

            if mode == "standard":
                loss = distillation_loss(
                    s_logits, t_logits, labels,
                    temperature=distill_params.temperature,
                    alpha=distill_params.alpha,
                )
            else:
                loss = modified_distillation_loss(s_logits, t_logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.detach().item() * imgs.size(0)
            correct    += s_logits.argmax(1).eq(labels).sum().item()
            n          += imgs.size(0)

            if (batch_idx + 1) % training_params.log_interval == 0:
                print(f"  Epoch {epoch}/{training_params.epochs} "
                      f"({batch_idx+1}/{len(train_loader)})  "
                      f"loss: {total_loss/n:.4f}  acc: {correct/n:.4f}")

        scheduler.step()
        val_loss, val_acc = _validate(student, val_loader, ce_criterion, device)
        print(f"  Train → loss: {total_loss/n:.4f}  acc: {correct/n:.4f}")
        print(f"  Val   → loss: {val_loss:.4f}  acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc   = val_acc
            best_state = copy.deepcopy(student.state_dict())
            torch.save(best_state, distill_params.student_path)
            print(f"  *** Best saved (val_acc={best_acc:.4f}) → {distill_params.student_path}")

    if best_state is not None:
        student.load_state_dict(best_state)
    print(f"\nDistillation done. Best val accuracy: {best_acc:.4f}")
