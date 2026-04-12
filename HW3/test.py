"""Evaluation utilities for HW3: Data Augmentation and Adversarial Samples.

Provides:
  - :func:`run_test` — standard CIFAR-10 test-set evaluation.
  - :func:`run_cifar10c_test` — evaluate on every CIFAR-10-C corruption / severity.
  - :func:`pgd_attack` — generate PGD adversarial examples (L∞ or L2 norm).
  - :func:`run_pgd_test` — measure accuracy under PGD-20 L∞ and L2 attacks.
  - :class:`GradCAM` — Grad-CAM heat-map generator.
  - :func:`visualize_gradcam` — plot Grad-CAM overlays on adversarial samples.
  - :func:`visualize_tsne_adversarial` — t-SNE scatter of clean vs. adversarial
    feature embeddings.
  - :func:`plot_cifar10c_results` — bar chart of per-corruption accuracies.
"""

import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

from parameters import DataParams, PGDParams, VisParams
from train import get_transforms

# ──────────────────────────── Constants ─────────────────────────────────────

CIFAR10_CLASSES: List[str] = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

#: The 15 standard CIFAR-10-C corruption types.
CORRUPTIONS: List[str] = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]

# ──────────────────────────── Standard test ─────────────────────────────────


@torch.no_grad()
def run_test(
    model: nn.Module,
    data_params: DataParams,
    device: torch.device,
    checkpoint_path: Optional[str] = None,
) -> float:
    """Evaluate *model* on the clean CIFAR-10 test set.

    Args:
        model: Model to evaluate (optionally loads weights from checkpoint).
        data_params: :class:`~parameters.DataParams`.
        device: Compute device.
        checkpoint_path: If provided, load weights from this ``.pth`` file
            before evaluation.

    Returns:
        Overall accuracy in [0, 1].
    """
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    tf     = get_transforms(data_params, train=False)
    ds     = datasets.CIFAR10(data_params.data_dir, train=False, download=True, transform=tf)
    loader = DataLoader(ds, batch_size=data_params.batch_size,
                        shuffle=False, num_workers=data_params.num_workers)

    correct, n                   = 0, 0
    class_correct: List[int]     = [0] * data_params.num_classes
    class_total:   List[int]     = [0] * data_params.num_classes

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(1)
        correct += preds.eq(labels).sum().item()
        n       += labels.size(0)
        for p, t in zip(preds.tolist(), labels.tolist()):
            class_correct[t] += int(p == t)
            class_total[t]   += 1

    print("\n=== Clean Test Results ===")
    print(f"Overall accuracy: {correct / n:.4f}  ({correct}/{n})\n")
    for i, name in enumerate(CIFAR10_CLASSES):
        acc = class_correct[i] / class_total[i]
        print(f"  {name:12s}: {acc:.4f}  ({class_correct[i]}/{class_total[i]})")

    return correct / n


# ──────────────────────────── CIFAR-10-C ─────────────────────────────────────


def _numpy_to_loader(
    data: np.ndarray,
    labels: np.ndarray,
    data_params: DataParams,
) -> DataLoader:
    """Build a DataLoader from raw uint8 numpy arrays.

    Applies the standard CIFAR-10 normalisation (no spatial augmentation).

    Args:
        data:   Image array of shape (N, 32, 32, 3), dtype uint8.
        labels: Label array of shape (N,), dtype int64.
        data_params: :class:`~parameters.DataParams` with mean/std.

    Returns:
        A :class:`~torch.utils.data.DataLoader`.
    """
    mean = torch.tensor(data_params.mean).view(3, 1, 1)
    std  = torch.tensor(data_params.std).view(3, 1, 1)

    # (N, H, W, C) uint8 → (N, C, H, W) float32 in [0, 1]
    x = torch.from_numpy(data).permute(0, 3, 1, 2).float() / 255.0
    x = (x - mean) / std
    y = torch.from_numpy(labels).long()

    return DataLoader(
        TensorDataset(x, y),
        batch_size=data_params.batch_size,
        shuffle=False,
        num_workers=0,          # tensors already in memory
        pin_memory=True,
    )


@torch.no_grad()
def _eval_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Return accuracy of *model* on *loader*.

    Args:
        model: Model in evaluation mode.
        loader: DataLoader.
        device: Compute device.

    Returns:
        Accuracy in [0, 1].
    """
    correct, n = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        correct += model(imgs).argmax(1).eq(labels).sum().item()
        n       += labels.size(0)
    return correct / n


def run_cifar10c_test(
    model: nn.Module,
    data_params: DataParams,
    device: torch.device,
    vis_params: Optional[VisParams] = None,
) -> Dict[str, List[float]]:
    """Evaluate *model* on every CIFAR-10-C corruption type and severity level.

    Expects the following directory layout::

        <data_params.cifar10c_dir>/
            labels.npy              # (50000,) int64
            gaussian_noise.npy      # (50000, 32, 32, 3) uint8
            shot_noise.npy
            ...                     # one file per corruption

    Each corruption file contains 5 severity levels × 10 000 images stacked
    along the first axis.

    Args:
        model: Model to evaluate (must already be on *device* and in eval mode).
        data_params: :class:`~parameters.DataParams`.
        device: Compute device.
        vis_params: If provided, save a bar-chart to ``vis_params.robustness_save``.

    Returns:
        Dict mapping corruption name → list of 5 accuracies (one per severity).
    """
    c_dir = data_params.cifar10c_dir
    labels_path = os.path.join(c_dir, "labels.npy")

    if not os.path.isfile(labels_path):
        raise FileNotFoundError(
            f"CIFAR-10-C labels not found at {labels_path}.\n"
            "Download the dataset from https://zenodo.org/record/2535967 and "
            f"extract to {c_dir}/"
        )

    all_labels = np.load(labels_path)   # (50 000,) — same for all corruptions
    results: Dict[str, List[float]] = {}
    model.eval()

    print("\n=== CIFAR-10-C Robustness Evaluation ===")
    for corruption in CORRUPTIONS:
        fpath = os.path.join(c_dir, f"{corruption}.npy")
        if not os.path.isfile(fpath):
            print(f"  [SKIP] {corruption} — file not found")
            continue

        data = np.load(fpath)   # (50 000, 32, 32, 3) uint8
        accs: List[float] = []
        for severity in range(1, 6):
            start = (severity - 1) * 10_000
            end   = severity * 10_000
            loader = _numpy_to_loader(
                data[start:end], all_labels[start:end], data_params
            )
            acc = _eval_loader(model, loader, device)
            accs.append(acc)

        results[corruption] = accs
        mean_acc = np.mean(accs)
        print(f"  {corruption:20s}: {[f'{a:.3f}' for a in accs]}  mean={mean_acc:.3f}")

    if results:
        mean_corruption_acc = np.mean([np.mean(v) for v in results.values()])
        print(f"\nMean Corruption Accuracy (mCA): {mean_corruption_acc:.4f}")

    if vis_params is not None and results:
        _plot_cifar10c_results(results, vis_params.robustness_save)

    return results


def _plot_cifar10c_results(
    results: Dict[str, List[float]],
    save_path: str,
) -> None:
    """Bar chart of mean per-corruption accuracy across all severities.

    Args:
        results: Output of :func:`run_cifar10c_test`.
        save_path: Output PNG path.
    """
    names = list(results.keys())
    means = [np.mean(v) for v in results.values()]

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(names, [m * 100 for m in means], color="#4363d8", alpha=0.85)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val*100:.1f}%",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Mean Accuracy (%) across severities")
    ax.set_title("CIFAR-10-C Robustness by Corruption Type")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=35, ha="right", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"[Plot] CIFAR-10-C bar chart saved → {save_path}")
    plt.close(fig)


# ──────────────────────────── PGD Attack ─────────────────────────────────────


def pgd_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    alpha: float,
    num_steps: int,
    norm: str = "linf",
    random_start: bool = True,
) -> torch.Tensor:
    """Generate adversarial examples using the PGD attack.

    Implements the Madry et al. (2018) PGD attack for both L∞ and L2 threat
    models.  The attack is applied in the (already-normalised) input space;
    ``epsilon`` is interpreted in that space.

    Args:
        model: Model to attack (must be in eval mode; gradients will be
            temporarily enabled).
        images: Clean input images, shape (N, C, H, W).
        labels: True class labels, shape (N,).
        epsilon: Perturbation budget (L∞ or L2 ball radius).
        alpha: Step size per iteration.
        num_steps: Number of PGD gradient-ascent steps.
        norm: Threat model — ``"linf"`` or ``"l2"``.
        random_start: Initialise with a random perturbation inside the ball.

    Returns:
        Adversarial images of shape (N, C, H, W), detached from the
        computation graph.
    """
    images = images.detach().clone()

    if random_start:
        if norm == "linf":
            delta = torch.empty_like(images).uniform_(-epsilon, epsilon)
        else:
            delta = torch.randn_like(images)
            norms = delta.view(delta.size(0), -1).norm(2, dim=1).view(-1, 1, 1, 1)
            delta = delta * (epsilon / (norms + 1e-8))
    else:
        delta = torch.zeros_like(images)

    for _ in range(num_steps):
        delta.requires_grad_(True)
        with torch.enable_grad():
            loss = F.cross_entropy(model(images + delta), labels)
        grad = torch.autograd.grad(loss, delta)[0].detach()

        with torch.no_grad():
            if norm == "linf":
                delta = delta.detach() + alpha * grad.sign()
                delta = torch.clamp(delta, -epsilon, epsilon)
            else:  # l2
                grad_norms = grad.view(grad.size(0), -1).norm(2, dim=1).view(-1, 1, 1, 1)
                delta      = delta.detach() + alpha * grad / (grad_norms + 1e-8)
                d_norms    = delta.view(delta.size(0), -1).norm(2, dim=1).view(-1, 1, 1, 1)
                factor     = torch.clamp(epsilon / (d_norms + 1e-8), max=1.0)
                delta      = delta * factor

    return (images + delta).detach()


def run_pgd_test(
    model: nn.Module,
    data_params: DataParams,
    pgd_params: PGDParams,
    device: torch.device,
    tag: str = "",
) -> Tuple[float, float]:
    """Evaluate *model* under PGD-20 L∞ and L2 attacks.

    Args:
        model: Model to evaluate.
        data_params: :class:`~parameters.DataParams`.
        pgd_params: :class:`~parameters.PGDParams`.
        device: Compute device.
        tag: Optional label prepended to printed output (e.g. ``"Standard"``).

    Returns:
        Tuple ``(acc_linf, acc_l2)`` — adversarial accuracy under each norm.
    """
    model.eval()
    tf     = get_transforms(data_params, train=False)
    ds     = datasets.CIFAR10(data_params.data_dir, train=False, download=True, transform=tf)

    n_samples = pgd_params.n_samples if pgd_params.n_samples > 0 else len(ds)
    indices   = np.random.choice(len(ds), min(n_samples, len(ds)), replace=False)
    subset    = torch.utils.data.Subset(ds, indices)
    loader    = DataLoader(subset, batch_size=data_params.batch_size,
                           shuffle=False, num_workers=data_params.num_workers)

    header = f"[{tag}] " if tag else ""

    def _attack_and_eval(norm: str, eps: float, step: float) -> float:
        """Run PGD with the given norm and return accuracy."""
        correct, n = 0, 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            adv = pgd_attack(
                model, imgs, labels, epsilon=eps, alpha=step,
                num_steps=pgd_params.num_steps, norm=norm,
                random_start=pgd_params.random_start,
            )
            preds    = model(adv).argmax(1)
            correct += preds.eq(labels).sum().item()
            n       += labels.size(0)
        return correct / n

    print(f"\n=== {header}PGD-{pgd_params.num_steps} Adversarial Accuracy "
          f"({n_samples} samples) ===")

    acc_linf = _attack_and_eval("linf", pgd_params.epsilon_linf, pgd_params.alpha_linf)
    print(f"  L∞  (ε={pgd_params.epsilon_linf:.4f}): {acc_linf:.4f}")

    acc_l2 = _attack_and_eval("l2", pgd_params.epsilon_l2, pgd_params.alpha_l2)
    print(f"  L2  (ε={pgd_params.epsilon_l2:.4f}): {acc_l2:.4f}")

    return acc_linf, acc_l2


# ──────────────────────────── Grad-CAM ───────────────────────────────────────


class GradCAM:
    """Gradient-weighted Class Activation Mapping (Grad-CAM).

    Hooks into a target convolutional layer to extract activations and
    back-propagated gradients, then computes a spatial importance map.

    Args:
        model: The model to visualise.
        target_layer: The :class:`~torch.nn.Module` layer to hook (e.g.
            ``model.layer4`` for ResNet).

    Example:
        >>> cam = GradCAM(model, model.layer4)
        >>> heat_maps = cam(images)   # shape (N, H, W)
        >>> cam.remove()
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model        = model
        self._features:   Optional[torch.Tensor] = None
        self._gradients:  Optional[torch.Tensor] = None

        self._fwd_hook = target_layer.register_forward_hook(self._save_features)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradients)

    # ── Hooks ────────────────────────────────────────────────────────────────

    def _save_features(
        self,
        _module: nn.Module,
        _inp: Tuple,
        output: torch.Tensor,
    ) -> None:
        """Cache the layer's forward output."""
        self._features = output

    def _save_gradients(
        self,
        _module: nn.Module,
        _grad_in: Tuple,
        grad_out: Tuple,
    ) -> None:
        """Cache the layer's gradient (first element of grad_out)."""
        self._gradients = grad_out[0]

    # ── Computation ──────────────────────────────────────────────────────────

    def __call__(
        self,
        x: torch.Tensor,
        class_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute Grad-CAM heat maps for a batch of images.

        Args:
            x: Input batch, shape (N, C, H, W).
            class_indices: Target class indices, shape (N,).  If ``None``,
                uses the predicted class (argmax).

        Returns:
            Normalised heat maps, shape (N, H_feat, W_feat) in [0, 1].
        """
        self.model.eval()
        x = x.requires_grad_(False)

        output = self.model(x)
        if class_indices is None:
            class_indices = output.argmax(dim=1)

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        for i, c in enumerate(class_indices):
            one_hot[i, c] = 1.0
        output.backward(gradient=one_hot, retain_graph=False)

        # Global-average-pool the gradients to get per-channel weights
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)   # (N, C, 1, 1)
        cam     = (weights * self._features).sum(dim=1)             # (N, H, W)
        cam     = F.relu(cam)

        # Normalise each map to [0, 1]
        flat     = cam.view(cam.size(0), -1)
        cam_min  = flat.min(dim=1)[0].view(-1, 1, 1)
        cam_max  = flat.max(dim=1)[0].view(-1, 1, 1)
        cam      = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam.detach()

    def remove(self) -> None:
        """Deregister hooks to avoid memory leaks."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def visualize_gradcam(
    model: nn.Module,
    data_params: DataParams,
    pgd_params: PGDParams,
    vis_params: VisParams,
    device: torch.device,
    target_layer: Optional[nn.Module] = None,
) -> None:
    """Plot Grad-CAM overlays for adversarial examples that fool the model.

    Iterates the test set until ``vis_params.gradcam_n_samples`` images are
    found where the clean prediction is correct but the L∞ adversarial
    prediction is wrong, then plots 6-column panels per sample:

        Clean | Clean GradCAM | L∞ Adv | L∞ GradCAM | L2 Adv | L2 GradCAM

    Args:
        model: Trained model.
        data_params: :class:`~parameters.DataParams`.
        pgd_params: :class:`~parameters.PGDParams`.
        vis_params: :class:`~parameters.VisParams`.
        device: Compute device.
        target_layer: Layer to hook for Grad-CAM.  Defaults to
            ``model.layer4`` if the model has that attribute.
    """
    if target_layer is None:
        if hasattr(model, "layer4"):
            target_layer = model.layer4
        else:
            raise ValueError(
                "target_layer must be specified for non-ResNet models."
            )

    model.eval()
    cam = GradCAM(model, target_layer)

    tf     = get_transforms(data_params, train=False)
    ds     = datasets.CIFAR10(data_params.data_dir, train=False, download=True, transform=tf)
    loader = DataLoader(ds, batch_size=data_params.batch_size,
                        shuffle=False, num_workers=data_params.num_workers)

    # CIFAR-10 un-normalisation for display
    mean = torch.tensor(data_params.mean).view(3, 1, 1)
    std  = torch.tensor(data_params.std).view(3, 1, 1)

    found_clean:      List[torch.Tensor] = []
    found_adv_linf:   List[torch.Tensor] = []
    found_adv_l2:     List[torch.Tensor] = []
    found_true:       List[int]          = []
    found_pred_clean: List[int]          = []
    found_pred_linf:  List[int]          = []
    found_pred_l2:    List[int]          = []

    target = vis_params.gradcam_n_samples

    for imgs, labels in loader:
        if len(found_clean) >= target:
            break
        imgs, labels = imgs.to(device), labels.to(device)

        with torch.no_grad():
            clean_preds = model(imgs).argmax(1)

        adv_linf = pgd_attack(
            model, imgs, labels,
            epsilon=pgd_params.epsilon_linf,
            alpha=pgd_params.alpha_linf,
            num_steps=pgd_params.num_steps,
            norm="linf",
        )
        adv_l2 = pgd_attack(
            model, imgs, labels,
            epsilon=pgd_params.epsilon_l2,
            alpha=pgd_params.alpha_l2,
            num_steps=pgd_params.num_steps,
            norm="l2",
        )
        with torch.no_grad():
            preds_linf = model(adv_linf).argmax(1)
            preds_l2   = model(adv_l2).argmax(1)

        for i in range(imgs.size(0)):
            if len(found_clean) >= target:
                break
            # Select samples where L∞ fools the model
            if clean_preds[i] == labels[i] and preds_linf[i] != labels[i]:
                found_clean.append(imgs[i:i+1].cpu())
                found_adv_linf.append(adv_linf[i:i+1].cpu())
                found_adv_l2.append(adv_l2[i:i+1].cpu())
                found_true.append(labels[i].item())
                found_pred_clean.append(clean_preds[i].item())
                found_pred_linf.append(preds_linf[i].item())
                found_pred_l2.append(preds_l2[i].item())

    if not found_clean:
        print("[GradCAM] No adversarial misclassifications found.")
        cam.remove()
        return

    print(f"[GradCAM] Found {len(found_clean)} adversarial examples. Plotting …")

    n   = len(found_clean)
    fig, axes = plt.subplots(n, 6, figsize=(21, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = [
        "Clean image", "Clean Grad-CAM",
        "L∞ Adv image", "L∞ Grad-CAM",
        "L2 Adv image", "L2 Grad-CAM",
    ]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11, fontweight="bold")

    def _to_img(t: torch.Tensor) -> np.ndarray:
        img = (t.squeeze(0).cpu() * std + mean).clamp(0, 1)
        return img.permute(1, 2, 0).numpy()

    def _upscale_heat(heat: np.ndarray, ref_shape) -> np.ndarray:
        if heat.shape == ref_shape[:2]:
            return np.array(plt.cm.jet(heat)[:, :, :3])
        upscaled = F.interpolate(
            torch.from_numpy(heat).unsqueeze(0).unsqueeze(0),
            size=ref_shape[:2], mode="bilinear", align_corners=False,
        ).squeeze().numpy()
        return np.array(plt.cm.jet(upscaled)[:, :, :3])

    for row in range(n):
        x_clean    = found_clean[row].to(device)
        x_linf     = found_adv_linf[row].to(device)
        x_l2       = found_adv_l2[row].to(device)
        true_c     = found_true[row]
        pred_c     = found_pred_clean[row]
        pred_linf  = found_pred_linf[row]
        pred_l2    = found_pred_l2[row]

        heat_clean = cam(x_clean)
        heat_linf  = cam(x_linf)
        heat_l2    = cam(x_l2)

        img_clean = _to_img(x_clean)
        img_linf  = _to_img(x_linf)
        img_l2    = _to_img(x_l2)
        hc_up     = _upscale_heat(heat_clean[0].cpu().numpy(), img_clean.shape)
        hl_up     = _upscale_heat(heat_linf[0].cpu().numpy(),  img_linf.shape)
        h2_up     = _upscale_heat(heat_l2[0].cpu().numpy(),    img_l2.shape)

        overlay_clean = np.clip(0.6 * img_clean + 0.4 * hc_up, 0, 1)
        overlay_linf  = np.clip(0.6 * img_linf  + 0.4 * hl_up, 0, 1)
        overlay_l2    = np.clip(0.6 * img_l2    + 0.4 * h2_up, 0, 1)

        axes[row, 0].imshow(img_clean)
        axes[row, 0].set_xlabel(
            f"True: {CIFAR10_CLASSES[true_c]}\nPred: {CIFAR10_CLASSES[pred_c]}", fontsize=9
        )
        axes[row, 1].imshow(overlay_clean)
        axes[row, 1].set_xlabel("Clean Grad-CAM", fontsize=9)

        axes[row, 2].imshow(img_linf)
        axes[row, 2].set_xlabel(
            f"True: {CIFAR10_CLASSES[true_c]}\nPred: {CIFAR10_CLASSES[pred_linf]}", fontsize=9
        )
        axes[row, 3].imshow(overlay_linf)
        axes[row, 3].set_xlabel("L∞ Grad-CAM", fontsize=9)

        axes[row, 4].imshow(img_l2)
        axes[row, 4].set_xlabel(
            f"True: {CIFAR10_CLASSES[true_c]}\nPred: {CIFAR10_CLASSES[pred_l2]}", fontsize=9
        )
        axes[row, 5].imshow(overlay_l2)
        axes[row, 5].set_xlabel("L2 Grad-CAM", fontsize=9)

        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle("Grad-CAM: Clean vs PGD L∞ vs PGD L2", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(vis_params.gradcam_save, dpi=150, bbox_inches="tight")
    print(f"[Plot] Grad-CAM saved → {vis_params.gradcam_save}")
    plt.close(fig)
    cam.remove()


# ──────────────────────────── T-SNE adversarial ──────────────────────────────


def visualize_tsne_adversarial(
    model: nn.Module,
    data_params: DataParams,
    pgd_params: PGDParams,
    vis_params: VisParams,
    device: torch.device,
) -> None:
    """Embed clean, L∞-adversarial, and L2-adversarial features with t-SNE.

    Extracts the penultimate-layer features (``model.get_features``) for
    ``vis_params.tsne_n_samples`` test images under three conditions, combines
    them, runs t-SNE, and saves a scatter plot with distinct markers.

    Args:
        model: Trained model implementing :meth:`get_features`.
        data_params: :class:`~parameters.DataParams`.
        pgd_params: :class:`~parameters.PGDParams`.
        vis_params: :class:`~parameters.VisParams`.
        device: Compute device.
    """
    from sklearn.manifold import TSNE

    model.eval()
    tf      = get_transforms(data_params, train=False)
    ds      = datasets.CIFAR10(data_params.data_dir, train=False, download=True, transform=tf)
    n       = min(vis_params.tsne_n_samples, len(ds))
    indices = np.random.choice(len(ds), n, replace=False)
    subset  = torch.utils.data.Subset(ds, indices)
    loader  = DataLoader(subset, batch_size=data_params.batch_size,
                         shuffle=False, num_workers=data_params.num_workers)

    print(f"[T-SNE] Extracting features for {n} samples …")
    feats_clean, feats_linf, feats_l2, all_labels = [], [], [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        adv_linf = pgd_attack(
            model, imgs, labels,
            epsilon=pgd_params.epsilon_linf,
            alpha=pgd_params.alpha_linf,
            num_steps=pgd_params.num_steps,
            norm="linf",
        )
        adv_l2 = pgd_attack(
            model, imgs, labels,
            epsilon=pgd_params.epsilon_l2,
            alpha=pgd_params.alpha_l2,
            num_steps=pgd_params.num_steps,
            norm="l2",
        )

        with torch.no_grad():
            feats_clean.append(model.get_features(imgs).cpu().numpy())
            feats_linf.append(model.get_features(adv_linf).cpu().numpy())
            feats_l2.append(model.get_features(adv_l2).cpu().numpy())
        all_labels.extend(labels.cpu().tolist())

    feats_clean = np.concatenate(feats_clean, axis=0)
    feats_linf  = np.concatenate(feats_linf,  axis=0)
    feats_l2    = np.concatenate(feats_l2,    axis=0)
    all_labels  = np.array(all_labels)

    # Stack: [clean | linf | l2], shape (3N, D)
    combined   = np.concatenate([feats_clean, feats_linf, feats_l2], axis=0)
    split_tags = (["clean"] * n + ["linf"] * n + ["l2"] * n)

    print("[T-SNE] Running t-SNE (this may take a few minutes) …")
    emb = TSNE(
        n_components=2, perplexity=30, random_state=42, max_iter=1000
    ).fit_transform(combined)

    emb_clean = emb[:n]
    emb_linf  = emb[n: 2*n]
    emb_l2    = emb[2*n:]

    # Colour by class, marker by type
    colors = [
        "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
        "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, emb_i, title in zip(
        axes,
        [emb_clean, emb_linf, emb_l2],
        ["Clean", f"PGD L∞ (ε={pgd_params.epsilon_linf:.4f})",
                  f"PGD L2 (ε={pgd_params.epsilon_l2:.2f})"],
    ):
        for cls_idx, cls_name in enumerate(CIFAR10_CLASSES):
            mask = all_labels == cls_idx
            ax.scatter(emb_i[mask, 0], emb_i[mask, 1],
                       c=colors[cls_idx], label=cls_name, s=8, alpha=0.6)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("t-SNE dim 1")
        ax.set_ylabel("t-SNE dim 2")

    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", ncol=5,
               markerscale=2, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("t-SNE: Clean vs. Adversarial Feature Embeddings", fontsize=13)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(vis_params.tsne_save, dpi=150, bbox_inches="tight")
    print(f"[Plot] t-SNE adversarial saved → {vis_params.tsne_save}")
    plt.close(fig)


# ──────────────────────────── Adversarial transfer ───────────────────────────


def run_transfer_pgd(
    teacher: nn.Module,
    student: nn.Module,
    data_params: DataParams,
    pgd_params: PGDParams,
    device: torch.device,
) -> Tuple[float, float]:
    """Measure transferability of teacher PGD adversarial examples to student.

    Generates PGD-20 L∞ adversarial examples using the **teacher** model and
    tests them on both the teacher and the student.

    Args:
        teacher: Teacher model (used to generate adversarial examples).
        student: Student model (target of the transfer attack).
        data_params: :class:`~parameters.DataParams`.
        pgd_params: :class:`~parameters.PGDParams`.
        device: Compute device.

    Returns:
        Tuple ``(teacher_adv_acc, student_adv_acc)`` — adversarial accuracy
        of each model on the teacher-generated examples.
    """
    teacher.eval()
    student.eval()

    tf     = get_transforms(data_params, train=False)
    ds     = datasets.CIFAR10(data_params.data_dir, train=False, download=True, transform=tf)
    n      = min(pgd_params.n_samples, len(ds)) if pgd_params.n_samples > 0 else len(ds)
    idxs   = np.random.choice(len(ds), n, replace=False)
    subset = torch.utils.data.Subset(ds, idxs)
    loader = DataLoader(subset, batch_size=data_params.batch_size,
                        shuffle=False, num_workers=data_params.num_workers)

    teacher_correct, student_correct, total = 0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        # Generate adversarial examples from the teacher
        adv = pgd_attack(
            teacher, imgs, labels,
            epsilon=pgd_params.epsilon_linf,
            alpha=pgd_params.alpha_linf,
            num_steps=pgd_params.num_steps,
            norm="linf",
        )

        with torch.no_grad():
            teacher_correct += teacher(adv).argmax(1).eq(labels).sum().item()
            student_correct += student(adv).argmax(1).eq(labels).sum().item()
        total += labels.size(0)

    teacher_acc = teacher_correct / total
    student_acc = student_correct / total

    print("\n=== Adversarial Transferability (Teacher PGD → Student) ===")
    print(f"  Teacher accuracy on teacher adv. examples : {teacher_acc:.4f}")
    print(f"  Student accuracy on teacher adv. examples : {student_acc:.4f}")
    print(f"  Transfer attack success rate (on student)  : {1 - student_acc:.4f}")

    return teacher_acc, student_acc
