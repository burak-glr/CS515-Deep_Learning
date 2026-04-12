"""Entry point for HW3: Data Augmentation and Adversarial Samples.

Dispatches to one of seven task pipelines based on ``--task``:

  robustness
      Load a fine-tuned ResNet checkpoint and evaluate it on the clean
      CIFAR-10 test set and all CIFAR-10-C corruption types / severities.

  augmix
      Fine-tune ResNet-18 on CIFAR-10 with the AugMix framework
      (augmentation chains + JSD consistency loss), then evaluate on
      both the clean test set and CIFAR-10-C.

  pgd
      Run PGD-20 with L∞ (ε=4/255) and L2 (ε=0.25) attacks on the model
      specified by ``--model_path`` and report adversarial accuracy.

  gradcam
      Generate Grad-CAM heat-map overlays for adversarial examples that
      fool the model (L∞ PGD attack), saved to ``--gradcam_save``.

  tsne_adv
      Embed clean, L∞-adversarial, and L2-adversarial penultimate-layer
      features with t-SNE and save a scatter plot.

  distill_augmix
      Knowledge distillation using the AugMix-trained ResNet as teacher;
      supports ``--distill_mode standard|modified`` and student
      architectures ``--student_model cnn|mobilenet``.

  transfer_pgd
      Generate PGD-20 L∞ adversarial examples using the teacher model
      and test them on the student model to assess transferability.

Usage examples::

    # Robustness evaluation (clean + CIFAR-10-C)
    python main.py --task robustness --model_path best_resnet.pth

    # Train AugMix model then evaluate on clean + CIFAR-10-C
    python main.py --task augmix --mode both --save_path augmix_resnet.pth --save_plots

    # Only test an existing AugMix checkpoint
    python main.py --task augmix --mode test --save_path augmix_resnet.pth

    # PGD evaluation on a standard checkpoint
    python main.py --task pgd --model_path best_resnet.pth --pgd_n_samples 500

    # Grad-CAM visualization
    python main.py --task gradcam --model_path best_resnet.pth

    # t-SNE of adversarial features
    python main.py --task tsne_adv --model_path best_resnet.pth --tsne_n_samples 300

    # Distillation: AugMix teacher → SimpleCNN student
    python main.py --task distill_augmix \\
                   --teacher_path augmix_resnet.pth \\
                   --student_model cnn --distill_mode standard \\
                   --save_path best_student_cnn.pth

    # Adversarial transferability
    python main.py --task transfer_pgd \\
                   --teacher_path augmix_resnet.pth \\
                   --student_path best_student_cnn.pth \\
                   --student_model cnn
"""

import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from models import BasicBlock, MobileNetV2, ResNet, SimpleCNN
from parameters import DataParams, Params, get_params
from test import (
    run_cifar10c_test,
    run_pgd_test,
    run_test,
    run_transfer_pgd,
    visualize_gradcam,
    visualize_tsne_adversarial,
)
from train import run_augmix_training, run_distillation_training, run_training


# ──────────────────────────── Helpers ────────────────────────────────────────


def set_seed(seed: int) -> None:
    """Fix all random seeds for reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def resolve_device(requested: str) -> torch.device:
    """Return the best available device.

    Prefers the requested CUDA device, then MPS, then CPU.

    Args:
        requested: Device string from CLI (e.g. ``"cuda"`` or ``"cpu"``).

    Returns:
        A :class:`torch.device` instance.
    """
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_resnet(params: Params) -> nn.Module:
    """Instantiate the CIFAR-10 ResNet teacher model.

    Always creates a ResNet regardless of ``params.model_arch``.  Used
    exclusively for the teacher role in distillation and PGD-transfer tasks.

    Args:
        params: Top-level :class:`~parameters.Params` instance.

    Returns:
        An uninitialised :class:`~models.ResNet` instance.
    """
    return ResNet(BasicBlock, params.resnet_layers, num_classes=params.data.num_classes)


def build_model(params: Params) -> nn.Module:
    """Instantiate the *finetuned / student* model based on ``params.model_arch``.

    This is the primary model for robustness evaluation, AugMix training,
    PGD testing, Grad-CAM, and t-SNE tasks.  In HW3 the default is
    MobileNetV2 — the KD-trained student from HW2.

    Args:
        params: Top-level :class:`~parameters.Params` instance; uses
            ``params.model_arch`` and ``params.data.num_classes``.

    Returns:
        An uninitialised :class:`~torch.nn.Module`.

    Raises:
        ValueError: For unknown architecture names.
    """
    nc = params.data.num_classes
    if params.model_arch == "mobilenet":
        return MobileNetV2(num_classes=nc)
    if params.model_arch == "resnet":
        return ResNet(BasicBlock, params.resnet_layers, num_classes=nc)
    if params.model_arch == "cnn":
        return SimpleCNN(num_classes=nc)
    raise ValueError(f"Unknown model_arch: {params.model_arch!r}")


def get_gradcam_layer(model: nn.Module) -> nn.Module:
    """Return the best convolutional layer to hook for Grad-CAM.

    * ResNet   → ``model.layer4`` (last residual stage).
    * MobileNetV2 → ``model.conv2`` (final 1×1 projection, 1280 channels).
    * SimpleCNN   → ``model.conv2`` (second conv layer).

    Args:
        model: A model instance from the ``models`` package.

    Returns:
        The target :class:`~torch.nn.Module` layer.

    Raises:
        ValueError: If no suitable layer can be identified.
    """
    if hasattr(model, "layer4"):          # ResNet
        return model.layer4
    if isinstance(model, MobileNetV2):    # MobileNetV2
        return model.conv2                # (N, 1280, H, W) before pooling
    if isinstance(model, SimpleCNN):      # SimpleCNN
        return model.conv2
    raise ValueError(
        f"Cannot determine Grad-CAM target layer for {type(model).__name__}. "
        "Pass target_layer explicitly."
    )


def build_student(params: Params) -> nn.Module:
    """Instantiate the KD student architecture.

    Args:
        params: Top-level :class:`~parameters.Params` instance; uses
            ``params.distill.student_model``.

    Returns:
        An uninitialised student :class:`~torch.nn.Module`.

    Raises:
        ValueError: For unknown student architecture names.
    """
    name = params.distill.student_model
    nc   = params.data.num_classes
    if name == "cnn":
        return SimpleCNN(num_classes=nc)
    if name == "mobilenet":
        return MobileNetV2(num_classes=nc)
    raise ValueError(f"Unknown student model: {name!r}. Choose 'cnn' or 'mobilenet'.")


def load_checkpoint(
    model: nn.Module,
    path: str,
    device: torch.device,
) -> nn.Module:
    """Load state-dict from *path* into *model*.

    Args:
        model: Model architecture (weights loaded in-place).
        path: Path to the ``.pth`` checkpoint file.
        device: Target device for weight mapping.

    Returns:
        The same *model* with loaded weights.
    """
    model.load_state_dict(torch.load(path, map_location=device))
    print(f"Loaded checkpoint: {path}")
    return model


def print_flops(model: nn.Module, input_size: tuple) -> None:
    """Print FLOPs and parameter count using *ptflops*.

    Silently skips if *ptflops* is not installed.

    Args:
        model: The model to profile.
        input_size: Input tensor size excluding the batch dimension,
            e.g. ``(3, 32, 32)``.
    """
    try:
        from ptflops import get_model_complexity_info
        macs, params_str = get_model_complexity_info(
            model, input_size, as_strings=True,
            print_per_layer_stat=False, verbose=False,
        )
        print(f"\n[FLOPs] MACs: {macs}  |  Parameters: {params_str}")
    except ImportError:
        print("\n[FLOPs] ptflops not installed. Run: pip install ptflops")


def count_params(model: nn.Module) -> int:
    """Count trainable parameters.

    Args:
        model: A :class:`~torch.nn.Module`.

    Returns:
        Number of parameters with ``requires_grad=True``.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ──────────────────────────── Task runners ───────────────────────────────────


def run_robustness(params: Params, device: torch.device) -> None:
    """Evaluate a fine-tuned model against clean and corrupted test data.

    Loads the checkpoint at ``params.model_path`` into the architecture
    selected by ``params.model_arch`` (default: MobileNetV2, the KD-trained
    student from HW2), then evaluates on the clean CIFAR-10 test set and
    every CIFAR-10-C corruption type across all five severity levels.

    Args:
        params: Top-level :class:`~parameters.Params`.
        device: Compute device.
    """
    model = build_model(params).to(device)
    load_checkpoint(model, params.model_path, device)
    model.eval()

    if params.count_flops:
        print_flops(model, (3, 32, 32))

    # Clean accuracy
    run_test(model, params.data, device)

    # CIFAR-10-C accuracy
    run_cifar10c_test(model, params.data, device, vis_params=params.vis)


def run_augmix_task(params: Params, device: torch.device) -> None:
    """Fine-tune with AugMix and evaluate on clean + corrupted data.

    Re-trains the architecture selected by ``params.model_arch`` (default:
    MobileNetV2) with AugMix data augmentation and JSD consistency loss.

    * ``mode="train"`` or ``mode="both"``: runs AugMix training and saves the
      best checkpoint to ``params.training.save_path``.
    * ``mode="test"``  or ``mode="both"``: loads the checkpoint and evaluates
      on the clean CIFAR-10 test set and CIFAR-10-C.

    Args:
        params: Top-level :class:`~parameters.Params`.
        device: Compute device.
    """
    model = build_model(params).to(device)
    print(f"Model parameters: {count_params(model):,}")

    if params.count_flops:
        print_flops(model, (3, 32, 32))

    if params.mode in ("train", "both"):
        print(f"\n>>> AugMix fine-tuning → {params.training.save_path}")
        run_augmix_training(
            model, params.data, params.training, params.augmix, device
        )

    if params.mode in ("test", "both"):
        ckpt = params.training.save_path if params.mode == "both" else params.model_path
        load_checkpoint(model, ckpt, device)
        model.eval()
        run_test(model, params.data, device)
        run_cifar10c_test(model, params.data, device, vis_params=params.vis)


def run_pgd_task(params: Params, device: torch.device) -> None:
    """Evaluate a model under PGD-20 L∞ and L2 adversarial attacks.

    Loads the checkpoint at ``params.model_path`` into the architecture
    selected by ``params.model_arch`` and reports adversarial accuracy for
    both threat models.

    Args:
        params: Top-level :class:`~parameters.Params`.
        device: Compute device.
    """
    model = build_model(params).to(device)
    load_checkpoint(model, params.model_path, device)
    model.eval()
    run_pgd_test(model, params.data, params.pgd, device, tag=params.model_path)


def run_gradcam_task(params: Params, device: torch.device) -> None:
    """Generate Grad-CAM visualisations for adversarial misclassifications.

    Uses PGD L∞ to attack the model at ``params.model_path`` and plots
    side-by-side clean / adversarial image + Grad-CAM overlay panels.
    The Grad-CAM target layer is chosen automatically per architecture:

    * MobileNetV2 → ``model.conv2`` (1280-channel projection layer)
    * ResNet       → ``model.layer4``
    * SimpleCNN    → ``model.conv2``

    Args:
        params: Top-level :class:`~parameters.Params`.
        device: Compute device.
    """
    model = build_model(params).to(device)
    load_checkpoint(model, params.model_path, device)
    model.eval()
    visualize_gradcam(
        model, params.data, params.pgd, params.vis, device,
        target_layer=get_gradcam_layer(model),
    )


def run_tsne_adv_task(params: Params, device: torch.device) -> None:
    """Run t-SNE on clean and adversarial penultimate-layer features.

    Uses the architecture from ``params.model_arch`` (default: MobileNetV2).

    Args:
        params: Top-level :class:`~parameters.Params`.
        device: Compute device.
    """
    model = build_model(params).to(device)
    load_checkpoint(model, params.model_path, device)
    model.eval()
    visualize_tsne_adversarial(model, params.data, params.pgd, params.vis, device)


def run_distill_augmix_task(params: Params, device: torch.device) -> None:
    """Knowledge distillation using the AugMix-trained ResNet as teacher.

    Loads the teacher from ``params.distill.teacher_path``, builds the
    student specified by ``params.distill.student_model``, and runs the
    selected distillation mode.

    * ``mode="train"`` or ``mode="both"``: trains the student.
    * ``mode="test"``  or ``mode="both"``: evaluates the student on the
      clean test set.

    Args:
        params: Top-level :class:`~parameters.Params`.
        device: Compute device.
    """
    # Teacher
    teacher = build_resnet(params).to(device)
    load_checkpoint(teacher, params.distill.teacher_path, device)
    teacher.eval()
    print(f"Teacher parameters: {count_params(teacher):,}")

    if params.count_flops:
        print("\n[Teacher FLOPs]")
        print_flops(teacher, (3, 32, 32))

    # Student
    student = build_student(params).to(device)
    print(f"Student ({params.distill.student_model}) parameters: {count_params(student):,}")

    if params.count_flops:
        print("\n[Student FLOPs]")
        print_flops(student, (3, 32, 32))

    if params.mode in ("train", "both"):
        run_distillation_training(
            student, teacher,
            params.data, params.training, params.distill, device,
        )

    if params.mode in ("test", "both"):
        ckpt = params.distill.student_path
        load_checkpoint(student, ckpt, device)
        student.eval()
        run_test(student, params.data, device,
                 checkpoint_path=None)   # already loaded


def run_transfer_pgd_task(params: Params, device: torch.device) -> None:
    """Test adversarial transferability from teacher PGD to student.

    Generates PGD-20 L∞ adversarial examples with the teacher model and
    measures the attack success rate on the student model.

    Args:
        params: Top-level :class:`~parameters.Params`.
        device: Compute device.
    """
    teacher = build_resnet(params).to(device)
    load_checkpoint(teacher, params.distill.teacher_path, device)
    teacher.eval()

    student = build_student(params).to(device)
    load_checkpoint(student, params.distill.student_path, device)
    student.eval()

    run_transfer_pgd(teacher, student, params.data, params.pgd, device)


# ──────────────────────────── Main ───────────────────────────────────────────


def main() -> None:
    """Parse arguments and dispatch to the appropriate task pipeline."""
    params = get_params()
    set_seed(params.training.seed)

    device = resolve_device(params.device)
    print(f"Device: {device}")
    print(f"Task:   {params.task}")

    task = params.task
    if task == "robustness":
        run_robustness(params, device)
    elif task == "augmix":
        run_augmix_task(params, device)
    elif task == "pgd":
        run_pgd_task(params, device)
    elif task == "gradcam":
        run_gradcam_task(params, device)
    elif task == "tsne_adv":
        run_tsne_adv_task(params, device)
    elif task == "distill_augmix":
        run_distill_augmix_task(params, device)
    elif task == "transfer_pgd":
        run_transfer_pgd_task(params, device)
    else:
        raise ValueError(f"Unknown task: {task!r}")


if __name__ == "__main__":
    main()
