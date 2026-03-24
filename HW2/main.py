import random
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn

from parameters import get_params
from models.CNN import SimpleCNN
from models.ResNet import ResNet, BasicBlock
from models.mobilenet import MobileNetV2
from models.transfer_models import TransferConfig, build_transfer_model, count_trainable_params
from train import get_loaders, run_training, run_distillation_training, plot_tsne, plot_flops_comparison
from test import run_test


# ─────────────────────────── Helpers ─────────────────────────────────────────

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


def build_scratch_model(params: Dict[str, Any]) -> nn.Module:
    """Instantiate a model for scratch training or as the student in distillation.

    Args:
        params: Configuration dict; uses ``"model"`` and ``"num_classes"``.

    Returns:
        An :class:`~torch.nn.Module` instance.

    Raises:
        ValueError: For unknown model names.
    """
    name = params["model"]
    nc   = params["num_classes"]
    if name == "cnn":
        return SimpleCNN(num_classes=nc)
    if name == "resnet":
        return ResNet(BasicBlock, params["resnet_layers"], num_classes=nc)
    if name == "mobilenet":
        return MobileNetV2(num_classes=nc)
    raise ValueError(f"Unknown model: {name!r}")


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
        macs, params = get_model_complexity_info(
            model, input_size, as_strings=True,
            print_per_layer_stat=False, verbose=False,
        )
        print(f"\n[FLOPs] MACs: {macs}  |  Parameters: {params}")
    except ImportError:
        print("\n[FLOPs] ptflops not installed. Run: pip install ptflops")


# ─────────────────────────── Task runners ────────────────────────────────────

def run_classify(params: Dict[str, Any], device: torch.device) -> None:
    """Train and/or test a model from scratch (Part B – baseline + label smoothing).

    Args:
        params: Configuration dict.
        device: Compute device.
    """
    model = build_scratch_model(params).to(device)
    print(model)
    print(f"Trainable parameters: {count_trainable_params(model):,}")

    if params["count_flops"]:
        print_flops(model, (3, 32, 32))

    if params["mode"] in ("train", "both"):
        ls = params["label_smoothing"]
        print(f"\n>>> Training {'with' if ls > 0 else 'without'} label smoothing"
              f"{f' (ε={ls})' if ls > 0 else ''}")
        run_training(model, params, device)

    if params["mode"] in ("test", "both"):
        run_test(model, params, device)


def run_transfer(params: Dict[str, Any], device: torch.device) -> None:
    """Fine-tune a pretrained backbone on CIFAR-10 (Part A).

    Option 1: resize images to 224, freeze feature extractor, train FC head.
    Option 2: replace first conv for 32×32, fine-tune the whole network.

    Args:
        params: Configuration dict.
        device: Compute device.
    """
    cfg = TransferConfig(
        arch=params["arch"],
        option=params["transfer_option"],
        num_classes=params["num_classes"],
    )
    model, img_size = build_transfer_model(cfg)
    model = model.to(device)

    # Inject the resize dimension so data loaders pick it up.
    params["resize"] = img_size if img_size != 32 else 0

    print(model)
    print(f"Trainable parameters: {count_trainable_params(model):,}")
    print(f"Input image size:      {img_size}×{img_size}")

    if params["count_flops"]:
        print_flops(model, (3, img_size, img_size))

    if params["mode"] in ("train", "both"):
        print(f"\n>>> Transfer learning — Option {params['transfer_option']} "
              f"({params['arch']}, img={img_size})")
        run_training(model, params, device)

    if params["mode"] in ("test", "both"):
        run_test(model, params, device)

    # Clean up so other code doesn't accidentally pick up the resize value.
    params.pop("resize", None)


def run_distill(params: Dict[str, Any], device: torch.device) -> None:
    """Knowledge distillation from a saved ResNet teacher (Part B).

    Loads the teacher from ``params["teacher_path"]``, then trains the
    student model using the specified distillation mode.

    Args:
        params: Configuration dict.
        device: Compute device.
    """
    # Build and load teacher (always ResNet-18).
    teacher = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=params["num_classes"]).to(device)
    teacher.load_state_dict(torch.load(params["teacher_path"], map_location=device))
    teacher.eval()
    print(f"Teacher loaded from: {params['teacher_path']}")
    print(f"Teacher parameters:  {count_trainable_params(teacher):,}")

    if params["count_flops"]:
        print("\n[Teacher FLOPs]")
        print_flops(teacher, (3, 32, 32))

    # Build student.
    student = build_scratch_model(params).to(device)
    print(f"\nStudent: {params['model']}")
    print(f"Student parameters: {count_trainable_params(student):,}")

    if params["count_flops"]:
        print("\n[Student FLOPs]")
        print_flops(student, (3, 32, 32))

    if params["mode"] in ("train", "both"):
        print(f"\n>>> Distillation mode: {params['distill_mode']}")
        run_distillation_training(student, teacher, params, device)

    if params["mode"] in ("test", "both"):
        run_test(student, params, device)


# ─────────────────────────── Main ────────────────────────────────────────────

def run_visualize(params: Dict[str, Any]) -> None:
    """Dispatch to the requested visualization and exit.

    Args:
        params: Configuration dict; uses ``"visualize"``, ``"n_samples"``,
            ``"teacher_acc"``, ``"student_acc"``, ``"teacher_path"``,
            ``"student_path"``, and ``"model"``.
    """
    vis = params["visualize"]

    if vis == "tsne":
        plot_tsne(n_samples=params["n_samples"])
        return

    # Both flops plots need teacher + student models and accuracies.
    teacher = ResNet(BasicBlock, params["resnet_layers"], num_classes=params["num_classes"])

    if vis == "flops_kd":
        student = SimpleCNN(num_classes=params["num_classes"])
        plot_flops_comparison(
            model_a=teacher,
            model_b=student,
            name_a="ResNet (teacher)",
            name_b="SimpleCNN (distilled)",
            acc_a=params["teacher_acc"],
            acc_b=params["student_acc"],
            save_path="flops_kd_comparison.png",
        )

    elif vis == "flops_mob":
        student = MobileNetV2(num_classes=params["num_classes"])
        plot_flops_comparison(
            model_a=teacher,
            model_b=student,
            name_a="ResNet (teacher)",
            name_b="MobileNetV2 (modified KD)",
            acc_a=params["teacher_acc"],
            acc_b=params["student_acc"],
            save_path="flops_mobilenet_comparison.png",
        )


def main() -> None:
    """Parse arguments and dispatch to the appropriate task pipeline."""
    params = get_params()
    set_seed(params["seed"])

    # Visualization mode: generate a plot and exit immediately.
    if params.get("visualize"):
        run_visualize(params)
        return

    device = resolve_device(params["device"])
    print(f"Device:  {device}")
    print(f"Task:    {params['task']}")
    print(f"Dataset: {params['dataset']}")

    task = params["task"]
    if task == "classify":
        run_classify(params, device)
    elif task == "transfer":
        run_transfer(params, device)
    elif task == "distill":
        run_distill(params, device)
    else:
        raise ValueError(f"Unknown task: {task!r}")


if __name__ == "__main__":
    main()
