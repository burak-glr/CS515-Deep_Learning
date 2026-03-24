"""Command-line argument parsing for HW2.

Exposes :func:`get_params` which returns a plain dictionary consumed by
``main.py``, ``train.py``, and ``test.py``.

Usage examples::

    # Transfer learning – Option 1 (freeze early, resize to 224)
    python main.py --task transfer --arch resnet18 --transfer_option 1 --epochs 10

    # Transfer learning – Option 2 (32×32 input, fine-tune all)
    python main.py --task transfer --arch resnet18 --transfer_option 2 --epochs 20

    # ResNet from scratch with label smoothing
    python main.py --task classify --model resnet --label_smoothing 0.1 --epochs 50

    # Knowledge distillation: ResNet teacher → SimpleCNN student
    python main.py --task distill --distill_mode standard --teacher_path best_resnet.pth --epochs 30

    # Modified KD: ResNet teacher → MobileNetV2 student
    python main.py --task distill --distill_mode modified --model mobilenet --teacher_path best_resnet.pth

    # Count FLOPs for a model
    python main.py --task classify --model resnet --count_flops --mode test
"""

import argparse
from typing import Any, Dict


def get_params() -> Dict[str, Any]:
    """Parse CLI arguments and return a configuration dictionary.

    Returns:
        A dictionary with all hyper-parameters and paths required to run
        training, evaluation, transfer learning, or knowledge distillation.
    """
    parser = argparse.ArgumentParser(
        description="HW2 – Transfer Learning and Knowledge Distillation on CIFAR-10"
    )

    # ── High-level task ──────────────────────────────────────────────────────
    parser.add_argument(
        "--task",
        choices=["classify", "transfer", "distill"],
        default="classify",
        help=(
            "Main task to run. "
            "'classify' trains/tests a model from scratch. "
            "'transfer' uses a pretrained ImageNet backbone. "
            "'distill' runs knowledge distillation."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["train", "test", "both"],
        default="both",
        help="Whether to train, test, or do both.",
    )

    # ── Model ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--model",
        choices=["cnn", "resnet", "mobilenet"],
        default="resnet",
        help=(
            "Student / scratch model. "
            "For task=distill this is the *student*. "
            "Ignored when task=transfer (arch is used instead)."
        ),
    )
    parser.add_argument(
        "--resnet_layers",
        type=int,
        nargs=4,
        default=[2, 2, 2, 2],
        metavar=("L1", "L2", "L3", "L4"),
        help="Blocks per ResNet stage. [2 2 2 2] → ResNet-18.",
    )

    # ── Transfer learning ────────────────────────────────────────────────────
    parser.add_argument(
        "--arch",
        choices=["resnet18", "vgg16"],
        default="resnet18",
        help="Pretrained backbone for task=transfer.",
    )
    parser.add_argument(
        "--transfer_option",
        choices=["1", "2"],
        default="1",
        help=(
            "Adaptation strategy for transfer learning. "
            "1 = resize to 224 + freeze early layers. "
            "2 = 3×3 first-conv for 32×32 + fine-tune all."
        ),
    )

    # ── Label smoothing ──────────────────────────────────────────────────────
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
        help="Label-smoothing epsilon (0 = disabled). Applies to classify and distill tasks.",
    )

    # ── Knowledge distillation ───────────────────────────────────────────────
    parser.add_argument(
        "--distill_mode",
        choices=["standard", "modified"],
        default="standard",
        help=(
            "Distillation variant. "
            "'standard' = Hinton et al. soft-target KD. "
            "'modified' = teacher probability assigned to true class only."
        ),
    )
    parser.add_argument(
        "--teacher_path",
        type=str,
        default="best_teacher.pth",
        help="Path to saved teacher model weights (.pth) for distillation.",
    )
    parser.add_argument(
        "--distill_temp",
        type=float,
        default=4.0,
        help="Distillation temperature T (only used for standard KD).",
    )
    parser.add_argument(
        "--distill_alpha",
        type=float,
        default=0.7,
        help="Weight of the soft-target (KD) loss vs. hard-label loss. "
             "loss = alpha * CE + (1-alpha) * KD.",
    )

    # ── Training hyper-parameters ─────────────────────────────────────────────
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--device",     type=str,   default="cuda")

    # ── Misc ─────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--count_flops",
        action="store_true",
        help="Print FLOPs and parameter count using ptflops before training.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="best_model.pth",
        help="File path for saving the best checkpoint.",
    )
    parser.add_argument(
        "--save_plots",
        action="store_true",
        help="Save training loss/accuracy curves as a PNG after training.",
    )

    # ── Visualization (standalone, skips training) ────────────────────────────
    parser.add_argument(
        "--visualize",
        choices=["tsne", "flops_kd", "flops_mob"],
        default=None,
        help=(
            "Generate a plot and exit (no training). "
            "'tsne'     — t-SNE scatter of CIFAR-10 raw pixels. "
            "'flops_kd' — FLOPs + accuracy bar chart: ResNet vs distilled SimpleCNN "
            "             (pass --teacher_acc and --student_acc). "
            "'flops_mob'— FLOPs + accuracy bar chart: ResNet vs distilled MobileNetV2 "
            "             (pass --teacher_acc and --student_acc)."
        ),
    )
    parser.add_argument("--n_samples",   type=int,   default=1000,
                        help="Number of CIFAR-10 samples for t-SNE (default: 1000).")
    parser.add_argument("--teacher_acc", type=float, default=0.0,
                        help="Teacher test accuracy (0-1) for FLOPs plots.")
    parser.add_argument("--student_acc", type=float, default=0.0,
                        help="Student test accuracy (0-1) for FLOPs plots.")
    parser.add_argument("--student_path", type=str,  default="best_student.pth",
                        help="Path to distilled student weights for FLOPs plots.")

    args = parser.parse_args()

    # CIFAR-10 normalisation statistics
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    return {
        # Data
        "dataset":    "cifar10",
        "data_dir":   "./data",
        "num_workers": 2,
        "num_classes": 10,
        "mean": mean,
        "std":  std,

        # Task / mode
        "task": args.task,
        "mode": args.mode,

        # Scratch / student model
        "model":         args.model,
        "resnet_layers": args.resnet_layers,

        # Transfer learning
        "arch":            args.arch,
        "transfer_option": args.transfer_option,

        # Label smoothing
        "label_smoothing": args.label_smoothing,

        # Knowledge distillation
        "distill_mode":  args.distill_mode,
        "teacher_path":  args.teacher_path,
        "distill_temp":  args.distill_temp,
        "distill_alpha": args.distill_alpha,

        # Training
        "epochs":        args.epochs,
        "batch_size":    args.batch_size,
        "learning_rate": args.lr,
        "weight_decay":  1e-4,

        # Misc
        "seed":        42,
        "device":      args.device,
        "save_path":   args.save_path,
        "log_interval": 100,
        "count_flops": args.count_flops,
        "save_plots":  args.save_plots,

        # Visualization
        "visualize":    args.visualize,
        "n_samples":    args.n_samples,
        "teacher_acc":  args.teacher_acc,
        "student_acc":  args.student_acc,
        "student_path": args.student_path,
    }
