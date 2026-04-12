"""Command-line argument parsing for HW3: Data Augmentation and Adversarial Samples.

All parameter groups are represented as ``dataclasses`` and populated from
``argparse`` arguments.  :func:`get_params` returns a :class:`Params` instance
that is passed through the pipeline.

Tasks
-----
robustness
    Load a fine-tuned ResNet checkpoint and evaluate it on both the clean
    CIFAR-10 test set and every severity level of every CIFAR-10-C corruption.

augmix
    Fine-tune ResNet-18 with the AugMix framework (augmentation chains +
    Jensen-Shannon divergence consistency loss), then evaluate on clean and
    corrupted test sets.

pgd
    Run PGD-20 attacks (L∞ ε=4/255 and L2 ε=0.25) on a checkpoint and report
    adversarial accuracy for both the standard and AugMix-trained models.

gradcam
    Generate Grad-CAM heat-maps on adversarial samples where the model is
    fooled, overlaid on both the clean and perturbed images.

tsne_adv
    Embed clean, L∞-adversarial, and L2-adversarial samples in 2-D with t-SNE
    using the ResNet's penultimate-layer features.

distill_augmix
    Knowledge distillation using the AugMix-trained ResNet as teacher.

transfer_pgd
    Generate PGD-20 L∞ adversarial examples using the teacher model and
    measure the attack success rate when transferred to the student model.

Usage examples::

    # Evaluate a standard fine-tuned checkpoint on CIFAR-10-C
    python main.py --task robustness --model_path best_resnet.pth

    # Train with AugMix and evaluate
    python main.py --task augmix --mode both --save_path augmix_resnet.pth --save_plots

    # PGD evaluation on the standard model
    python main.py --task pgd --model_path best_resnet.pth

    # PGD evaluation on the AugMix model
    python main.py --task pgd --model_path augmix_resnet.pth --pgd_n_samples 500

    # Grad-CAM on adversarial images
    python main.py --task gradcam --model_path best_resnet.pth

    # t-SNE of adversarial embeddings
    python main.py --task tsne_adv --model_path best_resnet.pth --tsne_n_samples 500

    # KD with AugMix teacher → SimpleCNN student
    python main.py --task distill_augmix --teacher_path augmix_resnet.pth \\
                   --student_model cnn --distill_mode standard

    # Adversarial transferability: teacher PGD on student
    python main.py --task transfer_pgd --teacher_path augmix_resnet.pth \\
                   --student_path best_student.pth --student_model cnn
"""

import argparse
from dataclasses import dataclass, field
from typing import List, Tuple


# ──────────────────────────── Parameter dataclasses ──────────────────────────


@dataclass
class DataParams:
    """Parameters controlling dataset loading and pre-processing.

    Attributes:
        data_dir: Root directory where CIFAR-10 will be downloaded / cached.
        cifar10c_dir: Directory containing the CIFAR-10-C ``.npy`` files.
        num_workers: DataLoader worker processes.
        batch_size: Mini-batch size for training and evaluation.
        num_classes: Number of target classes (10 for CIFAR-10).
        mean: Per-channel mean for normalisation.
        std: Per-channel standard deviation for normalisation.
    """

    data_dir: str = "./data"
    cifar10c_dir: str = "./data/CIFAR-10-C"
    num_workers: int = 2
    batch_size: int = 128
    num_classes: int = 10
    mean: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
    std: Tuple[float, float, float] = (0.2023, 0.1994, 0.2010)


@dataclass
class TrainingParams:
    """Hyper-parameters for the standard and AugMix training loops.

    Attributes:
        epochs: Total number of training epochs.
        lr: Initial learning rate for Adam.
        weight_decay: L2 regularisation coefficient.
        seed: Random seed for reproducibility.
        save_path: File path for the best-checkpoint ``.pth`` file.
        save_plots: If ``True``, save loss/accuracy curves as a PNG.
        log_interval: Print progress every this many mini-batches.
    """

    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    save_path: str = "best_model.pth"
    save_plots: bool = False
    log_interval: int = 100


@dataclass
class AugMixParams:
    """Parameters for the AugMix augmentation strategy.

    Attributes:
        severity: Maximum magnitude of each augmentation operation (1–10).
        width: Number of parallel augmentation chains mixed together.
        depth: Fixed augmentation chain depth; ``-1`` samples randomly in 1–3.
        alpha: Dirichlet/Beta concentration parameter for mixing weights.
        jsd_lambda: Weight λ that scales the JSD consistency loss term.
    """

    severity: int = 3
    width: int = 3
    depth: int = -1
    alpha: float = 1.0
    jsd_lambda: float = 12.0


@dataclass
class PGDParams:
    """Parameters for the PGD-20 adversarial attack.

    The step size ``alpha`` is set to ``epsilon / 4`` by convention when not
    provided explicitly (a common heuristic for PGD).

    Attributes:
        epsilon_linf: L∞ perturbation budget in [0, 1] pixel space.
        epsilon_l2: L2 perturbation budget.
        num_steps: Number of PGD gradient-ascent iterations.
        alpha_linf: L∞ step size (per-step perturbation magnitude).
        alpha_l2: L2 step size.
        random_start: Initialise perturbation uniformly at random.
        n_samples: Number of test images to attack (0 = full test set).
    """

    epsilon_linf: float = 4 / 255
    epsilon_l2: float = 0.25
    num_steps: int = 20
    alpha_linf: float = 1 / 255
    alpha_l2: float = 0.05
    random_start: bool = True
    n_samples: int = 1000


@dataclass
class DistillParams:
    """Parameters for knowledge distillation.

    Attributes:
        teacher_path: Path to the saved teacher model weights.
        student_model: Architecture of the student (``"cnn"`` or ``"mobilenet"``).
        student_path: Path to save / load the student model weights.
        temperature: Distillation temperature T (standard KD only).
        alpha: Weight for the hard-label cross-entropy term.
        mode: ``"standard"`` (Hinton KD) or ``"modified"`` (true-class only).
    """

    teacher_path: str = "best_teacher_augmix.pth"
    student_model: str = "cnn"
    student_path: str = "best_student.pth"
    temperature: float = 4.0
    alpha: float = 0.7
    mode: str = "standard"


@dataclass
class VisParams:
    """Parameters for visualisation utilities.

    Attributes:
        tsne_n_samples: Number of images to embed with t-SNE.
        gradcam_n_samples: Number of adversarial misclassifications to plot.
        gradcam_save: File path for the Grad-CAM figure.
        tsne_save: File path for the t-SNE scatter plot.
        robustness_save: File path for the CIFAR-10-C bar chart.
    """

    tsne_n_samples: int = 500
    gradcam_n_samples: int = 2
    gradcam_save: str = "gradcam.png"
    tsne_save: str = "tsne_adversarial.png"
    robustness_save: str = "cifar10c_robustness.png"


@dataclass
class Params:
    """Top-level configuration container passed throughout the pipeline.

    Attributes:
        task: The task to execute (see module docstring for choices).
        mode: ``"train"``, ``"test"``, or ``"both"`` (for trainable tasks).
        model_path: Path to the checkpoint for evaluation-only tasks.
        resnet_layers: Block counts per ResNet stage, e.g. ``[2, 2, 2, 2]``.
        count_flops: Print FLOPs / parameter count with *ptflops*.
        device: Device string (``"cuda"``, ``"cpu"``).
        data: :class:`DataParams` instance.
        training: :class:`TrainingParams` instance.
        augmix: :class:`AugMixParams` instance.
        pgd: :class:`PGDParams` instance.
        distill: :class:`DistillParams` instance.
        vis: :class:`VisParams` instance.
    """

    task: str = "robustness"
    mode: str = "both"
    model_arch: str = "mobilenet"   # architecture of the finetuned / student model
    model_path: str = "best_model.pth"
    resnet_layers: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    count_flops: bool = False
    device: str = "cuda"
    data: DataParams = field(default_factory=DataParams)
    training: TrainingParams = field(default_factory=TrainingParams)
    augmix: AugMixParams = field(default_factory=AugMixParams)
    pgd: PGDParams = field(default_factory=PGDParams)
    distill: DistillParams = field(default_factory=DistillParams)
    vis: VisParams = field(default_factory=VisParams)


# ──────────────────────────── Argument parsing ───────────────────────────────


def get_params() -> Params:
    """Parse CLI arguments and return a populated :class:`Params` instance.

    Returns:
        A :class:`Params` dataclass ready to be consumed by ``main.py``.
    """
    parser = argparse.ArgumentParser(
        description="HW3 – Data Augmentation and Adversarial Samples on CIFAR-10",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── High-level control ───────────────────────────────────────────────────
    parser.add_argument(
        "--task",
        choices=["robustness", "augmix", "pgd", "gradcam", "tsne_adv",
                 "distill_augmix", "transfer_pgd"],
        default="robustness",
        help="Pipeline task to execute.",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "test", "both"],
        default="both",
        help="Whether to train, test, or do both (for trainable tasks).",
    )
    parser.add_argument(
        "--model_arch",
        choices=["mobilenet", "resnet", "cnn"],
        default="mobilenet",
        help=(
            "Architecture of the finetuned / student model. "
            "'mobilenet' = MobileNetV2 trained via KD in HW2 (default). "
            "'resnet'    = CIFAR-10 adapted ResNet. "
            "'cnn'       = SimpleCNN."
        ),
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="best_model.pth",
        help="Checkpoint path for evaluation / transfer tasks.",
    )
    parser.add_argument(
        "--resnet_layers",
        type=int,
        nargs=4,
        default=[2, 2, 2, 2],
        metavar=("L1", "L2", "L3", "L4"),
        help="Blocks per ResNet stage. [2 2 2 2] → ResNet-18.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Compute device (cuda / cpu).",
    )
    parser.add_argument(
        "--count_flops",
        action="store_true",
        help="Print FLOPs and parameter count using ptflops.",
    )

    # ── Data ─────────────────────────────────────────────────────────────────
    parser.add_argument("--data_dir",      type=str, default="./data",
                        help="Root directory for CIFAR-10 download/cache.")
    parser.add_argument("--cifar10c_dir",  type=str, default="./data/CIFAR-10-C",
                        help="Directory containing CIFAR-10-C .npy files.")
    parser.add_argument("--batch_size",    type=int, default=128)
    parser.add_argument("--num_workers",   type=int, default=2)

    # ── Training ─────────────────────────────────────────────────────────────
    parser.add_argument("--epochs",        type=int,   default=50)
    parser.add_argument("--lr",            type=float, default=1e-3)
    parser.add_argument("--weight_decay",  type=float, default=1e-4)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--save_path",     type=str,   default="best_model.pth",
                        help="File path for the best checkpoint (.pth).")
    parser.add_argument("--save_plots",    action="store_true",
                        help="Save training curves as PNG after training.")
    parser.add_argument("--log_interval",  type=int,   default=100)

    # ── AugMix ───────────────────────────────────────────────────────────────
    parser.add_argument("--augmix_severity",   type=int,   default=3,
                        help="AugMix augmentation severity (1–10).")
    parser.add_argument("--augmix_width",      type=int,   default=3,
                        help="Number of parallel augmentation chains.")
    parser.add_argument("--augmix_depth",      type=int,   default=-1,
                        help="Chain depth (-1 = random 1–3).")
    parser.add_argument("--augmix_alpha",      type=float, default=1.0,
                        help="Dirichlet/Beta concentration parameter.")
    parser.add_argument("--jsd_lambda",        type=float, default=12.0,
                        help="Weight for the JSD consistency loss.")

    # ── PGD ──────────────────────────────────────────────────────────────────
    parser.add_argument("--pgd_eps_linf",    type=float, default=4/255,
                        help="L∞ perturbation budget (pixel space 0–1).")
    parser.add_argument("--pgd_eps_l2",      type=float, default=0.25,
                        help="L2 perturbation budget.")
    parser.add_argument("--pgd_steps",       type=int,   default=20,
                        help="Number of PGD iterations.")
    parser.add_argument("--pgd_alpha_linf",  type=float, default=1/255,
                        help="L∞ step size per PGD iteration.")
    parser.add_argument("--pgd_alpha_l2",    type=float, default=0.05,
                        help="L2 step size per PGD iteration.")
    parser.add_argument("--pgd_n_samples",   type=int,   default=1000,
                        help="Number of test samples to attack (0 = all).")

    # ── Distillation ─────────────────────────────────────────────────────────
    parser.add_argument("--teacher_path",   type=str, default="best_teacher_augmix.pth",
                        help="Path to the teacher model checkpoint.")
    parser.add_argument("--student_model",  choices=["cnn", "mobilenet"], default="mobilenet",
                        help="Student architecture for KD (mobilenet = MobileNetV2).")
    parser.add_argument("--student_path",   type=str, default="best_student.pth",
                        help="Path to save/load the student checkpoint.")
    parser.add_argument("--distill_temp",   type=float, default=4.0,
                        help="Distillation temperature T.")
    parser.add_argument("--distill_alpha",  type=float, default=0.7,
                        help="Weight for the hard-label CE loss.")
    parser.add_argument("--distill_mode",   choices=["standard", "modified"],
                        default="standard",
                        help="Distillation variant.")

    # ── Visualisation ────────────────────────────────────────────────────────
    parser.add_argument("--tsne_n_samples",   type=int, default=500,
                        help="Samples for t-SNE adversarial plot.")
    parser.add_argument("--gradcam_n_samples", type=int, default=2,
                        help="Adversarial misclassifications to show in Grad-CAM.")
    parser.add_argument("--gradcam_save",     type=str, default="gradcam.png")
    parser.add_argument("--tsne_save",        type=str, default="tsne_adversarial.png")
    parser.add_argument("--robustness_save",  type=str, default="cifar10c_robustness.png")

    args = parser.parse_args()

    return Params(
        task=args.task,
        mode=args.mode,
        model_arch=args.model_arch,
        model_path=args.model_path,
        resnet_layers=args.resnet_layers,
        count_flops=args.count_flops,
        device=args.device,
        data=DataParams(
            data_dir=args.data_dir,
            cifar10c_dir=args.cifar10c_dir,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
        ),
        training=TrainingParams(
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed,
            save_path=args.save_path,
            save_plots=args.save_plots,
            log_interval=args.log_interval,
        ),
        augmix=AugMixParams(
            severity=args.augmix_severity,
            width=args.augmix_width,
            depth=args.augmix_depth,
            alpha=args.augmix_alpha,
            jsd_lambda=args.jsd_lambda,
        ),
        pgd=PGDParams(
            epsilon_linf=args.pgd_eps_linf,
            epsilon_l2=args.pgd_eps_l2,
            num_steps=args.pgd_steps,
            alpha_linf=args.pgd_alpha_linf,
            alpha_l2=args.pgd_alpha_l2,
            n_samples=args.pgd_n_samples,
        ),
        distill=DistillParams(
            teacher_path=args.teacher_path,
            student_model=args.student_model,
            student_path=args.student_path,
            temperature=args.distill_temp,
            alpha=args.distill_alpha,
            mode=args.distill_mode,
        ),
        vis=VisParams(
            tsne_n_samples=args.tsne_n_samples,
            gradcam_n_samples=args.gradcam_n_samples,
            gradcam_save=args.gradcam_save,
            tsne_save=args.tsne_save,
            robustness_save=args.robustness_save,
        ),
    )
