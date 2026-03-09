import argparse
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class DataParams:
    """Dataset and data loading parameters."""

    dataset: str = "mnist"
    data_dir: str = "./data"
    num_workers: int = 1
    mean: Tuple[float, ...] = (0.1307,)
    std: Tuple[float, ...] = (0.3081,)
    num_classes: int = 10


@dataclass
class ModelParams:
    """MLP architecture parameters."""

    input_size: int = 784
    hidden_sizes: List[int] = field(default_factory=lambda: [512, 256, 128])
    activation: str = "relu"
    dropout: float = 0.3
    num_classes: int = 10
    bn_after_act: bool = False


@dataclass
class TrainParams:
    """Training hyperparameters."""

    mode: str = "both"
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    device: str = "cpu"
    save_path: str = "best_model.pth"
    log_interval: int = 100
    early_stop_patience: int = 0  # 0 = disabled
    scheduler: str = "steplr"
    visualize: List[str] = field(default_factory=list)  # e.g. ["torchviz", "curves", "tsne"]
    regularizer: str = "l2"


def get_params() -> Tuple[DataParams, ModelParams, TrainParams]:
    """Parse command-line arguments and return structured parameter dataclasses.

    Returns:
        Tuple of (DataParams, ModelParams, TrainParams).
    """
    parser = argparse.ArgumentParser(description="HW1a: MNIST classification with MLP")

    parser.add_argument("--mode",         choices=["train", "test", "both"], default="both")
    parser.add_argument("--epochs",       type=int,   default=10)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--device",       type=str,   default="cpu")
    parser.add_argument("--batch_size",   type=int,   default=64)
    parser.add_argument("--hidden_sizes", type=int,   nargs="+", default=[512, 256, 128],
                        help="Hidden layer widths, e.g. --hidden_sizes 512 256 128")
    parser.add_argument("--activation",   choices=["relu", "leakyrelu", "gelu"], default="relu")
    parser.add_argument("--dropout",      type=float, default=0.3)
    parser.add_argument("--early_stop",   type=int,   default=0,
                        help="Early stopping patience (0 = disabled)")
    parser.add_argument("--scheduler",    choices=["steplr", "reducelronplateau", "cosineannealinglr"],
                        default="steplr")
    parser.add_argument("--save_path",    type=str,   default="best_model.pth")
    parser.add_argument("--visualize",    nargs="+",  default=[],
                        choices=["torchviz", "curves", "tsne"],
                        help="Visualization options: torchviz curves tsne")
    parser.add_argument("--bn_after_act", action="store_true", default=False,
                        help="Place BN after activation instead of before")
    parser.add_argument("--regularizer",  choices=["l1", "l2"], default="l2",
                        help="Regularization type: l1 or l2")

    args = parser.parse_args()

    data_params = DataParams()

    model_params = ModelParams(
        hidden_sizes=args.hidden_sizes,
        activation=args.activation,
        dropout=args.dropout,
        bn_after_act=args.bn_after_act,
    )

    train_params = TrainParams(
        mode=args.mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        early_stop_patience=args.early_stop,
        scheduler=args.scheduler,
        save_path=args.save_path,
        visualize=args.visualize,
        regularizer=args.regularizer,
    )

    return data_params, model_params, train_params
