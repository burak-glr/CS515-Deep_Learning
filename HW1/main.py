import random
import ssl

import numpy as np
import torch
import torch.nn as nn

from parameters import DataParams, ModelParams, TrainParams, get_params
from models.MLP import MLP
from train import run_training
from test import run_test


def visualize_torchviz(model: nn.Module, device: torch.device) -> None:
    """Save a torchviz computation graph of the model as mlp_graph.png.

    Args:
        model: The neural network model.
        device: Device the model is on.
    """
    try:
        from torchviz import make_dot
    except ImportError:
        print("torchviz not installed. Run: pip install torchviz")
        return

    model.eval()
    x   = torch.randn(1, 1, 28, 28).to(device)
    out = model(x)
    dot = make_dot(out, params=dict(model.named_parameters()))
    dot.render("mlp_graph", format="png", cleanup=True)
    print("Saved computation graph → mlp_graph.png")


def visualize_curves(
    train_losses: list,
    val_losses: list,
    train_accs: list,
    val_accs: list,
) -> None:
    """Plot and save training/validation loss and accuracy curves.

    Args:
        train_losses: Per-epoch training losses.
        val_losses: Per-epoch validation losses.
        train_accs: Per-epoch training accuracies.
        val_accs: Per-epoch validation accuracies.
    """
    import matplotlib.pyplot as plt

    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_losses, label="Train")
    ax1.plot(epochs, val_losses,   label="Val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss over Epochs")
    ax1.legend()

    ax2.plot(epochs, train_accs, label="Train")
    ax2.plot(epochs, val_accs,   label="Val")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy over Epochs")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("training_curves.png")
    plt.close()
    print("Saved training curves → training_curves.png")


def visualize_tsne(
    model: nn.Module,
    data_params: DataParams,
    train_params: TrainParams,
    device: torch.device,
) -> None:
    """Run t-SNE on learned features and save a 2D scatter plot.

    Args:
        model: The trained neural network model.
        data_params: Dataset parameters.
        train_params: Training parameters (batch size).
        device: Device the model is on.
    """
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    from train import get_loaders

    _, val_loader = get_loaders(data_params, train_params)

    features, labels = [], []
    model.eval()
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs = imgs.to(device)
            # Extract features before the final output layer
            feat = model.flatten(imgs)
            for block in model.hidden:
                feat = block(feat)
            features.append(feat.cpu())
            labels.append(lbls)

    features = torch.cat(features).numpy()
    labels   = torch.cat(labels).numpy()

    print("Running t-SNE (this may take a moment)...")
    emb = TSNE(n_components=2, random_state=42).fit_transform(features)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(emb[:, 0], emb[:, 1], c=labels, cmap="tab10", s=2, alpha=0.7)
    plt.colorbar(scatter, ticks=range(data_params.num_classes))
    plt.title("t-SNE of learned features")
    plt.tight_layout()
    plt.savefig("tsne.png")
    plt.close()
    print("Saved t-SNE plot → tsne.png")


# Fix for macOS SSL certificate verification error when downloading MNIST
ssl._create_default_https_context = ssl._create_unverified_context


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all libraries.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main() -> None:
    """Entry point: parse params, build model, run training and/or testing."""
    data_params, model_params, train_params = get_params()

    set_seed(train_params.seed)
    print(f"Seed set to: {train_params.seed}", flush=True)
    print(f"Hidden sizes: {model_params.hidden_sizes}  |  Activation: {model_params.activation}", flush=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}", flush=True)

    model = MLP(model_params).to(device)
    print(model)

    if "torchviz" in train_params.visualize:
        visualize_torchviz(model, device)

    history = None
    if train_params.mode in ("train", "both"):
        history = run_training(model, data_params, train_params, device)

    if train_params.mode in ("test", "both"):
        run_test(model, data_params, train_params, device)

    if "curves" in train_params.visualize and history is not None:
        visualize_curves(*history)

    if "tsne" in train_params.visualize:
        visualize_tsne(model, data_params, train_params, device)


if __name__ == "__main__":
    main()
