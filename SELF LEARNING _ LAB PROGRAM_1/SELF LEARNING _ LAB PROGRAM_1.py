"""
SELF LEARNING - LAB PROGRAM 1 (CONTINUOUS EVALUATION)

Experiments covered:
1. Batch, stochastic, and mini-batch gradient descent on a small 2D dataset.
2. MLP on a subset of Fashion-MNIST.
3. Momentum-based and adaptive optimizers with constant and scheduled learning rates.
4. Overfitting analysis and comparison of dropout, L2 regularization, and data augmentation.
5. Early stopping.
6. Plain MLP vs residual MLP under controlled training conditions.

Run:
    python self_learning_lab_program1.py

Dependencies:
    pip install torch torchvision matplotlib scikit-learn

The script downloads Fashion-MNIST automatically the first time it is run.
"""

from __future__ import annotations

import copy
import csv
import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_moons
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# -----------------------------------------------------------------------------
# 1. CONFIGURATION
# -----------------------------------------------------------------------------

SEED = 42
DATA_DIR = "./data"
OUTPUT_DIR = Path("./lab1_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Fashion-MNIST subset sizes. The first 7,000 selected images are used for
# training and the next 1,000 for validation. Testing uses 2,000 test images.
TRAIN_SIZE = 7000
VAL_SIZE = 1000
TEST_SIZE = 2000
BATCH_SIZE = 128
NUM_WORKERS = 0  # 0 is reliable on Windows and avoids multiprocessing issues.

# Epoch settings. Increase these for a more extensive experiment if needed.
GD_EPOCHS = 100
OPTIMIZER_EPOCHS = 8
REGULARIZATION_EPOCHS = 12
ARCHITECTURE_EPOCHS = 12
EARLY_STOPPING_PATIENCE = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------------------------
# 2. REPRODUCIBILITY AND DEVICE
# -----------------------------------------------------------------------------

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Deterministic algorithms improve reproducibility. The benchmark may run
    # slightly slower, but the comparisons become easier to reproduce.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed()
print("=" * 78)
print("SELF LEARNING - LAB PROGRAM 1")
print(f"PyTorch version : {torch.__version__}")
print(f"Device          : {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU             : {torch.cuda.get_device_name(0)}")
print("=" * 78)


# -----------------------------------------------------------------------------
# 3. BATCH, STOCHASTIC, AND MINI-BATCH GRADIENT DESCENT
# -----------------------------------------------------------------------------

class SmallMLP(nn.Module):
    """A small MLP used only for the 2D gradient-descent demonstration."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def demonstrate_gradient_descent() -> List[Dict[str, object]]:
    print("\n[1/4] Demonstrating gradient-descent variants on a small dataset...")

    # A small, nonlinear two-class dataset.
    x, y = make_moons(n_samples=300, noise=0.20, random_state=SEED)
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    # Fixed train/test split.
    permutation = torch.randperm(len(x), generator=torch.Generator().manual_seed(SEED))
    train_indices = permutation[:240]
    test_indices = permutation[240:]
    x_train, y_train = x[train_indices], y[train_indices]
    x_test, y_test = x[test_indices], y[test_indices]

    methods = {
        "Batch GD": len(x_train),
        "Stochastic GD": 1,
        "Mini-batch GD (32)": 32,
    }
    histories: Dict[str, List[float]] = {}
    summary: List[Dict[str, object]] = []

    for method_name, batch_size in methods.items():
        set_seed(SEED)  # Same initial parameter values for each method.
        model = SmallMLP()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        criterion = nn.CrossEntropyLoss()
        epoch_losses: List[float] = []

        for epoch in range(GD_EPOCHS):
            model.train()
            order = torch.randperm(len(x_train))

            for start in range(0, len(x_train), batch_size):
                batch_indices = order[start : start + batch_size]
                xb = x_train[batch_indices]
                yb = y_train[batch_indices]

                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                epoch_loss = criterion(model(x_train), y_train).item()
            epoch_losses.append(epoch_loss)

        model.eval()
        with torch.no_grad():
            predictions = model(x_test).argmax(dim=1)
            test_accuracy = (predictions == y_test).float().mean().item()

        histories[method_name] = epoch_losses
        summary.append(
            {
                "Method": method_name,
                "Batch size": batch_size,
                "Final train loss": epoch_losses[-1],
                "Test accuracy": test_accuracy,
            }
        )

    plt.figure(figsize=(9, 5))
    for method_name, losses in histories.items():
        plt.plot(range(1, GD_EPOCHS + 1), losses, label=method_name)
    plt.xlabel("Epoch")
    plt.ylabel("Training cross-entropy loss")
    plt.title("Batch vs Stochastic vs Mini-batch Gradient Descent")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "gradient_descent_comparison.png", dpi=160)
    plt.close()

    print("Gradient-descent results:")
    for row in summary:
        print(
            f"  {row['Method']:<22} | batch={row['Batch size']:<3} "
            f"| train loss={row['Final train loss']:.4f} "
            f"| test accuracy={row['Test accuracy']:.4f}"
        )
    print(f"Saved: {OUTPUT_DIR / 'gradient_descent_comparison.png'}")
    return summary


# -----------------------------------------------------------------------------
# 4. FASHION-MNIST DATA
# -----------------------------------------------------------------------------

NORMALIZE = transforms.Normalize((0.2860,), (0.3530,))

EVAL_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        NORMALIZE,
    ]
)

# Small translations, rotations, and scale changes are reasonable image
# augmentations for Fashion-MNIST. Horizontal flipping is intentionally omitted.
AUGMENTED_TRANSFORM = transforms.Compose(
    [
        transforms.RandomAffine(
            degrees=10,
            translate=(0.08, 0.08),
            scale=(0.90, 1.10),
        ),
        transforms.ToTensor(),
        NORMALIZE,
    ]
)


def prepare_fashion_mnist() -> Tuple[datasets.FashionMNIST, datasets.FashionMNIST, datasets.FashionMNIST, List[int], List[int], List[int]]:
    print("\nDownloading/loading Fashion-MNIST...")

    train_eval_dataset = datasets.FashionMNIST(
        root=DATA_DIR, train=True, download=True, transform=EVAL_TRANSFORM
    )
    train_aug_dataset = datasets.FashionMNIST(
        root=DATA_DIR, train=True, download=True, transform=AUGMENTED_TRANSFORM
    )
    test_dataset = datasets.FashionMNIST(
        root=DATA_DIR, train=False, download=True, transform=EVAL_TRANSFORM
    )

    if TRAIN_SIZE + VAL_SIZE > len(train_eval_dataset):
        raise ValueError("TRAIN_SIZE + VAL_SIZE is larger than the Fashion-MNIST training set.")
    if TEST_SIZE > len(test_dataset):
        raise ValueError("TEST_SIZE is larger than the Fashion-MNIST test set.")

    generator = torch.Generator().manual_seed(SEED)
    train_permutation = torch.randperm(len(train_eval_dataset), generator=generator).tolist()
    train_indices = train_permutation[:TRAIN_SIZE]
    val_indices = train_permutation[TRAIN_SIZE : TRAIN_SIZE + VAL_SIZE]

    test_generator = torch.Generator().manual_seed(SEED + 1)
    test_indices = torch.randperm(len(test_dataset), generator=test_generator)[:TEST_SIZE].tolist()

    print(f"Training images   : {len(train_indices)}")
    print(f"Validation images : {len(val_indices)}")
    print(f"Testing images    : {len(test_indices)}")
    return (
        train_eval_dataset,
        train_aug_dataset,
        test_dataset,
        train_indices,
        val_indices,
        test_indices,
    )


TRAIN_EVAL_DATASET, TRAIN_AUG_DATASET, TEST_DATASET, TRAIN_INDICES, VAL_INDICES, TEST_INDICES = prepare_fashion_mnist()


def make_loaders(use_augmentation: bool = False, seed: int = SEED):
    selected_train_dataset = TRAIN_AUG_DATASET if use_augmentation else TRAIN_EVAL_DATASET

    train_subset = Subset(selected_train_dataset, TRAIN_INDICES)
    val_subset = Subset(TRAIN_EVAL_DATASET, VAL_INDICES)
    test_subset = Subset(TEST_DATASET, TEST_INDICES)

    generator = torch.Generator().manual_seed(seed)
    common_loader_args = {
        "num_workers": NUM_WORKERS,
        "pin_memory": DEVICE.type == "cuda",
    }

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        **common_loader_args,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        **common_loader_args,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        **common_loader_args,
    )
    return train_loader, val_loader, test_loader


# -----------------------------------------------------------------------------
# 5. PLAIN MLP AND RESIDUAL MLP
# -----------------------------------------------------------------------------

class PlainMLP(nn.Module):
    """Three hidden fully connected layers followed by a 10-class output."""

    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.classifier = nn.Linear(256, 10)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        return self.classifier(x)


class ResidualBlock(nn.Module):
    """A two-layer residual block with an identity skip connection."""

    def __init__(self, width: int = 256, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = F.relu(self.fc1(x))
        residual = self.dropout(residual)
        residual = self.fc2(residual)
        residual = self.dropout(residual)
        return F.relu(x + residual)


class ResidualMLP(nn.Module):
    """Same number of hidden linear transformations as PlainMLP."""

    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.input_layer = nn.Linear(28 * 28, 256)
        self.input_dropout = nn.Dropout(dropout)
        self.residual_block = ResidualBlock(256, dropout)
        self.classifier = nn.Linear(256, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, start_dim=1)
        x = self.input_dropout(F.relu(self.input_layer(x)))
        x = self.residual_block(x)
        return self.classifier(x)


# -----------------------------------------------------------------------------
# 6. OPTIMIZERS, LEARNING-RATE SCHEDULES, TRAINING, AND EVALUATION
# -----------------------------------------------------------------------------

def create_optimizer(
    optimizer_name: str,
    model: nn.Module,
    learning_rate: float,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    name = optimizer_name.lower()

    if name == "sgd_momentum":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
        )
    if name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    if name == "rmsprop":
        return torch.optim.RMSprop(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def evaluate(model: nn.Module, loader: DataLoader) -> Tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer_name: str = "adam",
    learning_rate: float = 0.001,
    scheduler_name: str = "constant",
    weight_decay: float = 0.0,
    epochs: int = 10,
    patience: int | None = EARLY_STOPPING_PATIENCE,
) -> Dict[str, object]:
    model = model.to(DEVICE)
    optimizer = create_optimizer(optimizer_name, model, learning_rate, weight_decay)
    criterion = nn.CrossEntropyLoss()

    scheduler = None
    if scheduler_name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(2, epochs // 3),
            gamma=0.5,
        )
    elif scheduler_name != "constant":
        raise ValueError("scheduler_name must be 'constant' or 'step'.")

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "learning_rate": [],
    }

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    stopped_early = False

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_accuracy = correct / total
        val_loss, val_accuracy = evaluate(model, val_loader)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        history["learning_rate"].append(current_lr)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if scheduler is not None:
            scheduler.step()

        print(
            f"    Epoch {epoch:02d}/{epochs} | "
            f"train loss={train_loss:.4f}, train acc={train_accuracy:.4f} | "
            f"val loss={val_loss:.4f}, val acc={val_accuracy:.4f} | "
            f"lr={current_lr:.6f}"
        )

        if patience is not None and epochs_without_improvement >= patience:
            stopped_early = True
            print(f"    Early stopping: validation loss did not improve for {patience} epochs.")
            break

    # Restore the checkpoint with the lowest validation loss.
    model.load_state_dict(best_state)
    best_val_loss, best_val_accuracy = evaluate(model, val_loader)

    return {
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "epochs_ran": len(history["train_loss"]),
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_accuracy,
        "stopped_early": stopped_early,
    }


def run_trial(
    name: str,
    *,
    model_kind: str = "plain",
    dropout: float = 0.0,
    optimizer_name: str = "adam",
    learning_rate: float = 0.001,
    scheduler_name: str = "constant",
    weight_decay: float = 0.0,
    use_augmentation: bool = False,
    epochs: int = 10,
    patience: int | None = EARLY_STOPPING_PATIENCE,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    print(f"\n--- Experiment: {name} ---")
    set_seed(SEED)
    train_loader, val_loader, test_loader = make_loaders(use_augmentation, seed=SEED)

    if model_kind == "plain":
        model = PlainMLP(dropout=dropout)
    elif model_kind == "residual":
        model = ResidualMLP(dropout=dropout)
    else:
        raise ValueError("model_kind must be 'plain' or 'residual'.")

    result = train_model(
        model,
        train_loader,
        val_loader,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        scheduler_name=scheduler_name,
        weight_decay=weight_decay,
        epochs=epochs,
        patience=patience,
    )

    test_loss, test_accuracy = evaluate(result["model"], test_loader)
    row: Dict[str, object] = {
        "Experiment": name,
        "Model": model_kind,
        "Optimizer": optimizer_name,
        "LR schedule": scheduler_name,
        "Learning rate": learning_rate,
        "Dropout": dropout,
        "L2 weight decay": weight_decay,
        "Data augmentation": use_augmentation,
        "Best epoch": result["best_epoch"],
        "Epochs run": result["epochs_ran"],
        "Early stopped": result["stopped_early"],
        "Validation accuracy": result["best_val_accuracy"],
        "Test accuracy": test_accuracy,
        "Test loss": test_loss,
    }
    print(
        f"Result: validation accuracy={row['Validation accuracy']:.4f}, "
        f"test accuracy={test_accuracy:.4f}, test loss={test_loss:.4f}, "
        f"best epoch={result['best_epoch']}"
    )
    return row, result


# -----------------------------------------------------------------------------
# 7. PLOTTING AND REPORT UTILITIES
# -----------------------------------------------------------------------------

def plot_learning_curves(history: Dict[str, List[float]], filename: str, title: str) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, history["train_loss"], label="Training loss")
    axes[0].plot(epochs, history["val_loss"], label="Validation loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_accuracy"], label="Training accuracy")
    axes[1].plot(epochs, history["val_accuracy"], label="Validation accuracy")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=160)
    plt.close(fig)


def plot_group_comparison(
    rows: List[Dict[str, object]],
    selected_names: List[str],
    filename: str,
    title: str,
) -> None:
    selected = [row for row in rows if row["Experiment"] in selected_names]
    if not selected:
        return

    names = [str(row["Experiment"]) for row in selected]
    values = [float(row["Validation accuracy"]) for row in selected]

    plt.figure(figsize=(max(8, len(names) * 1.5), 5))
    bars = plt.bar(names, values)
    plt.ylabel("Best validation accuracy")
    plt.ylim(0, 1.0)
    plt.title(title)
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.3)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=160)
    plt.close()


def save_results_csv(rows: List[Dict[str, object]], filename: str = "fashion_mnist_results.csv") -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with (OUTPUT_DIR / filename).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_results_table(rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    headers = ["Experiment", "Best epoch", "Val acc", "Test acc", "Test loss", "Early stopped"]
    print("\n" + "=" * 100)
    print("FASHION-MNIST EXPERIMENT SUMMARY")
    print("=" * 100)
    print(f"{headers[0]:<27} {headers[1]:>10} {headers[2]:>10} {headers[3]:>10} {headers[4]:>10} {headers[5]:>15}")
    print("-" * 100)
    for row in rows:
        print(
            f"{str(row['Experiment']):<27} "
            f"{int(row['Best epoch']):>10} "
            f"{float(row['Validation accuracy']):>10.4f} "
            f"{float(row['Test accuracy']):>10.4f} "
            f"{float(row['Test loss']):>10.4f} "
            f"{str(row['Early stopped']):>15}"
        )
    print("=" * 100)


# -----------------------------------------------------------------------------
# 8. MAIN FASHION-MNIST EXPERIMENTS
# -----------------------------------------------------------------------------

def run_fashion_mnist_experiments() -> List[Dict[str, object]]:
    all_rows: List[Dict[str, object]] = []
    histories: Dict[str, Dict[str, object]] = {}

    # ------------------------------------------------------------------
    # Experiment A: momentum and adaptive optimizers, constant vs StepLR
    # ------------------------------------------------------------------
    print("\n[2/4] Comparing optimizers and learning-rate schedules...")
    optimizer_settings = [
        ("sgd_momentum", 0.05),
        ("adam", 0.001),
        ("rmsprop", 0.001),
    ]
    optimizer_names: List[str] = []

    for optimizer_name, learning_rate in optimizer_settings:
        for scheduler_name in ("constant", "step"):
            name = f"{optimizer_name}_{scheduler_name}"
            row, result = run_trial(
                name,
                optimizer_name=optimizer_name,
                learning_rate=learning_rate,
                scheduler_name=scheduler_name,
                epochs=OPTIMIZER_EPOCHS,
                patience=EARLY_STOPPING_PATIENCE,
            )
            all_rows.append(row)
            histories[name] = result
            optimizer_names.append(name)

    plot_group_comparison(
        all_rows,
        optimizer_names,
        "optimizer_and_schedule_comparison.png",
        "Optimizer and Learning-Rate Schedule Comparison",
    )

    # ------------------------------------------------------------------
    # Experiment B: overfitting and regularization methods
    # ------------------------------------------------------------------
    print("\n[3/4] Investigating overfitting and regularization...")
    regularization_configs = [
        {
            "name": "regularization_baseline",
            "dropout": 0.0,
            "weight_decay": 0.0,
            "use_augmentation": False,
        },
        {
            "name": "dropout_0.3",
            "dropout": 0.3,
            "weight_decay": 0.0,
            "use_augmentation": False,
        },
        {
            "name": "L2_weight_decay_1e-4",
            "dropout": 0.0,
            "weight_decay": 1e-4,
            "use_augmentation": False,
        },
        {
            "name": "data_augmentation",
            "dropout": 0.0,
            "weight_decay": 0.0,
            "use_augmentation": True,
        },
    ]

    regularization_names: List[str] = []
    for config in regularization_configs:
        name = str(config["name"])
        row, result = run_trial(
            name,
            dropout=float(config["dropout"]),
            weight_decay=float(config["weight_decay"]),
            use_augmentation=bool(config["use_augmentation"]),
            optimizer_name="adam",
            learning_rate=0.001,
            scheduler_name="constant",
            epochs=REGULARIZATION_EPOCHS,
            patience=EARLY_STOPPING_PATIENCE,
        )
        all_rows.append(row)
        histories[name] = result
        regularization_names.append(name)

    # This baseline plot lets you inspect whether training loss continues to
    # decrease while validation loss starts increasing (a sign of overfitting).
    plot_learning_curves(
        histories["regularization_baseline"]["history"],
        "baseline_overfitting_curves.png",
        "Baseline MLP: Training vs Validation",
    )
    plot_group_comparison(
        all_rows,
        regularization_names,
        "regularization_comparison.png",
        "Dropout, L2 Regularization, and Data Augmentation",
    )

    # ------------------------------------------------------------------
    # Experiment C: plain MLP vs residual MLP
    # ------------------------------------------------------------------
    print("\n[4/4] Comparing plain MLP and residual MLP...")
    architecture_names = ["plain_mlp", "residual_mlp"]
    for model_kind, name in (("plain", "plain_mlp"), ("residual", "residual_mlp")):
        row, result = run_trial(
            name,
            model_kind=model_kind,
            optimizer_name="adam",
            learning_rate=0.001,
            scheduler_name="constant",
            dropout=0.0,
            weight_decay=0.0,
            use_augmentation=False,
            epochs=ARCHITECTURE_EPOCHS,
            patience=EARLY_STOPPING_PATIENCE,
        )
        all_rows.append(row)
        histories[name] = result

    plot_group_comparison(
        all_rows,
        architecture_names,
        "plain_vs_residual_mlp.png",
        "Plain MLP vs Residual MLP",
    )

    return all_rows


# -----------------------------------------------------------------------------
# 9. PROGRAM ENTRY POINT
# -----------------------------------------------------------------------------

def main() -> None:
    gd_summary = demonstrate_gradient_descent()
    fashion_rows = run_fashion_mnist_experiments()

    print_results_table(fashion_rows)
    save_results_csv(fashion_rows)

    print("\nGradient-descent summary:")
    for row in gd_summary:
        print(row)

    print("\nAll experiments completed.")
    print(f"Results and plots are saved in: {OUTPUT_DIR.resolve()}")
    print("Files generated:")
    for path in sorted(OUTPUT_DIR.iterdir()):
        print(f"  - {path.name}")


if __name__ == "__main__":
    main()