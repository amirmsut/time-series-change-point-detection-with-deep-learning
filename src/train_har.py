"""
Training script for UCI HAR Change-Point Detection.

Pipeline:
    UCI HAR
        -> subject-level train/validation/test split
        -> reconstructed real time-series
        -> activity transitions as change points
        -> fixed-size CPD windows
        -> 1D-CNN
        -> weighted binary classification
        -> validation monitoring
        -> best-model checkpoint
        -> early stopping
        -> training history
        -> loss curve
"""

import os
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import build_windows, CPDWindowDataset
from model import CPD_CNN
from real_data_loader_har import get_har_splits


# ============================================================
# Paths
# ============================================================

THIS_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

RESULTS_DIR = os.path.join(
    THIS_DIR,
    "..",
    "results"
)

MODEL_PATH = os.path.join(
    RESULTS_DIR,
    "model_har.pt"
)

LOSS_CURVE_PATH = os.path.join(
    RESULTS_DIR,
    "har_loss_curve.png"
)

HISTORY_PATH = os.path.join(
    RESULTS_DIR,
    "har_training_history.csv"
)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed=42):
    """
    تنظیم Seed برای تکرارپذیری آزمایش‌ها.

    باعث می‌شود initialization مدل، shuffle داده‌ها و سایر
    بخش‌های تصادفی تا حد ممکن در اجراهای مختلف یکسان باشند.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # تنظیمات مربوط به CUDA برای deterministic بودن
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================
# Training
# ============================================================

def train_har_model(
    epochs=25,
    window_size=128,
    batch_size=64,
    lr=1e-3,
    tolerance=20,
    step=8,
    patience=5,
    seed=42,
    device=None
):
    """
    آموزش مدل CPD روی داده واقعی UCI HAR.

    Parameters
    ----------
    epochs : int
        حداکثر تعداد epochها.

    window_size : int
        طول پنجره ورودی CNN.

    batch_size : int
        اندازه batch.

    lr : float
        Learning rate.

    tolerance : int
        محدوده اطراف Change Point که یک window به عنوان
        positive در نظر گرفته می‌شود.

    step : int
        فاصله بین مرکز پنجره‌های متوالی هنگام ساخت dataset.

    patience : int
        تعداد epochهای مجاز بدون بهبود Validation Loss
        قبل از Early Stopping.

    seed : int
        Seed برای reproducibility.

    device : str | None
        cpu / cuda. اگر None باشد به صورت خودکار انتخاب می‌شود.

    Returns
    -------
    model
        مدل ساخته‌شده.

    history
        تاریخچه Train Loss و Validation Loss.
    """

    # --------------------------------------------------------
    # 1. Reproducibility
    # --------------------------------------------------------

    set_seed(seed)

    # --------------------------------------------------------
    # 2. Device
    # --------------------------------------------------------

    device = device or (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Using device:", device)
    print("Random seed:", seed)

    # --------------------------------------------------------
    # 3. Results directory
    # --------------------------------------------------------

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 4. Load UCI HAR
    #
    # Split در سطح Subject انجام می‌شود تا داده یک فرد
    # هم‌زمان در Train و Test قرار نگیرد.
    # --------------------------------------------------------

    train_signals, val_signals, test_signals, split_info = (
        get_har_splits(
            seed=seed
        )
    )

    print("\nSubject split:")
    print(split_info)

    print("\nSignals:")
    print("Train:", len(train_signals))
    print("Validation:", len(val_signals))
    print("Test:", len(test_signals))

    # --------------------------------------------------------
    # 5. Convert signals to CPD windows
    # --------------------------------------------------------

    X_train, y_train = build_windows(
        train_signals,
        window_size=window_size,
        tolerance=tolerance,
        step=step
    )

    X_val, y_val = build_windows(
        val_signals,
        window_size=window_size,
        tolerance=tolerance,
        step=step
    )

    print(
        "\nTraining windows:",
        X_train.shape
    )

    print(
        "Validation windows:",
        X_val.shape
    )

    # --------------------------------------------------------
    # 6. Class distribution
    # --------------------------------------------------------

    positive_ratio = float(
        y_train.mean()
    )

    print(
        "Train positive ratio:",
        positive_ratio
    )

    if positive_ratio <= 0:
        raise RuntimeError(
            "No positive change-point windows "
            "were found in the training set."
        )

    if positive_ratio >= 1:
        raise RuntimeError(
            "Training set contains only positive windows."
        )

    # --------------------------------------------------------
    # 7. Positive class weight
    #
    # Change Points بسیار کمتر از non-change windows هستند.
    # بنابراین برای کلاس positive وزن بیشتری در Loss قرار می‌دهیم.
    # --------------------------------------------------------

    pos_weight = (
        (1.0 - positive_ratio)
        / positive_ratio
    )

    print(
        "Positive weight:",
        pos_weight
    )

    # --------------------------------------------------------
    # 8. PyTorch datasets
    # --------------------------------------------------------

    train_dataset = CPDWindowDataset(
        X_train,
        y_train
    )

    val_dataset = CPDWindowDataset(
        X_val,
        y_val
    )

    # Generator جداگانه باعث می‌شود shuffle نیز
    # reproducible باشد.
    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # --------------------------------------------------------
    # 9. Model
    # --------------------------------------------------------

    model = CPD_CNN().to(device)

    total_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        "Trainable parameters:",
        f"{total_params:,}"
    )

    # --------------------------------------------------------
    # 10. Loss and optimizer
    # --------------------------------------------------------

    criterion = nn.BCELoss(
        reduction="none"
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    # وزن‌ها را یک بار می‌سازیم، نه در هر batch
    positive_weight_tensor = torch.tensor(
        pos_weight,
        dtype=torch.float32,
        device=device
    )

    negative_weight_tensor = torch.tensor(
        1.0,
        dtype=torch.float32,
        device=device
    )

    # --------------------------------------------------------
    # 11. Training state
    # --------------------------------------------------------

    best_val_loss = float("inf")
    best_epoch = 0

    epochs_without_improvement = 0

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": []
    }

    print("\nTraining")
    print("--------")

    # ========================================================
    # 12. Training loop
    # ========================================================

    for epoch in range(
        1,
        epochs + 1
    ):

        # ====================================================
        # TRAIN
        # ====================================================

        model.train()

        train_losses = []

        for xb, yb in train_loader:

            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            predictions = model(xb)

            # وزن بیشتر برای positive samples
            weights = torch.where(
                yb == 1,
                positive_weight_tensor,
                negative_weight_tensor
            )

            loss = (
                criterion(
                    predictions,
                    yb
                )
                * weights
            ).mean()

            loss.backward()

            optimizer.step()

            train_losses.append(
                loss.item()
            )

        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()

        val_losses = []

        with torch.no_grad():

            for xb, yb in val_loader:

                xb = xb.to(device)
                yb = yb.to(device)

                predictions = model(xb)

                weights = torch.where(
                    yb == 1,
                    positive_weight_tensor,
                    negative_weight_tensor
                )

                loss = (
                    criterion(
                        predictions,
                        yb
                    )
                    * weights
                ).mean()

                val_losses.append(
                    loss.item()
                )

        # ----------------------------------------------------
        # Epoch statistics
        # ----------------------------------------------------

        train_loss = (
            sum(train_losses)
            / len(train_losses)
        )

        val_loss = (
            sum(val_losses)
            / len(val_losses)
        )

        history["epoch"].append(
            epoch
        )

        history["train_loss"].append(
            train_loss
        )

        history["val_loss"].append(
            val_loss
        )

        print(
            f"Epoch {epoch:02d}/{epochs} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f}"
        )

        # ====================================================
        # BEST MODEL CHECKPOINT
        # ====================================================

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_epoch = epoch

            epochs_without_improvement = 0

            torch.save(
                model.state_dict(),
                MODEL_PATH
            )

            print(
                "  -> best model saved"
            )

        else:

            epochs_without_improvement += 1

            print(
                "  -> no improvement "
                f"({epochs_without_improvement}/{patience})"
            )

        # ====================================================
        # EARLY STOPPING
        # ====================================================

        if (
            epochs_without_improvement
            >= patience
        ):

            print(
                "\nEarly stopping triggered."
            )

            print(
                f"Validation loss did not improve "
                f"for {patience} consecutive epochs."
            )

            break

    # ========================================================
    # 13. Save training history
    # ========================================================

    history_df = pd.DataFrame(
        history
    )

    history_df.to_csv(
        HISTORY_PATH,
        index=False
    )

    # ========================================================
    # 14. Plot loss curve
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        history["epoch"],
        history["train_loss"],
        marker="o",
        markersize=3,
        label="Train Loss"
    )

    ax.plot(
        history["epoch"],
        history["val_loss"],
        marker="o",
        markersize=3,
        label="Validation Loss"
    )

    # محل بهترین epoch
    ax.axvline(
        best_epoch,
        linestyle="--",
        alpha=0.7,
        label=f"Best Epoch = {best_epoch}"
    )

    ax.set_xlabel(
        "Epoch"
    )

    ax.set_ylabel(
        "Weighted Binary Cross-Entropy Loss"
    )

    ax.set_title(
        "UCI HAR - Training and Validation Loss"
    )

    ax.legend()

    ax.grid(
        alpha=0.2
    )

    plt.tight_layout()

    plt.savefig(
        LOSS_CURVE_PATH,
        dpi=150
    )

    plt.close()

    # ========================================================
    # 15. Load best model again
    #
    # در صورت Early Stopping، model فعلی ممکن است مربوط
    # به آخرین epoch باشد. بنابراین بهترین checkpoint
    # را دوباره بارگذاری می‌کنیم.
    # ========================================================

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    model.eval()

    # ========================================================
    # 16. Final summary
    # ========================================================

    print("\nTraining finished.")
    print("------------------")

    print(
        "Epochs executed:",
        len(history["epoch"])
    )

    print(
        "Best epoch:",
        best_epoch
    )

    print(
        "Best validation loss:",
        f"{best_val_loss:.6f}"
    )

    print(
        "Model saved:",
        MODEL_PATH
    )

    print(
        "Loss curve saved:",
        LOSS_CURVE_PATH
    )

    print(
        "History saved:",
        HISTORY_PATH
    )

    return model, history


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    train_har_model(
        epochs=25,
        window_size=128,
        batch_size=64,
        lr=1e-3,
        tolerance=20,
        step=8,
        patience=5,
        seed=42
    )