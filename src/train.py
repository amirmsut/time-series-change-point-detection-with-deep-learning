"""
STEP 4 — Training
====================
چرا این مرحله لازم است؟
اینجا مدل واقعاً "یاد می‌گیرد"

1) Class imbalance: چون تعداد پنجره‌های label=1 خیلی کمتر از label=0 است،
   از `pos_weight` در BCELoss استفاده می‌کنیم تا مدل بیش از حد به سمت
   پیش‌بینی "بدون تغییر" (label=0) متمایل نشود.

2) Train/Validation split: برای اینکه بفهمیم مدل overfit نکرده، هر epoch را
   هم روی train و هم روی validation ارزیابی می‌کنیم.

3) Early stopping ساده: بهترین وزن مدل (بر اساس val loss) را ذخیره می‌کنیم.
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_generation import generate_dataset
from dataset import build_windows, CPDWindowDataset
from model import CPD_CNN

# مسیر results را نسبت به خود این فایل می‌سازیم (نه نسبت به جایی که از آن
# اسکریپت را صدا زده‌ای). این‌طوری چه از روت پروژه اجرا کنی چه از داخل src/,
# مسیر درست پیدا می‌شود.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SAVE_PATH = os.path.join(THIS_DIR, "..", "results", "model.pt")


def train_model(epochs=25, window_size=64, batch_size=64, lr=1e-3,
                 device=None, save_path=DEFAULT_SAVE_PATH):

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # ---- داده ----
    train_signals = generate_dataset(n_signals=60, seed=10)
    val_signals = generate_dataset(n_signals=15, seed=11)

    X_train, y_train = build_windows(train_signals, window_size=window_size, tolerance=10, step=2)
    X_val, y_val = build_windows(val_signals, window_size=window_size, tolerance=10, step=2)

    train_loader = DataLoader(CPDWindowDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(CPDWindowDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    # ---- مدل و loss با جبران class imbalance ----
    model = CPD_CNN().to(device)
    pos_ratio = y_train.mean()
    pos_weight = (1 - pos_ratio) / pos_ratio  # وزن بیشتر برای کلاس اقلیت (change point)
    print(f"Positive ratio in train: {pos_ratio:.3f} -> loss pos_weight: {pos_weight:.2f}")

    criterion = nn.BCELoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            weights = torch.where(yb == 1, pos_weight, 1.0)
            loss = (criterion(preds, yb) * weights).mean()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                weights = torch.where(yb == 1, pos_weight, 1.0)
                loss = (criterion(preds, yb) * weights).mean()
                val_losses.append(loss.item())

        train_loss = sum(train_losses) / len(train_losses)
        val_loss = sum(val_losses) / len(val_losses)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"Epoch {epoch:3d}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)

    print(f"\nBest model saved to {save_path} (val_loss={best_val_loss:.4f})")
    return model, history


if __name__ == "__main__":
    train_model(epochs=25)