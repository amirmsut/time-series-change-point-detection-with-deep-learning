# src/dataset.py
"""
STEP 2 — Dataset Preparation
=============================
چرا این مرحله لازم است؟
شبکه‌ی CNN ما ورودی با طول ثابت می‌خواهد (یک "پنجره" از سیگنال)، نه کل سیگنال
با طول متغیر. پس این ماژول سیگنال‌ها را به پنجره‌های کوچک با طول ثابت
(window_size) می‌شکند و برای هر پنجره یک برچسب می‌سازد:

    label = 1  اگر یک change point واقعی نزدیک مرکز پنجره باشد (در بازه‌ی tolerance)
    label = 0  در غیر این صورت

این دقیقاً همان کاری است که هر پروژه‌ی یادگیری عمیق روی سری‌زمانی باید انجام دهد:
"raw signal" -> "(X, y) های قابل آموزش".
"""
import numpy as np
import torch
from torch.utils.data import Dataset


def build_windows(dataset, window_size=64, tolerance=10, step=2):
    """
    dataset: لیستی از (signal, true_cps) — خروجی data_generation.py

    برمی‌گرداند:
        X: np.ndarray به شکل (N, window_size)
        y: np.ndarray به شکل (N,)   -> 0 یا 1
    """
    half = window_size // 2
    X, y = [], []

    for signal, true_cps in dataset:
        n = len(signal)
        for center in range(half, n - half, step):
            window = signal[center - half: center + half]
            label = int(any(abs(center - cp) <= tolerance for cp in true_cps))
            X.append(window)
            y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return X, y


class CPDWindowDataset(Dataset):
    """کلاس Dataset استاندارد PyTorch — برای استفاده در DataLoader."""

    def __init__(self, X, y):
        self.X = torch.from_numpy(X).unsqueeze(1)  # shape: (N, 1, window_size) -> channel=1 برای Conv1d
        self.y = torch.from_numpy(y).unsqueeze(1)  # shape: (N, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


if __name__ == "__main__":
    # تست سریع
    from data_generation import generate_dataset
    ds = generate_dataset(n_signals=5, seed=1)
    X, y = build_windows(ds, window_size=64, tolerance=10, step=2)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"Class balance -> positives: {y.sum():.0f} / {len(y)}  ({100*y.mean():.1f}%)")

    torch_ds = CPDWindowDataset(X, y)
    xb, yb = torch_ds[0]
    print(f"Single sample tensor shape: {xb.shape}, label: {yb.item()}")