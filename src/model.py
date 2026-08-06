# src/model.py
"""
STEP 3 — Model Architecture
=============================
چرا این معماری؟
یک شبکه‌ی 1D-CNN ساده طراحی می‌کنیم که یک پنجره از سیگنال (طول ثابت) را
می‌گیرد و یک عدد بین ۰ و ۱ (احتمال change point بودن) خروجی می‌دهد.

معماری:
    Input (1, window_size)
      -> Conv1d + ReLU + BatchNorm   (استخراج الگوهای محلی سطح پایین)
      -> Conv1d + ReLU + BatchNorm   (استخراج الگوهای پیچیده‌تر)
      -> MaxPool                     (کاهش ابعاد، افزایش receptive field)
      -> Conv1d + ReLU + BatchNorm
      -> Global Average Pooling      (خلاصه‌سازی کل پنجره در یک بردار)
      -> Fully Connected -> Sigmoid  (خروجی احتمال)

چرا Global Average Pooling؟
به‌جای Flatten (که وابسته به طول دقیق پنجره است)، GAP باعث می‌شود مدل به طول
ورودی حساس نباشد و over-fitting کمتری داشته باشد — یک انتخاب استاندارد در
شبکه‌های CNN مدرن.
"""
import torch
import torch.nn as nn


class CPD_CNN(nn.Module):
    def __init__(self, in_channels=1, base_channels=16):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),

            nn.Conv1d(base_channels, base_channels * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels * 4),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),  # Global Average Pooling
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels * 4, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x shape: (batch, 1, window_size)
        feat = self.features(x)
        out = self.classifier(feat)
        return out


if __name__ == "__main__":
    model = CPD_CNN()
    dummy = torch.randn(8, 1, 64)  # batch=8, channel=1, window=64
    out = model(dummy)
    print(f"Output shape: {out.shape}")  # باید (8, 1) باشد
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total trainable parameters: {n_params:,}")