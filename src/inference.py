"""
چرا این مرحله لازم است؟
مدل روی "پنجره‌های تکی" آموزش دیده، ولی در عمل ما یک سیگنال کامل داریم و
می‌خواهیم لیست change pointها را استخراج کنیم. این ماژول:

1) با یک پنجره‌ی لغزان، مدل را روی کل سیگنال اجرا می‌کند -> یک "منحنی احتمال"
   می‌سازد (هر نقطه از سیگنال یک عدد بین ۰ و ۱ می‌گیرد).
2) روی این منحنی peak detection انجام می‌دهد (نقاطی که هم بالاتر از یک آستانه
   هستند و هم از هم فاصله‌ی کافی دارند) -> این‌ها همان change pointهای نهایی هستند.
"""
import os
import numpy as np
import torch
from scipy.signal import find_peaks

from model import CPD_CNN

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class DLChangePointDetector:
    def __init__(self, model_path=None, window_size=64, device=None):
        model_path = model_path or os.path.join(THIS_DIR, "..", "results", "model.pt")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.window_size = window_size
        self.half = window_size // 2

        self.model = CPD_CNN().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    @torch.no_grad()
    def score_signal(self, signal, stride=1):
        """منحنی احتمال change-point بودن برای هر نقطه از سیگنال."""
        n = len(signal)
        scores = np.zeros(n)
        centers = list(range(self.half, n - self.half, stride))
        if not centers:
            return scores

        windows = np.stack([signal[c - self.half:c + self.half] for c in centers])
        windows_t = torch.from_numpy(windows.astype(np.float32)).unsqueeze(1).to(self.device)
        preds = self.model(windows_t).cpu().numpy().flatten()

        for c, p in zip(centers, preds):
            scores[c] = p
        return scores

    def predict(self, signal, height=0.5, distance=30, stride=1):
        """لیست change pointهای پیش‌بینی‌شده را برمی‌گرداند."""
        scores = self.score_signal(signal, stride=stride)
        peaks, _ = find_peaks(scores, height=height, distance=distance)
        return list(peaks), scores


if __name__ == "__main__":
    from data_generation import generate_mean_shift_signal

    detector = DLChangePointDetector(window_size=64)

    signal, true_cps = generate_mean_shift_signal(seed=999)
    pred_cps, scores = detector.predict(signal)

    print(f"True change points:      {true_cps}")
    print(f"Predicted change points: {pred_cps}")