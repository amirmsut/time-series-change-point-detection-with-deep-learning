"""
چرا این مرحله لازم است؟
هر مدل یادگیری عمیق نیاز به داده‌ی برچسب‌دار (label) دارد. برای CPD به سختی
می‌شود داده‌ی واقعی با change point دقیقاً مشخص پیدا کرد، پس یک generator
می‌سازیم که سیگنال‌های synthetic با ground-truth کاملاً معلوم تولید کند.

منطق تولید داده:
- هر سیگنال از چند "قطعه" (segment) تشکیل شده که هرکدام میانگین متفاوتی دارند.
- محل اتصال دو قطعه = change point واقعی.
- نویز گاوسی هم اضافه می‌کنیم تا شبیه داده‌ی واقعی‌تر شود.
"""
import numpy as np


def generate_mean_shift_signal(n_segments=5, seg_len_range=(60, 150),
                                noise_std=0.5, mean_range=(-5, 5), seed=None):
    """یک سیگنال با چند تغییر میانگین تولید می‌کند.

    Returns:
        signal: np.ndarray, شکل (N,)
        true_cps: list[int], اندیس‌های change point واقعی
    """
    rng = np.random.default_rng(seed)
    signal, true_cps = [], []
    current_len = 0

    means = rng.uniform(mean_range[0], mean_range[1], size=n_segments)
    for i in range(1, len(means)):
        while abs(means[i] - means[i - 1]) < 1.5:  # جلوگیری از change point نامحسوس
            means[i] = rng.uniform(mean_range[0], mean_range[1])

    for i in range(n_segments):
        seg_len = int(rng.integers(seg_len_range[0], seg_len_range[1]))
        segment = rng.normal(loc=means[i], scale=noise_std, size=seg_len)
        signal.append(segment)
        current_len += seg_len
        if i < n_segments - 1:
            true_cps.append(current_len)

    return np.concatenate(signal).astype(np.float32), true_cps


def generate_dataset(n_signals=50, seed=0, **kwargs):
    """لیستی از (signal, true_cps) برمی‌گرداند — برای train/val/test استفاده می‌شود."""
    rng = np.random.default_rng(seed)
    dataset = []
    for _ in range(n_signals):
        s = int(rng.integers(0, 1_000_000))
        n_seg = int(rng.integers(3, 7))
        noise = float(rng.uniform(0.3, 1.2))
        sig, cps = generate_mean_shift_signal(n_segments=n_seg, noise_std=noise, seed=s)
        dataset.append((sig, cps))
    return dataset


if __name__ == "__main__":
    # تست سریع: یک سیگنال بساز و چاپ کن
    sig, cps = generate_mean_shift_signal(seed=42)
    print(f"Signal length: {len(sig)}")
    print(f"True change points: {cps}")