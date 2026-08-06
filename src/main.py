"""
این فایل تمام مراحل قبلی را کنار هم می‌گذارد و خروجی نهایی برای گزارش تولید می‌کند:
    1) آموزش مدل (اگر از قبل train نشده)
    2) ارزیابی روی یک تست‌ست جدا
    3) پلات چند سیگنال نمونه با change point واقعی/پیش‌بینی‌شده
    4) تحلیل حساسیت به نویز (sensitivity analysis)
    5) ذخیره‌ی جدول نتایج نهایی
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

from data_generation import generate_dataset, generate_mean_shift_signal
from train import train_model
from inference import DLChangePointDetector
from evaluate import evaluate_on_dataset, precision_recall_f1

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(THIS_DIR, "..", "results")
MODEL_PATH = os.path.join(RESULTS_DIR, "model.pt")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---- 1) آموزش (اگر مدل موجود نیست) ----
    if not os.path.exists(MODEL_PATH):
        print(">> Training model...")
        train_model(epochs=25, save_path=MODEL_PATH)
    else:
        print(">> Found existing trained model, skipping training.")

    detector = DLChangePointDetector(model_path=MODEL_PATH, window_size=64)

    # ---- 2) ارزیابی روی تست‌ست جدا ----
    print("\n>> Evaluating on held-out test set...")
    test_set = generate_dataset(n_signals=25, seed=777)
    precision, recall, f1 = evaluate_on_dataset(detector, test_set, tolerance=15)
    print(f"Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}")

    pd.DataFrame([{"Precision": precision, "Recall": recall, "F1": f1}]).to_csv(
        os.path.join(RESULTS_DIR, "final_metrics.csv"), index=False)

    # ---- 3) پلات چند سیگنال نمونه ----
    print("\n>> Saving example plots...")
    for idx in range(3):
        signal, true_cps = test_set[idx]
        pred_cps, scores = detector.predict(signal)
        p, r, f1_i = precision_recall_f1(true_cps, pred_cps, tolerance=15)

        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axes[0].plot(signal, color="black", linewidth=0.8)
        for cp in true_cps:
            axes[0].axvline(cp, color="green", alpha=0.6, label="True CP" if cp == true_cps[0] else None)
        for cp in pred_cps:
            axes[0].axvline(cp, color="red", linestyle="--", alpha=0.9, label="Predicted CP" if cp == pred_cps[0] else None)
        axes[0].set_title(f"Signal #{idx}  (P={p:.2f}, R={r:.2f}, F1={f1_i:.2f})")
        axes[0].legend(loc="upper right")

        axes[1].plot(scores, color="purple")
        axes[1].axhline(0.5, color="gray", linestyle=":")
        axes[1].set_title("Model's change-point probability curve")

        plt.tight_layout()
        out_path = os.path.join(RESULTS_DIR, f"example_signal_{idx}.png")
        plt.savefig(out_path, dpi=130)
        plt.close()
        print(f"   saved {out_path}")

    # ---- 4) تحلیل حساسیت به نویز ----
    print("\n>> Sensitivity analysis (noise level vs F1)...")
    noise_levels = [0.2, 0.5, 1.0, 1.5, 2.0, 3.0]
    rows = []
    for noise in noise_levels:
        signals = [generate_mean_shift_signal(n_segments=5, noise_std=noise, seed=1000 + i)
                   for i in range(15)]
        p, r, f1_n = evaluate_on_dataset(detector, signals, tolerance=15)
        rows.append({"noise_std": noise, "Precision": p, "Recall": r, "F1": f1_n})
        print(f"   noise={noise:.1f}  F1={f1_n:.3f}")

    sens_df = pd.DataFrame(rows)
    sens_df.to_csv(os.path.join(RESULTS_DIR, "sensitivity_noise.csv"), index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sens_df.noise_std, sens_df.F1, marker="o", color="purple")
    ax.set_xlabel("Noise std")
    ax.set_ylabel("F1 score")
    ax.set_title("Sensitivity of DL model to noise level")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "sensitivity_noise.png"), dpi=130)
    plt.close()
    print(f"   saved {RESULTS_DIR}/sensitivity_noise.png")

    print("\nDONE. All results are in:", RESULTS_DIR)


if __name__ == "__main__":
    main()