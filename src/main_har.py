import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from real_data_loader_har import get_har_splits
from inference import DLChangePointDetector
from evaluate import (
    evaluate_on_dataset,
    precision_recall_f1
)


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


# ==========================================================
# Threshold tuning
# ==========================================================

def tune_threshold(
    detector,
    val_set,
    tolerance=32,
    distance=64
):

    thresholds = np.arange(
        0.30,
        0.96,
        0.05
    )

    rows = []

    print("\nThreshold tuning on Validation Set")
    print("----------------------------------")

    best_threshold = None
    best_f1 = -1

    for threshold in thresholds:

        precision, recall, f1 = (
            evaluate_on_dataset(
                detector,
                val_set,
                tolerance=tolerance,
                height=float(threshold),
                distance=distance
            )
        )

        rows.append({
            "threshold": float(threshold),
            "precision": precision,
            "recall": recall,
            "f1": f1
        })

        print(
            f"Threshold={threshold:.2f} "
            f"P={precision:.4f} "
            f"R={recall:.4f} "
            f"F1={f1:.4f}"
        )

        if f1 > best_f1:

            best_f1 = f1

            best_threshold = float(
                threshold
            )

    tuning_df = pd.DataFrame(rows)

    tuning_path = os.path.join(
        RESULTS_DIR,
        "har_threshold_tuning.csv"
    )

    tuning_df.to_csv(
        tuning_path,
        index=False
    )

    # --------------------------------------------------
    # Threshold plot
    # --------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        tuning_df["threshold"],
        tuning_df["precision"],
        marker="o",
        label="Precision"
    )

    ax.plot(
        tuning_df["threshold"],
        tuning_df["recall"],
        marker="o",
        label="Recall"
    )

    ax.plot(
        tuning_df["threshold"],
        tuning_df["f1"],
        marker="o",
        label="F1"
    )

    ax.axvline(
        best_threshold,
        linestyle="--",
        label=f"Best threshold = {best_threshold:.2f}"
    )

    ax.set_xlabel(
        "Decision Threshold"
    )

    ax.set_ylabel(
        "Score"
    )

    ax.set_title(
        "UCI HAR - Threshold Selection on Validation Set"
    )

    ax.legend()

    plt.tight_layout()

    plot_path = os.path.join(
        RESULTS_DIR,
        "har_threshold_tuning.png"
    )

    plt.savefig(
        plot_path,
        dpi=150
    )

    plt.close()

    print("\nBest Validation Threshold")
    print("-------------------------")
    print(
        f"Threshold: {best_threshold:.2f}"
    )

    print(
        f"Validation F1: {best_f1:.4f}"
    )

    return best_threshold


# ==========================================================
# Plot example signals
# ==========================================================

def save_example_plots(
    detector,
    test_set,
    threshold,
    tolerance=32,
    distance=64,
    max_examples=3
):

    print("\nSaving HAR example plots...")

    for idx, (signal, true_cps) in enumerate(
        test_set[:max_examples]
    ):

        pred_cps, scores = detector.predict(
            signal,
            height=threshold,
            distance=distance
        )

        precision, recall, f1 = (
            precision_recall_f1(
                true_cps,
                pred_cps,
                tolerance=tolerance
            )
        )

        fig, axes = plt.subplots(
            2,
            1,
            figsize=(14, 7),
            sharex=True
        )

        # --------------------------------------------------
        # Signal
        # --------------------------------------------------

        axes[0].plot(
            signal,
            linewidth=0.7
        )

        for j, cp in enumerate(true_cps):

            axes[0].axvline(
                cp,
                linestyle="-",
                alpha=0.7,
                label="True CP"
                if j == 0 else None
            )

        for j, cp in enumerate(pred_cps):

            axes[0].axvline(
                cp,
                linestyle="--",
                alpha=0.7,
                label="Predicted CP"
                if j == 0 else None
            )

        axes[0].set_title(
            f"UCI HAR Test Signal #{idx} | "
            f"P={precision:.3f} "
            f"R={recall:.3f} "
            f"F1={f1:.3f}"
        )

        axes[0].set_ylabel(
            "Normalized Acceleration"
        )

        axes[0].legend()

        # --------------------------------------------------
        # Probability Curve
        # --------------------------------------------------

        axes[1].plot(
            scores,
            linewidth=0.8
        )

        axes[1].axhline(
            threshold,
            linestyle="--",
            label=f"Threshold = {threshold:.2f}"
        )

        axes[1].set_xlabel(
            "Time Sample"
        )

        axes[1].set_ylabel(
            "CP Probability"
        )

        axes[1].set_title(
            "Change-Point Probability Curve"
        )

        axes[1].legend()

        plt.tight_layout()

        out_path = os.path.join(
            RESULTS_DIR,
            f"har_example_signal_{idx}.png"
        )

        plt.savefig(
            out_path,
            dpi=150
        )

        plt.close()

        print(
            "Saved:",
            out_path
        )


# ==========================================================
# Per-subject evaluation
# ==========================================================

def evaluate_per_subject(
    detector,
    test_set,
    test_subjects,
    threshold,
    tolerance=32,
    distance=64
):

    rows = []

    for subject_id, (
        signal,
        true_cps
    ) in zip(
        test_subjects,
        test_set
    ):

        pred_cps, _ = detector.predict(
            signal,
            height=threshold,
            distance=distance
        )

        precision, recall, f1 = (
            precision_recall_f1(
                true_cps,
                pred_cps,
                tolerance=tolerance
            )
        )

        rows.append({
            "subject": subject_id,
            "signal_length": len(signal),
            "true_change_points": len(true_cps),
            "predicted_change_points": len(pred_cps),
            "precision": precision,
            "recall": recall,
            "f1": f1
        })

    df = pd.DataFrame(rows)

    path = os.path.join(
        RESULTS_DIR,
        "har_subject_metrics.csv"
    )

    df.to_csv(
        path,
        index=False
    )

    return df


# ==========================================================
# Main
# ==========================================================

def main():

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            "model_har.pt not found. "
            "Run: python3 src/train_har.py"
        )

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    train_set, val_set, test_set, split_info = (
        get_har_splits(seed=42)
    )

    print("UCI HAR Subject Split")
    print("---------------------")

    print(
        "Train:",
        split_info["train_subjects"]
    )

    print(
        "Validation:",
        split_info["val_subjects"]
    )

    print(
        "Test:",
        split_info["test_subjects"]
    )

    # --------------------------------------------------
    # Detector
    # --------------------------------------------------

    detector = DLChangePointDetector(
        model_path=MODEL_PATH,
        window_size=128
    )

    # --------------------------------------------------
    # Threshold selection ONLY on Validation
    # --------------------------------------------------

    best_threshold = tune_threshold(
        detector,
        val_set,
        tolerance=32,
        distance=64
    )

    # --------------------------------------------------
    # Final Test Evaluation
    # --------------------------------------------------

    print("\nFinal evaluation on held-out Test Set")
    print("-------------------------------------")

    precision, recall, f1 = (
        evaluate_on_dataset(
            detector,
            test_set,
            tolerance=32,
            height=best_threshold,
            distance=64
        )
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )

    # --------------------------------------------------
    # Save final metrics
    # --------------------------------------------------

    final_df = pd.DataFrame([
        {
            "dataset": "UCI HAR",
            "threshold": best_threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    ])

    metrics_path = os.path.join(
        RESULTS_DIR,
        "har_final_metrics.csv"
    )

    final_df.to_csv(
        metrics_path,
        index=False
    )

    # --------------------------------------------------
    # Per-subject results
    # --------------------------------------------------

    subject_df = evaluate_per_subject(
        detector,
        test_set,
        split_info["test_subjects"],
        threshold=best_threshold,
        tolerance=32,
        distance=64
    )

    print("\nPer-subject results")
    print("-------------------")

    print(
        subject_df.to_string(index=False)
    )

    # --------------------------------------------------
    # Example plots
    # --------------------------------------------------

    save_example_plots(
        detector,
        test_set,
        threshold=best_threshold,
        tolerance=32,
        distance=64,
        max_examples=3
    )

    print("\nDONE")
    print("----")
    print(
        "All HAR results saved in:",
        RESULTS_DIR
    )


if __name__ == "__main__":
    main()