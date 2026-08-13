import os

from real_data_loader_har import get_har_splits
from inference import DLChangePointDetector
from evaluate import evaluate_on_dataset


THIS_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    THIS_DIR,
    "..",
    "results",
    "model_har.pt"
)


def main():

    _, _, test_set, split_info = get_har_splits(
        seed=42
    )

    print("Test subjects:")
    print(split_info["test_subjects"])

    detector = DLChangePointDetector(
        model_path=MODEL_PATH,
        window_size=128
    )

    precision, recall, f1 = evaluate_on_dataset(
        detector,
        test_set,
        tolerance=32,
        height=0.5,
        distance=64
    )

    print("\nUCI HAR results")
    print("----------------")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")


if __name__ == "__main__":
    main()