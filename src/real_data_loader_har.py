import os
import numpy as np


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(THIS_DIR, "..", "data", "uci_har")


def load_raw_har(data_dir=DEFAULT_DATA_DIR):
    """
    فایل‌های UCI HAR را می‌خواند.

    Returns
    -------
    X : ndarray, shape (N, 128)
        پنجره‌های total_acc_x

    y : ndarray, shape (N,)
        Activity label هر پنجره

    subjects : ndarray, shape (N,)
        شناسه فرد مربوط به هر پنجره
    """

    signal_path = os.path.join(
        data_dir,
        "Inertial Signals",
        "total_acc_x_train.txt"
    )

    label_path = os.path.join(
        data_dir,
        "y_train.txt"
    )

    subject_path = os.path.join(
        data_dir,
        "subject_train.txt"
    )

    if not os.path.exists(signal_path):
        raise FileNotFoundError(signal_path)

    if not os.path.exists(label_path):
        raise FileNotFoundError(label_path)

    if not os.path.exists(subject_path):
        raise FileNotFoundError(subject_path)

    X = np.loadtxt(signal_path, dtype=np.float32)
    y = np.loadtxt(label_path, dtype=np.int64)
    subjects = np.loadtxt(subject_path, dtype=np.int64)

    if not (len(X) == len(y) == len(subjects)):
        raise ValueError(
            "تعداد signal windows، labels و subjects برابر نیست."
        )

    return X, y, subjects


def reconstruct_subject_signal(windows, hop_size=64):
    """
    پنجره‌های متوالی یک فرد را به یک سری زمانی بلند تبدیل می‌کند.

    پنجره اول به طور کامل قرار می‌گیرد.
    از پنجره‌های بعدی فقط hop_size نمونه انتهایی اضافه می‌شود.

    برای داده فعلی:
        window_size = 128
        hop_size    = 64
    """

    if len(windows) == 0:
        return np.array([], dtype=np.float32)

    signal = [windows[0]]

    for window in windows[1:]:
        signal.append(window[-hop_size:])

    return np.concatenate(signal).astype(np.float32)


def labels_to_change_points(labels, hop_size=64):
    """
    تغییر Activity Label را به Change Point تبدیل می‌کند.

    مثال:
        5, 5, 5, 4, 4, 1, 1
                 ^     ^

    دو Change Point تولید می‌شود.
    """

    labels = np.asarray(labels)

    transitions = np.where(
        labels[1:] != labels[:-1]
    )[0] + 1

    # محل تقریبی تغییر در سری زمانی بازسازی‌شده
    cps = [
        int(i * hop_size + hop_size // 2)
        for i in transitions
    ]

    return cps


def load_har_dataset(
    data_dir=DEFAULT_DATA_DIR,
    subject_ids=None,
    hop_size=64,
    normalize=True
):
    """
    UCI HAR را به فرمت مورد انتظار پروژه CPD تبدیل می‌کند:

        [
            (signal_1, [cp1, cp2, ...]),
            (signal_2, [cp1, cp2, ...]),
            ...
        ]

    هر subject یک time series مستقل محسوب می‌شود.
    """

    X, y, subjects = load_raw_har(data_dir)

    available_subjects = np.unique(subjects)

    if subject_ids is None:
        subject_ids = available_subjects

    dataset = []

    for subject_id in subject_ids:

        mask = subjects == subject_id

        subject_windows = X[mask]
        subject_labels = y[mask]

        if len(subject_windows) == 0:
            continue

        signal = reconstruct_subject_signal(
            subject_windows,
            hop_size=hop_size
        )

        true_cps = labels_to_change_points(
            subject_labels,
            hop_size=hop_size
        )

        if normalize:
            mean = signal.mean()
            std = signal.std()

            if std > 1e-8:
                signal = (signal - mean) / std

        dataset.append(
            (signal.astype(np.float32), true_cps)
        )

    return dataset


def get_har_splits(
    data_dir=DEFAULT_DATA_DIR,
    train_ratio=0.6,
    val_ratio=0.2,
    seed=42
):
    """
    Split در سطح Subject انجام می‌شود.

    بسیار مهم:
    داده‌های یک فرد نباید هم در Train و هم Test باشند.
    """

    _, _, subjects = load_raw_har(data_dir)

    unique_subjects = np.unique(subjects)

    rng = np.random.default_rng(seed)

    unique_subjects = unique_subjects.copy()
    rng.shuffle(unique_subjects)

    n = len(unique_subjects)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_subjects = unique_subjects[:n_train]
    val_subjects = unique_subjects[n_train:n_train + n_val]
    test_subjects = unique_subjects[n_train + n_val:]

    train = load_har_dataset(
        data_dir,
        subject_ids=train_subjects
    )

    val = load_har_dataset(
        data_dir,
        subject_ids=val_subjects
    )

    test = load_har_dataset(
        data_dir,
        subject_ids=test_subjects
    )

    return train, val, test, {
        "train_subjects": train_subjects.tolist(),
        "val_subjects": val_subjects.tolist(),
        "test_subjects": test_subjects.tolist(),
    }


if __name__ == "__main__":

    X, y, subjects = load_raw_har()

    print("Raw dataset")
    print("-----------")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("subjects shape:", subjects.shape)

    print("\nUnique subjects:")
    print(np.unique(subjects))

    print("\nUnique activities:")
    print(np.unique(y))

    dataset = load_har_dataset()

    print("\nCPD dataset")
    print("-----------")
    print("Number of signals:", len(dataset))

    for i, (signal, cps) in enumerate(dataset[:3]):
        print(
            f"Signal {i}: "
            f"length={len(signal)}, "
            f"change_points={len(cps)}, "
            f"cps={cps[:10]}"
        )

    train, val, test, info = get_har_splits()

    print("\nSplit")
    print("-----")
    print(info)
    print("Train signals:", len(train))
    print("Validation signals:", len(val))
    print("Test signals:", len(test))