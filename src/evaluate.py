"""
چرا این مرحله لازم است؟
باید عملکرد مدل را با یک عدد قابل‌مقایسه نشان بدهیم، نه فقط با نگاه کردن به پلات.
معیار استاندارد در ادبیات CPD: یک change point پیش‌بینی‌شده "درست" حساب می‌شود
اگر در فاصله‌ی `tolerance` از یک change point واقعی باشد (هر واقعی فقط یک‌بار
match می‌شود).
"""
import numpy as np


def match_points(true_cps, pred_cps, tolerance=15):
    true_cps, pred_cps = sorted(true_cps), sorted(pred_cps)
    matched_true = set()
    matched_pred = 0

    for p in pred_cps:
        best_j, best_dist = None, None
        for j, t in enumerate(true_cps):
            if j in matched_true:
                continue
            dist = abs(p - t)
            if dist <= tolerance and (best_dist is None or dist < best_dist):
                best_dist, best_j = dist, j
        if best_j is not None:
            matched_true.add(best_j)
            matched_pred += 1

    tp = matched_pred
    fp = len(pred_cps) - tp
    fn = len(true_cps) - len(matched_true)
    return tp, fp, fn


def precision_recall_f1(true_cps, pred_cps, tolerance=15):
    tp, fp, fn = match_points(true_cps, pred_cps, tolerance)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def evaluate_on_dataset(detector, dataset, tolerance=15, height=0.5, distance=30):
    precisions, recalls, f1s = [], [], []
    for signal, true_cps in dataset:
        pred_cps, _ = detector.predict(signal, height=height, distance=distance)
        p, r, f1 = precision_recall_f1(true_cps, pred_cps, tolerance)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
    return float(np.mean(precisions)), float(np.mean(recalls)), float(np.mean(f1s))