# Time Series Change-Point Detection with Deep Learning

A seminar/research project for **change-point detection (CPD) in real human-activity time series** using a 1D convolutional neural network (1D-CNN).

The project is inspired by the paper:

> **Automatic Change-Point Detection in Time Series via Deep Learning**  
> Jie Li, Paul Fearnhead, Piotr Fryzlewicz, Tengyao Wang  
> Journal of the Royal Statistical Society: Series B, 2024

This repository does **not** claim to reproduce the paper's HASC experiment exactly.  
Instead, it evaluates the same general idea—learning change-point detection from labeled data—on the **UCI Human Activity Recognition (UCI HAR)** dataset.

---

## Project goal

The original UCI HAR task is activity classification:

```text
time-series window -> activity label
```

In this project, it is reformulated as a change-point detection problem:

```text
activity A -> activity B
             ^
         change point
```

A change point is therefore derived whenever two consecutive activity labels are different.

Example:

```text
WALKING
WALKING
WALKING
        <- Change Point
WALKING_UPSTAIRS
WALKING_UPSTAIRS
```

> Important: these change points are **derived from activity-label transitions**.  
> They are not independent sample-level CPD annotations provided directly by UCI HAR.

---

## Dataset

The real-data experiment uses the UCI HAR training subset available in this repository.

Files used:

```text
data/uci_har/
├── activity_labels.txt
├── subject_train.txt
├── y_train.txt
└── Inertial Signals/
    └── total_acc_x_train.txt
```

Current experiment characteristics:

- **7352** activity windows
- **128 samples** per window
- **21 subjects**
- **6 activities**
- one input channel: `total_acc_x`

Activities:

1. WALKING
2. WALKING_UPSTAIRS
3. WALKING_DOWNSTAIRS
4. SITTING
5. STANDING
6. LAYING

The current model is intentionally **univariate**, so only the X-axis total acceleration signal is used.

---

## Why UCI HAR instead of HASC?

The reference paper evaluates real human-activity data using **HASC (Human Activity Sensing Consortium)**.

For this project, the raw HASC data used by the paper was not reliably accessible from the referenced download path during implementation. Therefore, **UCI HAR** was selected as a public, real, reproducible, and same-domain alternative.

The goal is therefore:

> **to evaluate a deep-learning CPD pipeline on real human-activity sensor data, not to reproduce the paper's HASC results numerically.**

---

## Pipeline

The final real-data pipeline is:

```text
UCI HAR
   |
   v
Real Dataset Loading
   |
   v
Activity Labels -> Change Points
   |
   v
Subject-wise Signal Reconstruction
   |
   v
Ground-truth CP Generation
   |
   v
Subject-independent Train / Validation / Test Split
   |
   v
Window Generation
   |
   v
1D-CNN Training
   |
   v
Validation Threshold Tuning
   |
   v
Peak Detection
   |
   v
Held-out Test Evaluation
   |
   v
Precision / Recall / F1 + Plots + CSV Results
```

---

## Subject-wise split

The split is performed at the **subject level**, not at the window level.

This prevents the same person from appearing in both training and test data.

Final split:

- **12 subjects** for training
- **4 subjects** for validation
- **5 subjects** for testing

Test subjects:

```text
7, 28, 3, 22, 15
```

This means the final test evaluation is performed on **unseen subjects**.

---

## 1D-CNN model

The model receives a window of shape:

```text
(batch, 1, 128)
```

Architecture:

```text
Input
  |
  v
Conv1D: 1 -> 16, kernel=7
BatchNorm
ReLU
  |
  v
Conv1D: 16 -> 32, kernel=5
BatchNorm
ReLU
  |
  v
MaxPool1D(2)
  |
  v
Conv1D: 32 -> 64, kernel=3
BatchNorm
ReLU
  |
  v
Adaptive Average Pooling
  |
  v
Linear: 64 -> 32
ReLU
Dropout(0.2)
  |
  v
Linear: 32 -> 1
Sigmoid
  |
  v
Change-point score
```

Trainable parameters:

```text
11,265
```

---

## Training configuration

| Parameter               |                         Value |
| ----------------------- | ----------------------------: |
| Maximum epochs          |                            25 |
| Batch size              |                            64 |
| Learning rate           |                         0.001 |
| Optimizer               |                          Adam |
| Loss                    | Weighted Binary Cross Entropy |
| Random seed             |                            42 |
| Early-stopping patience |                             5 |
| Training window size    |                           128 |
| Training CP tolerance   |                            20 |
| Training step           |                             8 |

### Class imbalance

Training positive ratio:

```text
0.0217 ~= 2.17%
```

Positive class weight:

```text
45.03
```

---

## Early stopping and checkpointing

The final run produced:

```text
Epochs executed: 13
Best epoch: 8
Best validation loss: 0.617268
```

The best validation checkpoint is saved as:

```text
results/model_har.pt
```

---

## Threshold tuning

Thresholds from `0.30` to `0.95` were evaluated **only on the validation set**.

Best validation setting:

```text
Threshold = 0.90
Validation F1 = 0.4851
```

The same threshold was then applied to the held-out test set without further tuning.

Peak detection also uses:

```text
distance = 64
```

---

## Evaluation

Final evaluation tolerance:

```text
32 samples
```

Each true change point can be matched at most once.

Metrics:

- **Precision** — how many predicted CPs are correct
- **Recall** — how many real CPs are detected
- **F1** — harmonic mean of Precision and Recall

---

## Final real-data results

Held-out UCI HAR test set:

| Metric    |      Value |
| --------- | ---------: |
| Precision | **0.3383** |
| Recall    | **0.4783** |
| F1 Score  | **0.3870** |

Per-subject F1:

| Subject |    F1 |
| ------: | ----: |
|       7 | 0.385 |
|      28 | 0.417 |
|       3 | 0.276 |
|      22 | 0.387 |
|      15 | 0.471 |

The results show that the simplified 1D-CNN can detect part of the activity-transition change points on unseen subjects, but false positives and missed detections remain significant.

---

## Output files

The main real-data experiment generates:

```text
results/
├── model_har.pt
├── har_training_history.csv
├── har_loss_curve.png
├── har_threshold_tuning.csv
├── har_threshold_tuning.png
├── har_final_metrics.csv
├── har_subject_metrics.csv
├── har_example_signal_0.png
├── har_example_signal_1.png
└── har_example_signal_2.png
```

| File                       | Purpose                                                   |
| -------------------------- | --------------------------------------------------------- |
| `model_har.pt`             | best trained CNN checkpoint                               |
| `har_training_history.csv` | train/validation loss for each epoch                      |
| `har_loss_curve.png`       | visual training and validation-loss curve                 |
| `har_threshold_tuning.csv` | validation Precision/Recall/F1 for tested thresholds      |
| `har_threshold_tuning.png` | visual threshold-selection curve                          |
| `har_final_metrics.csv`    | final held-out test metrics                               |
| `har_subject_metrics.csv`  | metrics for each test subject                             |
| `har_example_signal_*.png` | real signals with true/predicted CPs and CNN score curves |

---

## Main source files

```text
src/
├── real_data_loader_har.py
├── dataset.py
├── model.py
├── train_har.py
├── inference.py
├── evaluate.py
└── main_har.py
```

---

## How to run

Install dependencies:

```bash
pip install -r requirements.txt
```

Check the dataset and CPD conversion:

```bash
python3 src/real_data_loader_har.py
```

Train the model:

```bash
python3 src/train_har.py
```

Run validation threshold tuning and final test evaluation:

```bash
python3 src/main_har.py
```

Final outputs are written to:

```text
results/
```

---

## Limitations

- UCI HAR is used instead of the HASC data from the reference paper.
- Only `total_acc_x` is used.
- CP ground truth is derived from activity-label transitions.
- The implemented CNN is simpler than the deep residual architecture used in the advanced experiments of the paper.
- Test F1 is moderate and there is still room for improvement.
- A direct CUSUM baseline comparison is not yet included.

Possible future extensions:

- CUSUM baseline comparison
- X/Y/Z multivariate accelerometer input
- deeper residual CNNs
- LSTM/GRU/Transformer models
- additional CP types and datasets

---

## Current status

The core real-data pipeline is complete:

```text
Real data
-> CPD conversion
-> subject-independent split
-> CNN training
-> early stopping
-> best checkpoint
-> validation threshold tuning
-> held-out test evaluation
-> plots and CSV results
```

The remaining work is mainly **analysis, seminar presentation, and comparison with the theoretical ideas from the reference paper**, especially CUSUM and the relationship between classical change-point tests and neural networks.
