# Time Series Change-Point Detection with Deep Learning

A small research project that trains a 1D-CNN to detect **change points** (abrupt mean shifts) in synthetic time series, and evaluates how well it generalizes and how it degrades under noise.

## What this project does

1. **Generates synthetic data** (`data_generation.py`): signals made of several segments, each with a different mean, glued together with Gaussian noise. The join between two segments is a ground-truth change point — this gives exact labels, which is hard to get from real-world data.
2. **Turns signals into a supervised problem** (`dataset.py`): slides a fixed-size window (64 samples) over each signal and labels a window `1` if a true change point falls near its center (within a tolerance), else `0`.
3. **Trains a 1D-CNN classifier** (`model.py`, `train.py`): the network takes a window and outputs the probability that its center is a change point. Class imbalance (change points are rare) is handled with a `pos_weight` in the loss. The best model (by validation loss) is checkpointed.
4. **Runs inference on full signals** (`inference.py`): slides the trained model over an entire signal to build a probability curve, then applies peak detection (`scipy.signal.find_peaks`) to turn that curve into a final list of predicted change points.
5. **Evaluates and reports** (`evaluate.py`, `main.py`): matches predicted vs. true change points within a tolerance to compute Precision/Recall/F1, saves example plots, and runs a noise-sensitivity sweep.

## Libraries used

| Library                             | What it's used for                                                                        |
| ----------------------------------- | ----------------------------------------------------------------------------------------- |
| `numpy`                             | Signal generation, array manipulation, computing scores/labels                            |
| `torch` (PyTorch)                   | Building and training the 1D-CNN, tensors, `DataLoader`                                   |
| `scipy` (`scipy.signal.find_peaks`) | Converting the model's probability curve into discrete predicted change points            |
| `pandas`                            | Saving metrics tables (`final_metrics.csv`, `sensitivity_noise.csv`)                      |
| `matplotlib`                        | Plotting example signals, probability curves, loss curve, and the noise-sensitivity curve |

## Results

**Held-out test set** (25 signals, tolerance = 15 samples):

| Precision | Recall | F1    |
| --------- | ------ | ----- |
| 0.940     | 0.896  | 0.912 |

**Example predictions** — the model correctly localizes change points on most signals (`example_signal_0`, `example_signal_2`: F1 = 1.00), but can miss or misplace points on messier signals (`example_signal_1`: F1 = 0.75), especially when segments are short or the mean shift is subtle.

**Training curve** (`loss_curve.png`): training loss decreases steadily; validation loss flattens around epoch 8–10 and then oscillates without a clear further decrease — a sign that ~10 epochs is roughly where the model stops meaningfully improving, and that more epochs mostly add noise rather than better generalization.

**Noise sensitivity** (`sensitivity_noise.png` / `.csv`): the model is robust up to moderate noise, but degrades clearly as noise grows:

| Noise std | Precision | Recall | F1    |
| --------- | --------- | ------ | ----- |
| 0.2       | 0.973     | 0.933  | 0.949 |
| 0.5       | 1.000     | 0.900  | 0.943 |
| 1.0       | 0.920     | 0.833  | 0.865 |
| 1.5       | 0.719     | 0.800  | 0.735 |
| 2.0       | 0.629     | 0.767  | 0.659 |
| 3.0       | 0.448     | 0.650  | 0.511 |

F1 drops from ~0.95 at low noise to ~0.51 at high noise (std = 3.0) — expected, since at high noise the mean shift between segments becomes hard to distinguish from within-segment fluctuation, even for a human eye.

## Project structure

```
.
├── data/            # (generated data, if cached to disk)
├── notebooks/        # exploratory notebooks
├── report/           # write-up / seminar report
├── results/          # model.pt, metrics CSVs, plots (this is what main.py produces)
├── src/               # data_generation.py, dataset.py, model.py, train.py, inference.py, evaluate.py, main.py
├── requirements.txt
└── README.md
```

## What's left to do

The pipeline is solid end-to-end (generate → train → infer → evaluate → report), but it's currently tested in a fairly narrow, favorable setting. Worth doing before calling it "done":

- **No comparison with classical baselines.** There's no CUSUM / Bayesian online CPD / `ruptures`-style comparison, so it's hard to say whether the CNN is actually better than simpler, non-DL methods — this is usually the first thing a reviewer will ask for.
- **Only one type of change is tested.** Every synthetic signal is a _mean shift_. Real-world change points also come from variance shifts, trend changes, or frequency changes — none of that is covered, so the reported F1 says more about "can it find pure mean shifts" than "can it detect change points" in general.
- **No real (or realistic) dataset.** Everything is synthetic and generated by the same process the model is trained on, which inflates results — it's the easiest possible test. Trying it on even one public benchmark (e.g. HASC, a well log, or a financial series with known regime shifts) would be the most convincing addition.
- **Single train/val/test split, single seed for the "final" run.** The main metrics come from one run; there's no variance estimate (e.g. across 5 seeds), so it's unclear how stable that 0.912 F1 really is.
- **No hyperparameter search.** Window size (64), tolerance (10/15), `height`/`distance` in peak detection, and the CNN's channel sizes are all fixed by hand — a quick ablation on at least window size and peak-detection threshold would strengthen the report.
- **Peak-detection thresholds are hardcoded** (`height=0.5`, `distance=30`) rather than tuned or justified — worth a sentence in the report on why these values, or a small grid search.

None of this means the current results are wrong — the pipeline and evaluation methodology (window-based training + tolerance-based matching) are both standard and reasonable choices. It's more that the current results describe a controlled, easy setting, and the report should be explicit about that scope rather than implying general CPD performance.
