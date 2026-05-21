"""
Evaluation protocols for piano difficulty prediction.

The published CIPI paper uses random splits stratified on difficulty. This
arguably leaks composer style across train/test: if the model sees ten Chopin
nocturnes in training, predicting the eleventh is partly a composer-ID task.
This module implements:

    - Leave-one-composer-out (LOCO) CV
    - Random stratified CV (for direct comparison to published numbers)
    - Calibration via reliability diagrams + ECE
    - Label-noise sensitivity (bootstrap with grade perturbation)
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .models import ordinal_metrics


def loco_cv(model, X: pd.DataFrame, y: np.ndarray, composers: np.ndarray,
            min_samples_per_composer: int = 3) -> pd.DataFrame:
    """Leave-one-composer-out cross-validation.

    For each composer with >= min_samples pieces, hold them all out and train
    on the rest. Returns per-fold metrics.

    This is the strict generalization test. Most published numbers are not
    this strict, so expect LOCO MAE to be 0.3-0.8 grades worse than random CV.
    """
    composers = np.asarray(composers)
    counts = pd.Series(composers).value_counts()
    eligible = counts[counts >= min_samples_per_composer].index.tolist()

    folds = []
    for comp in eligible:
        test_mask = composers == comp
        if test_mask.sum() == 0:
            continue
        m = deepcopy(model)
        m.fit(X[~test_mask], y[~test_mask])
        y_pred = m.predict(X[test_mask])
        metrics = ordinal_metrics(y[test_mask], y_pred)
        metrics['heldout_composer'] = comp
        metrics['n_test'] = int(test_mask.sum())
        folds.append(metrics)
    return pd.DataFrame(folds)


def stratified_cv(model, X: pd.DataFrame, y: np.ndarray,
                  n_splits: int = 5, random_state: int = 42) -> pd.DataFrame:
    """Random stratified k-fold. Use this for direct comparison to published
    benchmarks that use the same protocol."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = []
    for i, (tr, te) in enumerate(skf.split(X, y)):
        m = deepcopy(model)
        m.fit(X.iloc[tr], y[tr])
        y_pred = m.predict(X.iloc[te])
        metrics = ordinal_metrics(y[te], y_pred)
        metrics['fold'] = i
        folds.append(metrics)
    return pd.DataFrame(folds)


# -----------------------------------------------------------------------------
# Calibration
# -----------------------------------------------------------------------------

def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray,
                                n_bins: int = 10) -> float:
    """ECE over predicted-class confidence."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (conf >= bins[i]) & (conf < bins[i + 1])
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / len(y_true)) * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def reliability_curve(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10):
    """Return (bin_centers, accuracy_per_bin, confidence_per_bin) for plotting."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    centers, accs, confs = [], [], []
    for i in range(n_bins):
        mask = (conf >= bins[i]) & (conf < bins[i + 1])
        if mask.sum() == 0:
            continue
        centers.append((bins[i] + bins[i + 1]) / 2)
        accs.append(correct[mask].mean())
        confs.append(conf[mask].mean())
    return np.array(centers), np.array(accs), np.array(confs)


# -----------------------------------------------------------------------------
# Label-noise sensitivity
# -----------------------------------------------------------------------------

def label_noise_sensitivity(model, X: pd.DataFrame, y: np.ndarray,
                            composers: np.ndarray, n_bootstrap: int = 50,
                            perturb_sd: float = 0.5,
                            random_state: int = 42) -> pd.DataFrame:
    """How much does label noise change the MAE?

    Henle annotations have inter-annotator disagreement of roughly 1 grade
    on borderline pieces. We simulate this by perturbing y with rounded
    Gaussian noise and refitting. If MAE moves more than the noise itself,
    the model is overfitting to specific judgments.
    """
    rng = np.random.default_rng(random_state)
    rows = []
    for b in range(n_bootstrap):
        noise = np.round(rng.normal(0, perturb_sd, size=len(y))).astype(int)
        y_noisy = np.clip(y + noise, 0, y.max())
        folds = stratified_cv(model, X, y_noisy, n_splits=5,
                              random_state=b)
        rows.append({'bootstrap': b, 'mae_mean': folds['mae'].mean(),
                     'qwk_mean': folds['qwk'].mean()})
    return pd.DataFrame(rows)
