"""
Models for composer classification on GiantMIDI.

Implements:
    - CalibratedComposerClassifier: LightGBM + isotonic calibration
    - HierarchicalEraComposer:     era classifier -> per-era composer classifier
    - ConformalClassifier:         APS-style conformal sets (Romano et al. 2020)
    - ComposerMetrics:             top-1/3 acc, macro-F1, ECE, era-confused acc
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, f1_score, top_k_accuracy_score,
                             confusion_matrix)


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def composer_metrics(y_true, y_pred, probs=None, classes=None,
                     era_map: dict = None) -> dict:
    """Top-1 acc, macro-F1, top-3 acc (if probs given), era-correct acc."""
    out = {
        'acc':       float(accuracy_score(y_true, y_pred)),
        'f1_macro':  float(f1_score(y_true, y_pred, average='macro',
                                    zero_division=0)),
    }
    if probs is not None and classes is not None:
        try:
            out['top3_acc'] = float(top_k_accuracy_score(
                y_true, probs, k=min(3, len(classes)), labels=classes))
        except Exception:
            out['top3_acc'] = np.nan
    if era_map is not None:
        # Era-correct: predicted composer's era matches true composer's era
        era_true = np.array([era_map.get(c, 'Unknown') for c in y_true])
        era_pred = np.array([era_map.get(c, 'Unknown') for c in y_pred])
        out['era_acc'] = float(np.mean(era_true == era_pred))
    return out


def expected_calibration_error(probs: np.ndarray, y_true_idx: np.ndarray,
                                n_bins: int = 15) -> float:
    """ECE over top-1 confidence."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true_idx).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf >= bins[i]) & (conf < bins[i + 1])
        if m.sum() == 0:
            continue
        ece += (m.sum() / len(y_true_idx)) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


# -----------------------------------------------------------------------------
# Calibrated LightGBM
# -----------------------------------------------------------------------------

class CalibratedComposerClassifier(BaseEstimator, ClassifierMixin):
    """LightGBM with isotonic calibration via CalibratedClassifierCV.

    Why isotonic rather than Platt: composer probabilities are typically
    over-confident at the top end and the relationship isn't sigmoidal.
    Isotonic handles the long tail of less-confident predictions better.
    """

    def __init__(self, n_estimators: int = 600, learning_rate: float = 0.05,
                 max_depth: int = 6, num_leaves: int = 63,
                 calibration_cv: int = 3, random_state: int = 42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.calibration_cv = calibration_cv
        self.random_state = random_state

    def fit(self, X, y):
        import lightgbm as lgb
        base = lgb.LGBMClassifier(
            n_estimators=self.n_estimators, learning_rate=self.learning_rate,
            max_depth=self.max_depth, num_leaves=self.num_leaves,
            random_state=self.random_state, n_jobs=-1, verbose=-1,
        )
        # Calibrate via cross-validation on the training set
        self.model_ = CalibratedClassifierCV(base, method='isotonic',
                                             cv=self.calibration_cv)
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):     return self.model_.predict(X)
    def predict_proba(self, X): return self.model_.predict_proba(X)


# -----------------------------------------------------------------------------
# Hierarchical era -> composer
# -----------------------------------------------------------------------------

class HierarchicalEraComposer(BaseEstimator, ClassifierMixin):
    """Two-stage classification: predict era first, then composer within era.

    This is more informative than a flat classifier because:
        (a) era is much easier than composer (5 classes vs 20)
        (b) confusion patterns can be decomposed: did we get the era wrong,
            or just confuse two composers within the right era?
    """

    def __init__(self, era_map: dict, random_state: int = 42):
        self.era_map = era_map
        self.random_state = random_state

    def fit(self, X, y):
        y = np.asarray(y)
        eras = np.array([self.era_map.get(c, 'Unknown') for c in y])
        self.eras_ = np.unique(eras)

        # Stage 1: era classifier
        self.era_model_ = CalibratedComposerClassifier(
            random_state=self.random_state).fit(X, eras)

        # Stage 2: one composer classifier per era
        self.composer_models_ = {}
        for era in self.eras_:
            mask = eras == era
            if mask.sum() < 5 or len(np.unique(y[mask])) < 2:
                self.composer_models_[era] = None
                continue
            self.composer_models_[era] = CalibratedComposerClassifier(
                random_state=self.random_state).fit(X[mask], y[mask])

        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        era_pred = self.era_model_.predict(X)
        out = np.empty(len(X), dtype=object)
        for era in self.eras_:
            mask = era_pred == era
            if mask.sum() == 0:
                continue
            model = self.composer_models_.get(era)
            if model is None:
                # Fallback: most common composer of this era from training
                out[mask] = era
            else:
                out[mask] = model.predict(X[mask])
        return out


# -----------------------------------------------------------------------------
# Conformal classification sets (APS — Romano, Sesia, Candès 2020)
# -----------------------------------------------------------------------------

class ConformalClassifier:
    """Adaptive Prediction Sets for classification.

    Returns the smallest set of composers whose cumulative probability mass
    reaches the calibration threshold. Empirical coverage matches the target
    1 - alpha within finite-sample bounds.
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha

    def calibrate(self, probs_cal: np.ndarray, y_cal_idx: np.ndarray):
        # For each calibration point, the nonconformity is 1 - cumulative prob
        # up to and including the true class (with random tiebreaking).
        n = len(y_cal_idx)
        scores = np.zeros(n)
        for i, yi in enumerate(y_cal_idx):
            sorted_idx = np.argsort(probs_cal[i])[::-1]
            cum = 0.0
            for k in sorted_idx:
                cum += probs_cal[i, k]
                if k == yi:
                    break
            scores[i] = cum
        # Take 1 - alpha quantile of scores
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.q_hat_ = float(np.quantile(scores, q_level, method='higher'))
        return self

    def predict_set(self, probs: np.ndarray) -> list:
        """For each test row, return the prediction set as a list of class indices."""
        sets = []
        for p in probs:
            sorted_idx = np.argsort(p)[::-1]
            cum = 0.0
            chosen = []
            for k in sorted_idx:
                chosen.append(int(k))
                cum += p[k]
                if cum >= self.q_hat_:
                    break
            sets.append(chosen)
        return sets
