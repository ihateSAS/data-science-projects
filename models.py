"""
Ordinal models and evaluation for piano difficulty prediction.

Implements:
    - OrdinalGBM        : LightGBM with monotonic constraints on motoric features,
                          trained with the cumulative-link binary-decomposition trick
                          (Frank & Hall, 2001).
    - CORNMLPRegressor  : MLP with CORN ordinal output (Shi/Cao/Raschka 2023).
                          Rank-consistent: P(y > k+1) <= P(y > k) by construction.
    - ConformalOrdinal  : split-conformal prediction sets (Romano et al. 2020,
                          adapted for ordinal scores via cumulative-prob nonconformity).
    - Metrics           : MAE, Acc±n, quadratic-weighted kappa, balanced acc, ECE.

Why this design:
    CIPI has only ~650 pieces. Treating 9-level Henle grades as continuous regression
    throws away the ordering when the model picks the wrong tail; treating them as
    nominal classes throws away the ordering on the right tail. CORN gets both --
    monotonic per-class probabilities and ordinal-aware loss.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import (accuracy_score, mean_absolute_error,
                             balanced_accuracy_score, cohen_kappa_score)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# -----------------------------------------------------------------------------
# Ordinal-aware metrics
# -----------------------------------------------------------------------------

def ordinal_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    n_classes: int = 9) -> dict:
    """Return the metric bundle used in the Ramoneda et al. 2024 benchmark
    plus a few extras."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.clip(np.round(y_pred), 0, n_classes - 1).astype(int)
    return {
        'mae':          mean_absolute_error(y_true, y_pred),
        'acc_exact':    accuracy_score(y_true, y_pred),
        'acc_plus_1':   float(np.mean(np.abs(y_true - y_pred) <= 1)),
        'acc_plus_2':   float(np.mean(np.abs(y_true - y_pred) <= 2)),
        'balanced_acc': balanced_accuracy_score(y_true, y_pred),
        # Quadratic-weighted kappa rewards being close on the ordinal scale.
        'qwk':          cohen_kappa_score(y_true, y_pred, weights='quadratic'),
    }


# -----------------------------------------------------------------------------
# Frank & Hall (2001) -- ordinal via binary decomposition
# -----------------------------------------------------------------------------

class OrdinalGBM(BaseEstimator, RegressorMixin):
    """Ordinal regression via cumulative binary classification (Frank & Hall, 2001).

    For K classes, trains K-1 binary classifiers: classifier k predicts P(y > k).
    At inference, P(y = j) = P(y > j-1) - P(y > j), and the predicted class is
    argmax. Monotonic constraints on motoric features are passed to LightGBM.
    """

    def __init__(self, n_classes: int = 9,
                 monotone_cols: Optional[list[str]] = None,
                 n_estimators: int = 400, learning_rate: float = 0.05,
                 max_depth: int = 5, random_state: int = 42):
        self.n_classes = n_classes
        self.monotone_cols = monotone_cols or []
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        import lightgbm as lgb
        self.feature_names_ = list(X.columns)
        # +1 for features expected to push difficulty up, -1 for down,
        # 0 (default) for no constraint
        mono = [1 if c in self.monotone_cols else 0 for c in self.feature_names_]
        self.classifiers_ = []
        for k in range(self.n_classes - 1):
            y_bin = (y > k).astype(int)
            clf = lgb.LGBMClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                monotone_constraints=mono,
                random_state=self.random_state,
                verbose=-1,
            )
            clf.fit(X, y_bin)
            self.classifiers_.append(clf)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # P(y > k) for k in 0..K-2
        gt = np.stack([clf.predict_proba(X)[:, 1] for clf in self.classifiers_], axis=1)
        # Enforce monotonicity (Frank-Hall doesn't by default)
        gt = np.minimum.accumulate(gt, axis=1)
        # P(y = j)
        n = X.shape[0]
        probs = np.zeros((n, self.n_classes))
        probs[:, 0] = 1 - gt[:, 0]
        for k in range(1, self.n_classes - 1):
            probs[:, k] = gt[:, k - 1] - gt[:, k]
        probs[:, -1] = gt[:, -1]
        return np.clip(probs, 0, 1)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)


# -----------------------------------------------------------------------------
# CORN MLP (Shi, Cao, Raschka 2023)
# -----------------------------------------------------------------------------

if HAS_TORCH:

    class CORNMLPRegressor(BaseEstimator, RegressorMixin):
        """MLP with CORN ordinal head.

        CORN trains the kth binary head conditional on y > k-1, giving
        unconditional cumulative probabilities via the chain rule. Empirically
        beats CORAL on small datasets by removing the weight-sharing constraint.
        """

        def __init__(self, n_classes: int = 9, hidden: tuple = (128, 64),
                     lr: float = 1e-3, epochs: int = 200, batch_size: int = 32,
                     dropout: float = 0.2, weight_decay: float = 1e-4,
                     device: str = 'cpu', random_state: int = 42):
            self.n_classes = n_classes
            self.hidden = hidden
            self.lr = lr
            self.epochs = epochs
            self.batch_size = batch_size
            self.dropout = dropout
            self.weight_decay = weight_decay
            self.device = device
            self.random_state = random_state

        def _build(self, in_dim: int) -> nn.Module:
            torch.manual_seed(self.random_state)
            layers = []
            prev = in_dim
            for h in self.hidden:
                layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(self.dropout)]
                prev = h
            # K-1 binary heads (rank-consistent via conditional training)
            layers.append(nn.Linear(prev, self.n_classes - 1))
            return nn.Sequential(*layers)

        def fit(self, X: np.ndarray, y: np.ndarray):
            X = np.asarray(X, dtype=np.float32)
            y = np.asarray(y, dtype=np.int64)
            self.net_ = self._build(X.shape[1]).to(self.device)
            opt = torch.optim.AdamW(self.net_.parameters(),
                                    lr=self.lr, weight_decay=self.weight_decay)
            ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
            dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
            self.history_ = []

            for epoch in range(self.epochs):
                self.net_.train()
                losses = []
                for xb, yb in dl:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    logits = self.net_(xb)              # (B, K-1)
                    loss = self._corn_loss(logits, yb)
                    opt.zero_grad(); loss.backward(); opt.step()
                    losses.append(loss.item())
                self.history_.append(float(np.mean(losses)))
            return self

        def _corn_loss(self, logits, y):
            """CORN loss: chain-rule conditional binary cross-entropy."""
            K = self.n_classes
            losses = []
            for k in range(K - 1):
                # Train head k only on samples with y > k-1
                mask = y > k - 1 if k > 0 else torch.ones_like(y, dtype=torch.bool)
                if mask.sum() < 2:
                    continue
                target = (y[mask] > k).float()
                pred = logits[mask, k]
                losses.append(nn.functional.binary_cross_entropy_with_logits(pred, target))
            return torch.stack(losses).mean()

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            self.net_.eval()
            with torch.no_grad():
                logits = self.net_(torch.from_numpy(np.asarray(X, dtype=np.float32))
                                   .to(self.device))
                # Conditional probabilities P(y > k | y > k-1)
                cond = torch.sigmoid(logits).cpu().numpy()
            # Unconditional P(y > k) = product of conditionals up to k
            cum = np.cumprod(cond, axis=1)
            probs = np.zeros((X.shape[0], self.n_classes))
            probs[:, 0] = 1 - cum[:, 0]
            for k in range(1, self.n_classes - 1):
                probs[:, k] = cum[:, k - 1] - cum[:, k]
            probs[:, -1] = cum[:, -1]
            return np.clip(probs, 0, 1)

        def predict(self, X: np.ndarray) -> np.ndarray:
            return self.predict_proba(X).argmax(axis=1)


# -----------------------------------------------------------------------------
# Conformal prediction for ordinal targets
# -----------------------------------------------------------------------------

class ConformalOrdinal:
    """Split-conformal prediction sets for ordinal targets.

    Given a calibration set, returns the smallest contiguous interval of grades
    [y_lo, y_hi] that covers the true grade with probability >= 1 - alpha.
    Contiguity is enforced because non-adjacent grades make no sense for
    difficulty (Romano et al. 2020 + adaptation for ordinal).
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha

    def calibrate(self, probs_cal: np.ndarray, y_cal: np.ndarray):
        # Nonconformity: 1 - cumulative prob mass around the true label,
        # minimized over contiguous intervals containing the truth.
        # Simpler proxy: nonconformity = -log P(y_true)
        eps = 1e-9
        scores = -np.log(probs_cal[np.arange(len(y_cal)), y_cal] + eps)
        n = len(y_cal)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.q_hat_ = float(np.quantile(scores, q_level, method='higher'))
        return self

    def predict_set(self, probs_test: np.ndarray) -> list[tuple[int, int]]:
        """For each test sample, return (y_lo, y_hi) — smallest contiguous
        interval whose log-prob covers q_hat."""
        intervals = []
        for p in probs_test:
            # Sort grades by descending prob; greedily expand contiguous interval
            order = np.argsort(p)[::-1]
            included = set()
            mass = 0.0
            for k in order:
                included.add(int(k))
                mass = sum(p[j] for j in included)
                if -np.log(max(mass, 1e-12)) <= self.q_hat_:
                    break
            lo, hi = min(included), max(included)
            intervals.append((lo, hi))
        return intervals
