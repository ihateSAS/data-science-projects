"""
Three CV protocols for composer classification, each with different leakage
control.

    1. random_cv          : standard 5-fold, stratified by composer
    2. leave_one_work_out : same composer in train and test, but no
                            shared opus number. Tests robustness to
                            within-composer variation.
    3. leave_one_era_out  : hardest — train on Baroque + Classical + Romantic,
                            predict on Modern, etc. Tests whether features
                            generalize across stylistic periods.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .models import composer_metrics, expected_calibration_error


def random_cv(model, X, y, n_splits: int = 5, random_state: int = 42,
              era_map: dict = None) -> pd.DataFrame:
    """Standard random stratified k-fold."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows = []
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        m = deepcopy(model).fit(X.iloc[tr], y[tr])
        y_pred = m.predict(X.iloc[te])
        probs  = m.predict_proba(X.iloc[te]) if hasattr(m, 'predict_proba') else None
        metrics = composer_metrics(y[te], y_pred, probs=probs,
                                   classes=getattr(m, 'classes_', None),
                                   era_map=era_map)
        metrics['fold'] = fold
        rows.append(metrics)
    return pd.DataFrame(rows)


def leave_one_work_out(model, X, y, work_ids, era_map: dict = None,
                       max_works: int = 200) -> pd.DataFrame:
    """For each unique opus/work, hold all performances of it out.

    work_ids should be e.g. composer + opus number, so that different
    performances of the same piece never appear in both train and test.
    """
    works = pd.Series(work_ids).unique()
    if len(works) > max_works:
        rng = np.random.default_rng(42)
        works = rng.choice(works, max_works, replace=False)

    rows = []
    work_arr = np.asarray(work_ids)
    for work in works:
        te_mask = work_arr == work
        if te_mask.sum() == 0 or te_mask.sum() == len(work_arr):
            continue
        m = deepcopy(model).fit(X[~te_mask], y[~te_mask])
        y_pred = m.predict(X[te_mask])
        probs  = m.predict_proba(X[te_mask]) if hasattr(m, 'predict_proba') else None
        metrics = composer_metrics(y[te_mask], y_pred, probs=probs,
                                   classes=getattr(m, 'classes_', None),
                                   era_map=era_map)
        metrics.update({'heldout_work': work, 'n_test': int(te_mask.sum())})
        rows.append(metrics)
    return pd.DataFrame(rows)


def leave_one_era_out(model, X, y, eras, era_map: dict = None) -> pd.DataFrame:
    """For each era, train on all other eras and test on the held-out one.

    Restricted to pieces whose composer's era is in the era list.
    This is the hardest test: features must transfer across stylistic periods.
    """
    eras = np.asarray(eras)
    unique_eras = [e for e in np.unique(eras) if e != 'Unknown']

    rows = []
    for era in unique_eras:
        te_mask = eras == era
        if te_mask.sum() == 0 or te_mask.sum() == len(eras):
            continue
        m = deepcopy(model).fit(X[~te_mask], y[~te_mask])
        y_pred = m.predict(X[te_mask])
        # For era-disjoint: composer-level top-1 may be ~0 (composers don't
        # appear in train). Report era-level metric.
        try:
            era_pred = np.array([era_map.get(c, 'Unknown') for c in y_pred])
            era_true = np.full(te_mask.sum(), era)
            era_acc  = float(np.mean(era_pred == era_true))
        except Exception:
            era_acc  = np.nan
        rows.append({'heldout_era': era, 'n_test': int(te_mask.sum()),
                     'composer_acc': float((y_pred == y[te_mask]).mean()),
                     'era_acc': era_acc})
    return pd.DataFrame(rows)


def reliability_curve(probs, y_true_idx, n_bins=15):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true_idx).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    centers, accs, confs = [], [], []
    for i in range(n_bins):
        m = (conf >= bins[i]) & (conf < bins[i + 1])
        if m.sum() == 0:
            continue
        centers.append((bins[i] + bins[i + 1]) / 2)
        accs.append(correct[m].mean())
        confs.append(conf[m].mean())
    return np.array(centers), np.array(accs), np.array(confs)
