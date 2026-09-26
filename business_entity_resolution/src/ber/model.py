"""LightGBM pair matcher (MIT licence) trained with grouped, stratified cross-validation.

* Folds are built over Source-1 *entities* (never over pairs), grouped by connected components of
  the ground-truth graph, so no entity - and no target shared by two entities - leaks across folds.
* Folds are stratified by country x match-cardinality so every fold has singletons and multi-matches.
* The model is trained on the full candidate set produced by blocking, i.e. on exactly the pair
  distribution it will see at inference time. No negative subsampling, so probabilities stay
  calibratable.
* Test-time scores are the mean of the fold models; an isotonic calibrator is fit on the
  out-of-fold scores (and cross-fitted for honest validation numbers).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from .config import ModelConfig


def entity_groups(truth: Dict[str, Set[str]]) -> Dict[str, int]:
    """Connected components of the S1-target graph (S1 entities sharing any target end up together)."""
    parent: Dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for s, ts in truth.items():
        find(s)
        for t in ts:
            ra, rb = find(s), find(t)
            if ra != rb:
                parent[rb] = ra
    roots: Dict[str, int] = {}
    return {s: roots.setdefault(find(s), len(roots)) for s in truth}


def make_folds(truth: Dict[str, Set[str]], country: Dict[str, str], n_folds: int, seed: int) -> Dict[str, int]:
    ids = list(truth)
    groups = entity_groups(truth)
    card = [min(len(truth[s]), 3) for s in ids]
    strat = [f"{country.get(s, '')}|{c}" for s, c in zip(ids, card)]
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold = np.full(len(ids), -1)
    for f, (_, va) in enumerate(splitter.split(np.zeros(len(ids)), strat, [groups[s] for s in ids])):
        fold[va] = f
    return dict(zip(ids, fold.tolist()))


@dataclass
class Matcher:
    feature_names: List[str]
    models: List[lgb.Booster] = field(default_factory=list)
    calibrator: Optional[IsotonicRegression] = None
    importance: Optional[pd.DataFrame] = None
    cv_info: Dict[str, float] = field(default_factory=dict)

    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        X = X[self.feature_names]
        return np.mean([m.predict(X, num_iteration=m.best_iteration or None) for m in self.models], axis=0)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.predict_raw(X)
        return self.calibrator.predict(raw) if self.calibrator is not None else raw


def _fit_isotonic(score: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True)
    return iso.fit(score, y)


def train_matcher(X: pd.DataFrame, y: np.ndarray, pair_fold: np.ndarray, cfg: ModelConfig,
                  verbose: bool = True):
    """Grouped CV. Returns (Matcher, oof_raw, oof_calibrated_crossfit)."""
    params = dict(cfg.params)
    params["seed"] = cfg.seed
    n_folds = int(pair_fold.max()) + 1
    oof = np.zeros(len(y), dtype=np.float64)
    models, imps = [], []
    t0 = time.time()
    for f in range(n_folds):
        tr, va = pair_fold != f, pair_fold == f
        if verbose:
            print(f"[model] Fold {f+1}/{n_folds}: Fitting LightGBM on {tr.sum():,} train pairs (eval on {va.sum():,} val pairs)...", flush=True)
        dtr = lgb.Dataset(X[tr], label=y[tr], free_raw_data=True)
        dva = lgb.Dataset(X[va], label=y[va], reference=dtr, free_raw_data=True)
        callbacks = [
            lgb.early_stopping(cfg.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=100) if verbose else lgb.log_evaluation(period=0)
        ]
        booster = lgb.train(params, dtr, num_boost_round=cfg.num_boost_round, valid_sets=[dva],
                            callbacks=callbacks)
        oof[va] = booster.predict(X[va], num_iteration=booster.best_iteration)
        models.append(booster)
        imps.append(pd.Series(booster.feature_importance("gain"), index=X.columns))
        if verbose:
            print(f"[model] fold {f+1}/{n_folds} completed: best_iter={booster.best_iteration}, "
                  f"AP={average_precision_score(y[va], oof[va]):.4f} ({time.time() - t0:.0f}s)", flush=True)
    matcher = Matcher(feature_names=list(X.columns), models=models)
    imp = pd.concat(imps, axis=1).mean(axis=1).sort_values(ascending=False)
    matcher.importance = (imp / imp.sum()).rename("gain_share").to_frame()

    # cross-fitted calibration for honest out-of-fold probabilities, then a final calibrator on all OOF
    oof_cal = oof.copy()
    if cfg.calibrate:
        for f in range(n_folds):
            tr, va = pair_fold != f, pair_fold == f
            oof_cal[va] = _fit_isotonic(oof[tr], y[tr]).predict(oof[va])
        matcher.calibrator = _fit_isotonic(oof, y)
    matcher.cv_info = {"oof_auc": roc_auc_score(y, oof), "oof_average_precision": average_precision_score(y, oof),
                       "mean_best_iteration": float(np.mean([m.best_iteration for m in models]))}
    if verbose:
        print(f"[model] OOF AUC={matcher.cv_info['oof_auc']:.5f}  AP={matcher.cv_info['oof_average_precision']:.5f}")
    return matcher, oof, oof_cal
