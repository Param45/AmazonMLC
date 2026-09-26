"""From calibrated pair probabilities to one match list per Source-1 entity, tuned for macro F0.5.

Policies evaluated (each with and without "exclusive targets"):
  threshold          accept every candidate with p >= t
  source_threshold   separate thresholds for S2 and S3 targets
  expected_f         per entity, pick the top-k set maximising the *expected* F0.5 under the
                     calibrated probabilities (k = 0, i.e. "no match", is always an option). This
                     is the Bayes-optimal decision for a per-entity F-measure with independent
                     candidates, and it naturally protects singletons.
  exclusive          a target may be assigned to at most one Source-1 entity (the one with the
                     highest probability). Only sensible if the training truth rarely links one
                     target to several entities - the tuning decides on validation data.

Parameters are chosen by *cross-fitting*: for every fold the policy is tuned on the other folds'
out-of-fold predictions and scored on the held-out fold, so the reported number is honest.
Every Source-1 entity counts, including entities whose candidate list is empty.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

from .config import DecisionConfig
from .metrics import fbeta_from_counts


@dataclass
class Policy:
    kind: str = "threshold"
    exclusive: bool = False
    threshold: float = 0.5
    source_thresholds: Dict[str, float] = field(default_factory=dict)
    gamma: float = 1.0

    def describe(self) -> str:
        if self.kind == "threshold":
            core = f"threshold p>={self.threshold:.2f}"
        elif self.kind == "source_threshold":
            core = "per-source thresholds " + ", ".join(f"{k}>={v:.2f}" for k, v in sorted(self.source_thresholds.items()))
        else:
            core = f"expected-F0.5 top-k (gamma={self.gamma})"
        return core + (" + exclusive targets" if self.exclusive else "")


# ----------------------------------------------------------------------------- helpers
def target_best_mask(t_key: np.ndarray, p: np.ndarray) -> np.ndarray:
    """True for the single highest-probability pair of every target (ties broken by order)."""
    order = np.lexsort((-p, t_key))
    first = np.r_[True, t_key[order][1:] != t_key[order][:-1]]
    mask = np.zeros(len(p), dtype=bool)
    mask[order[first]] = True
    return mask


def _best_k(p: np.ndarray, b2: float) -> int:
    """argmax_k E[F_beta(top-k)] for independent Bernoulli(p), p sorted in decreasing order."""
    n = len(p)
    if n == 0:
        return 0
    pref = [np.ones(1)]
    for x in p:
        prev = pref[-1]
        nxt = np.zeros(len(prev) + 1)
        nxt[:-1] += prev * (1 - x)
        nxt[1:] += prev * x
        pref.append(nxt)
    suf = [None] * (n + 1)
    suf[n] = np.ones(1)
    for k in range(n - 1, -1, -1):
        prev = suf[k + 1]
        nxt = np.zeros(len(prev) + 1)
        nxt[:-1] += prev * (1 - p[k])
        nxt[1:] += prev * p[k]
        suf[k] = nxt
    best_k, best = 0, suf[0][0]          # k = 0 scores 1 only if nothing is a true match
    for k in range(1, n + 1):
        a = np.arange(k + 1)[:, None]
        b = np.arange(len(suf[k]))[None, :]
        val = float((pref[k][:, None] * suf[k][None, :] * ((1 + b2) * a / (b2 * (a + b) + k))).sum())
        if val > best:
            best, best_k = val, k
        elif k > best_k + 2:
            break
    return best_k


def expected_f_mask(s1_key: np.ndarray, p: np.ndarray, beta: float, min_prob: float) -> np.ndarray:
    b2 = beta * beta
    order = np.lexsort((-p, s1_key))
    sk, sp_ = s1_key[order], p[order]
    starts = np.r_[0, np.flatnonzero(sk[1:] != sk[:-1]) + 1] if len(sk) else np.array([], int)
    ends = np.r_[starts[1:], len(sk)]
    mask_sorted = np.zeros(len(p), dtype=bool)
    for s, e in zip(starts, ends):
        ps = sp_[s:e]
        m = int(np.searchsorted(-ps, -min_prob, side="right"))   # count of p >= min_prob (sorted desc)
        if m == 0:
            continue
        k = _best_k(ps[:m], b2)
        mask_sorted[s:s + k] = True
    mask = np.zeros(len(p), dtype=bool)
    mask[order] = mask_sorted
    return mask


def policy_mask(df: pd.DataFrame, policy: Policy, cfg: DecisionConfig,
                _cache: Optional[dict] = None) -> np.ndarray:
    """Boolean mask over the rows of df (columns s1_id, t_id, src, p) selected by the policy."""
    p = df["p"].to_numpy(dtype=np.float64)
    cache = _cache if _cache is not None else {}
    if "s1_key" not in cache:
        cache["s1_key"] = pd.factorize(df["s1_id"])[0]
        cache["t_key"] = pd.factorize(df["t_id"])[0]
        cache["best"] = target_best_mask(cache["t_key"], p)
        cache["src_code"], cache["src_names"] = pd.factorize(df["src"])
    allowed = cache["best"] if policy.exclusive else np.ones(len(p), dtype=bool)
    if policy.kind == "threshold":
        return allowed & (p >= policy.threshold)
    if policy.kind == "source_threshold":
        per_src = np.array([policy.source_thresholds.get(n, policy.threshold) for n in cache["src_names"]])
        return allowed & (p >= per_src[cache["src_code"]])
    if policy.kind == "expected_f":
        q = np.where(allowed, np.clip(p, 0, 1) ** policy.gamma, 0.0)
        return expected_f_mask(cache["s1_key"], q, cfg.beta, cfg.min_prob)
    raise ValueError(policy.kind)


def predictions_from_mask(df: pd.DataFrame, mask: np.ndarray) -> Dict[str, Set[str]]:
    kept = df.loc[mask, ["s1_id", "t_id"]]
    return kept.groupby("s1_id")["t_id"].apply(set).to_dict()


# ----------------------------------------------------------------------------- tuning
# lower = simpler; used to break near-ties in favour of the policy least likely to be overfit
_COMPLEXITY = {("threshold", False): 0, ("threshold", True): 1, ("expected_f", False): 2, ("expected_f", True): 3,
               ("source_threshold", False): 4, ("source_threshold", True): 5}


def candidate_policies(cfg: DecisionConfig, sources: List[str]) -> List[Policy]:
    out: List[Policy] = []
    for excl in ([False, True] if cfg.try_exclusive else [False]):
        out += [Policy("threshold", excl, threshold=t) for t in cfg.threshold_grid]
        if len(sources) > 1:
            grid = cfg.source_threshold_grid
            for t2 in grid:
                for t3 in grid:
                    out.append(Policy("source_threshold", excl, threshold=max(t2, t3),
                                      source_thresholds={sources[0]: t2, sources[1]: t3}))
        out += [Policy("expected_f", excl, gamma=g) for g in cfg.expected_f_gamma_grid]
    return out


def tune_policy(oof: pd.DataFrame, truth: Dict[str, Set[str]], s1_fold: Dict[str, int], cfg: DecisionConfig,
                verbose: bool = True):
    """Cross-fitted policy selection.

    oof: one row per candidate pair with columns s1_id, t_id, src, p (calibrated OOF probability), y.
    Returns (best_policy_on_all_folds, summary_table, crossfit_macro_f).
    """
    s1_ids = list(truth)
    code = {s: i for i, s in enumerate(s1_ids)}
    n_s1 = len(s1_ids)
    n_true = np.array([len(truth[s]) for s in s1_ids], dtype=np.float64)
    fold = np.array([s1_fold[s] for s in s1_ids])
    n_folds = fold.max() + 1
    fold_sizes = np.bincount(fold, minlength=n_folds).astype(np.float64)
    pair_code = oof["s1_id"].map(code).to_numpy()
    y = oof["y"].to_numpy().astype(bool)
    sources = sorted(oof["src"].unique())
    policies = candidate_policies(cfg, sources)
    cache: dict = {}
    fold_sums = np.zeros((len(policies), n_folds))
    for i, pol in enumerate(policies):
        mask = policy_mask(oof, pol, cfg, cache)
        n_pred = np.bincount(pair_code[mask], minlength=n_s1)
        tp = np.bincount(pair_code[mask & y], minlength=n_s1)
        f = fbeta_from_counts(tp, n_pred, n_true, cfg.beta)
        fold_sums[i] = np.bincount(fold, weights=f, minlength=n_folds)

    total = fold_sums.sum(axis=1) / n_s1
    complexity = np.array([_COMPLEXITY[(p.kind, p.exclusive)] for p in policies])

    def select(scores: np.ndarray) -> int:
        """Simplest policy whose score is within `tolerance` of the best (guards against grid overfitting)."""
        ok = np.flatnonzero(scores >= scores.max() - cfg.tolerance)
        ok = ok[complexity[ok] == complexity[ok].min()]
        return int(ok[np.argmax(scores[ok])])

    crossfit = 0.0
    chosen = []
    for f in range(n_folds):
        other = (fold_sums.sum(axis=1) - fold_sums[:, f]) / (n_s1 - fold_sizes[f])
        j = select(other)
        chosen.append(policies[j].describe())
        crossfit += fold_sums[j, f]
    crossfit /= n_s1
    best = policies[select(total)]

    table = pd.DataFrame({"policy": [p.describe() for p in policies], "family": [p.kind for p in policies],
                          "exclusive": [p.exclusive for p in policies], "macro_F0.5 (OOF)": total})
    summary = (table.sort_values("macro_F0.5 (OOF)", ascending=False)
               .groupby(["family", "exclusive"], as_index=False).first()
               .sort_values("macro_F0.5 (OOF)", ascending=False).reset_index(drop=True))
    summary.attrs["full"] = table
    if verbose:
        print(f"[decision] selected on all OOF: {best.describe()}  macro F0.5={total[policies.index(best)]:.5f} "
              f"(best of any policy {total.max():.5f}; ties within {cfg.tolerance} go to the simpler policy)")
        print(f"[decision] cross-fitted macro F0.5 (honest): {crossfit:.5f}; per-fold picks: {chosen}")
    return best, summary, crossfit
