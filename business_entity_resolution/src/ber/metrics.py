"""Macro F0.5 (the leaderboard metric) and diagnostic breakdowns.

Metric, exactly as specified: F_beta is computed per Source-1 entity and then averaged over *all*
Source-1 entities. A singleton scores 1.0 for an empty prediction and 0.0 for any prediction.
An entity with true matches and an empty prediction scores 0.0.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Set

import numpy as np
import pandas as pd

from .blocking import BLOCKERS


def fbeta_from_counts(tp, n_pred, n_true, beta: float = 0.5) -> np.ndarray:
    """Vectorised per-entity F_beta. F = (1+b^2) TP / (b^2 * n_true + n_pred); empty/empty -> 1."""
    tp, n_pred, n_true = (np.asarray(x, dtype=np.float64) for x in (tp, n_pred, n_true))
    b2 = beta * beta
    denom = b2 * n_true + n_pred
    f = np.divide((1 + b2) * tp, denom, out=np.zeros_like(denom), where=denom > 0)
    return np.where((n_true == 0) & (n_pred == 0), 1.0, f)


def per_entity_scores(pred: Dict[str, Set[str]], truth: Dict[str, Set[str]],
                      s1_ids: Optional[Iterable[str]] = None, beta: float = 0.5) -> pd.DataFrame:
    ids = list(s1_ids) if s1_ids is not None else list(truth)
    tp, n_pred, n_true = [], [], []
    for s in ids:
        p, t = pred.get(s, set()), truth.get(s, set())
        tp.append(len(p & t))
        n_pred.append(len(p))
        n_true.append(len(t))
    df = pd.DataFrame({"s1_id": ids, "tp": tp, "n_pred": n_pred, "n_true": n_true})
    df["f"] = fbeta_from_counts(df["tp"], df["n_pred"], df["n_true"], beta)
    df["precision"] = np.where(df["n_pred"] > 0, df["tp"] / df["n_pred"].clip(lower=1), np.nan)
    df["recall"] = np.where(df["n_true"] > 0, df["tp"] / df["n_true"].clip(lower=1), np.nan)
    return df


def macro_fbeta(pred, truth, s1_ids=None, beta: float = 0.5) -> float:
    return float(per_entity_scores(pred, truth, s1_ids, beta)["f"].mean())


def _bucket(n: int) -> str:
    return "3+" if n >= 3 else str(n)


def _restrict(d: Dict[str, Set[str]], prefix: str) -> Dict[str, Set[str]]:
    return {k: {x for x in v if x.startswith(prefix)} for k, v in d.items()}


def evaluation_report(pred: Dict[str, Set[str]], truth: Dict[str, Set[str]], s1_ids: Iterable[str],
                      country: Optional[Dict[str, str]] = None, beta: float = 0.5) -> Dict[str, object]:
    """Headline numbers + breakdown tables for one set of predictions."""
    ids = list(s1_ids)
    df = per_entity_scores(pred, truth, ids, beta)
    single = df["n_true"] == 0
    tp, n_pred, n_true = df["tp"].sum(), df["n_pred"].sum(), df["n_true"].sum()
    headline = {
        "macro_F0.5": df["f"].mean(),
        "macro_precision (entities with predictions)": df["precision"].mean(),
        "macro_recall (entities with true matches)": df["recall"].mean(),
        "pair_precision": tp / max(n_pred, 1),
        "pair_recall": tp / max(n_true, 1),
        "singleton_share": single.mean(),
        "singleton_accuracy (empty predicted)": (df.loc[single, "n_pred"] == 0).mean() if single.any() else np.nan,
        "F0.5 on non-singletons": df.loc[~single, "f"].mean() if (~single).any() else np.nan,
        "non-singletons predicted empty": (df.loc[~single, "n_pred"] == 0).mean() if (~single).any() else np.nan,
    }
    df["bucket"] = df["n_true"].map(_bucket)
    by_card = df.groupby("bucket").agg(entities=("f", "size"), macro_F=("f", "mean"),
                                       precision=("precision", "mean"), recall=("recall", "mean"))
    tables = {"by_cardinality": by_card}
    if country is not None:
        df["country"] = df["s1_id"].map(country).fillna("")
        tables["by_country"] = df.groupby("country").agg(
            entities=("f", "size"), macro_F=("f", "mean"), singleton_share=("n_true", lambda x: (x == 0).mean()),
            precision=("precision", "mean"), recall=("recall", "mean"))
    rows = []
    for prefix in ("S2-", "S3-"):
        sub = per_entity_scores(_restrict(pred, prefix), _restrict(truth, prefix), ids, beta)
        rows.append({"target_source": prefix[:2], "macro_F (this source only)": sub["f"].mean(),
                     "pair_precision": sub["tp"].sum() / max(sub["n_pred"].sum(), 1),
                     "pair_recall": sub["tp"].sum() / max(sub["n_true"].sum(), 1)})
    tables["by_source"] = pd.DataFrame(rows).set_index("target_source")
    return {"headline": pd.Series(headline, name="value"), "tables": tables, "per_entity": df}


def blocking_report(cand: pd.DataFrame, truth: Dict[str, Set[str]], n_s1: int, n_t: int,
                    country: Optional[Dict[str, str]] = None, beta: float = 0.5) -> Dict[str, object]:
    """Recall ceiling and size of a candidate set (cand needs columns s1_id, t_id, src, b_* flags)."""
    pos = pd.DataFrame([(s, t) for s, ts in truth.items() for t in ts], columns=["s1_id", "t_id"])
    pos["src"] = pos["t_id"].str[:2]
    m = cand.merge(pos, on=["s1_id", "t_id", "src"], how="right", indicator=True)
    m["found"] = (m["_merge"] == "both").values
    per_s1 = cand.groupby("s1_id").size()
    all_ids = list(truth)
    counts = per_s1.reindex(all_ids).fillna(0)
    oracle = {}
    cand_by = cand.groupby("s1_id")["t_id"].apply(set).to_dict()
    for s, ts in truth.items():
        oracle[s] = ts & cand_by.get(s, set())
    upper = per_entity_scores(oracle, truth, all_ids, beta)["f"].mean()
    s1_all = m.groupby("s1_id")["found"].all()
    headline = pd.Series({
        "candidate pairs": len(cand),
        "candidates per S1 (mean)": counts.mean(),
        "candidates per S1 (p95)": counts.quantile(0.95),
        "candidates per S1 (max)": counts.max(),
        "S1 with no candidates": int((counts == 0).sum()),
        "reduction ratio vs full cross join": 1 - len(cand) / max(n_s1 * n_t, 1),
        "pair recall (true pairs retrieved)": m["found"].mean() if len(m) else np.nan,
        "S1 with all true matches retrieved": s1_all.mean() if len(s1_all) else np.nan,
        "macro F0.5 upper bound (oracle matcher)": upper,
    }, name="value")
    tables = {"recall_by_source": m.groupby("src")["found"].agg(true_pairs="size", recall="mean")}
    if country is not None:
        m["country"] = m["s1_id"].map(country).fillna("")
        tables["recall_by_country"] = m.groupby("country")["found"].agg(true_pairs="size", recall="mean")
    # per-blocker contribution on the true pairs that survived pruning
    hit = m[m["found"]]
    rows, cum = [], np.zeros(len(hit), dtype=bool)
    flags = [b for b in BLOCKERS if b in hit.columns]
    for b in flags:
        alone = hit[b].fillna(0).astype(bool).values
        others = np.zeros(len(hit), dtype=bool)
        for o in flags:
            if o != b:
                others |= hit[o].fillna(0).astype(bool).values
        cum |= alone
        rows.append({"blocker": b,
                     "recall alone": alone.sum() / max(len(m), 1),
                     "cumulative recall": cum.sum() / max(len(m), 1),
                     "true pairs found only by this blocker": int((alone & ~others).sum()),
                     "candidate pairs flagged": int(cand[b].sum()) if b in cand else 0})
    tables["by_blocker"] = pd.DataFrame(rows).set_index("blocker")
    return {"headline": headline, "tables": tables, "missed": m.loc[~m["found"], ["s1_id", "t_id", "src"]]}
