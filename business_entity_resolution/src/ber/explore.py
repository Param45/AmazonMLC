"""Quantitative exploration of the training data (drives blocking and decision choices)."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from .blocking import rowwise_dot
from .data import SplitData
from .index import SplitIndex
from .normalize import is_missing, normalize_country


def _country_label(x: str) -> str:
    return x if not is_missing(x) else "<missing>"


def overview(data: SplitData) -> pd.DataFrame:
    rows = []
    for src, df in (("S1", data.s1), ("S2", data.s2), ("S3", data.s3)):
        rows.append({"source": src, "records": len(df),
                     "countries": ", ".join(f"{k}={v}" for k, v in Counter(map(_country_label, df["country"])).most_common())})
    return pd.DataFrame(rows).set_index("source")


def country_table(data: SplitData) -> pd.DataFrame:
    frames = [pd.DataFrame({"source": s, "country": df["country"].map(_country_label)})
              for s, df in (("S1", data.s1), ("S2", data.s2), ("S3", data.s3))]
    return pd.concat(frames).groupby(["country", "source"]).size().unstack(fill_value=0)


def cardinality(data: SplitData) -> Dict[str, pd.DataFrame]:
    truth = data.truth
    df = pd.DataFrame({"s1_id": list(truth),
                       "n_total": [len(v) for v in truth.values()],
                       "n_s2": [sum(x.startswith("S2-") for x in v) for v in truth.values()],
                       "n_s3": [sum(x.startswith("S3-") for x in v) for v in truth.values()]})
    df["country"] = df["s1_id"].map(dict(zip(data.s1["entity_id"], data.s1["country"].map(_country_label))))
    bucket = df["n_total"].clip(upper=3).map({0: "0 (singleton)", 1: "1", 2: "2", 3: "3+"})
    dist = pd.concat({"all": bucket.value_counts(normalize=True)}, axis=1)
    for c, sub in df.groupby("country"):
        dist[c] = bucket[sub.index].value_counts(normalize=True)
    per_source = pd.DataFrame({
        "S2": df["n_s2"].clip(upper=3).value_counts(normalize=True).sort_index(),
        "S3": df["n_s3"].clip(upper=3).value_counts(normalize=True).sort_index()}).fillna(0)
    per_source.index = [f"{i}{'+' if i == 3 else ''} matches" for i in per_source.index]
    stats = pd.Series({
        "mean matches per S1": df["n_total"].mean(),
        "max matches per S1": df["n_total"].max(),
        "max S2 matches per S1": df["n_s2"].max(),
        "max S3 matches per S1": df["n_s3"].max(),
        "S1 with both S2 and S3 matches": ((df["n_s2"] > 0) & (df["n_s3"] > 0)).mean(),
        "S1 with >1 match in the same source": ((df["n_s2"] > 1) | (df["n_s3"] > 1)).mean(),
    }, name="value")
    return {"per_entity": df, "distribution": dist.fillna(0).sort_index(), "per_source": per_source, "stats": stats}


def target_usage(data: SplitData) -> pd.DataFrame:
    usage = defaultdict(set)
    for s, ts in data.truth.items():
        for t in ts:
            usage[t].add(s)
    n_targets = len(data.s2) + len(data.s3)
    counts = Counter(len(v) for v in usage.values())
    rows = [{"S1 entities linked": k, "targets": v} for k, v in sorted(counts.items())]
    rows.append({"S1 entities linked": "0 (unmatched targets)", "targets": n_targets - len(usage)})
    return pd.DataFrame(rows)


def country_consistency(data: SplitData) -> pd.DataFrame:
    s1c = dict(zip(data.s1["entity_id"], data.s1["country"]))
    tc = dict(zip(data.s2["entity_id"], data.s2["country"])) | dict(zip(data.s3["entity_id"], data.s3["country"]))
    rows = []
    for s, ts in data.truth.items():
        for t in ts:
            a, b = s1c.get(s, ""), tc.get(t, "")
            status = ("missing label" if is_missing(a) or is_missing(b)
                      else "same" if normalize_country(a) == normalize_country(b) else "different")
            rows.append({"s1_country": _country_label(a), "target_source": t[:2], "status": status})
    df = pd.DataFrame(rows)
    return df.groupby(["s1_country", "target_source", "status"]).size().unstack(fill_value=0)


def completeness(idx: SplitIndex) -> pd.DataFrame:
    frames = []
    for df in (idx.s1, idx.t):
        frames.append(pd.DataFrame({
            "source": df["source"], "country": df["country_raw"].map(_country_label),
            "name missing": df["name_missing"], "address missing": df["addr_missing"],
            "postal code present": df["addr_postal"].map(bool),
            "house number present": df["addr_house"] != "",
            "landmark reference": df["addr_landmark"], "DBA / alternate name": df["name_alts"].map(bool),
            "legal form present": df["name_legal"] != "",
            "name tokens": df["name_tokens"].map(len), "address tokens": df["addr_tokens"].map(len)}))
    return pd.concat(frames).groupby(["source", "country"]).mean().round(3)


def similarity_profile(idx: SplitIndex, truth, n_neg_per_pos: int = 2, seed: int = 0) -> pd.DataFrame:
    """Similarities of true pairs vs random same-source pairs (long format, for plots and tables)."""
    rng = np.random.default_rng(seed)
    s1_pos = {s: i for i, s in enumerate(idx.s1["entity_id"])}
    t_pos = {s: i for i, s in enumerate(idx.t["entity_id"])}
    pos = [(s1_pos[s], t_pos[t]) for s, ts in truth.items() for t in ts if s in s1_pos and t in t_pos]
    if not pos:
        return pd.DataFrame()
    ia = np.array([p[0] for p in pos])
    ib = np.array([p[1] for p in pos])
    neg_a = rng.integers(0, idx.n_s1, len(pos) * n_neg_per_pos)
    neg_b = []
    src = idx.t["source"].to_numpy()
    for b in np.repeat(ib, n_neg_per_pos):
        cands = idx.source_positions[src[b]]
        neg_b.append(cands[rng.integers(0, len(cands))])
    ia_all = np.r_[ia, neg_a]
    ib_all = np.r_[ib, np.array(neg_b)]
    label = np.r_[np.ones(len(ia)), np.zeros(len(neg_a))]
    df = pd.DataFrame({
        "label": np.where(label == 1, "match", "random non-match"),
        "target_source": src[ib_all],
        "country": idx.s1["country_raw"].to_numpy()[ia_all],
        "name_char_cos": rowwise_dot(*idx.mats["name_char"], ia_all, ib_all),
        "addr_char_cos": rowwise_dot(*idx.mats["addr_char"], ia_all, ib_all),
        "name_token_set": process.cpdist(idx.s1["name_core"].to_numpy()[ia_all].tolist(),
                                         idx.t["name_core"].to_numpy()[ib_all].tolist(),
                                         scorer=fuzz.token_set_ratio, workers=-1) / 100.0,
    })
    miss = idx.s1["addr_missing"].to_numpy()[ia_all] | idx.t["addr_missing"].to_numpy()[ib_all]
    df.loc[miss, "addr_char_cos"] = np.nan
    return df


def save_tables(tables: Dict[str, pd.DataFrame], out_dir: Optional[Path]) -> None:
    if out_dir is None:
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, t in tables.items():
        (t if isinstance(t, pd.DataFrame) else t.to_frame()).to_csv(out_dir / f"{name}.csv")


def pair_view(pairs: pd.DataFrame, data: SplitData) -> pd.DataFrame:
    """Attach the raw records to (s1_id, t_id) pairs for eyeballing errors."""
    s1 = data.s1.set_index("entity_id")
    t = pd.concat([data.s2, data.s3]).set_index("entity_id")
    out = pairs.copy()
    out["s1_name"] = out["s1_id"].map(s1["business_name"])
    out["t_name"] = out["t_id"].map(t["business_name"])
    out["s1_address"] = out["s1_id"].map(s1["business_address"])
    out["t_address"] = out["t_id"].map(t["business_address"])
    out["s1_country"] = out["s1_id"].map(s1["country"])
    out["t_country"] = out["t_id"].map(t["country"])
    return out
