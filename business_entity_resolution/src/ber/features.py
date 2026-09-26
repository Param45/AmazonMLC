"""Pairwise features for every candidate pair (vectorised: sparse row-wise ops + RapidFuzz cpdist).

Missing information is encoded as NaN (LightGBM routes NaN natively) rather than as a fake 0 so
the model can tell "addresses disagree" from "one address is missing".
No feature encodes the country *value*; only whether the two country labels agree.
"""
from __future__ import annotations

import time
import warnings
from typing import List

import numpy as np
import pandas as pd
import scipy.sparse as sp
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from .blocking import BLOCKERS, rowwise_dot
from .index import SplitIndex


def _cp(scorer, a, b, scale: float = 100.0) -> np.ndarray:
    return process.cpdist(list(a), list(b), scorer=scorer, workers=-1, dtype=np.float32) / scale


def _rowwise_max_shared_idf(a: sp.csr_matrix, b: sp.csr_matrix, ia, ib, idf, batch: int = 400_000):
    out = np.zeros(len(ia), dtype=np.float32)
    d = sp.diags(idf.astype(np.float32))
    for s in range(0, len(ia), batch):
        e = s + batch
        shared = a[ia[s:e]].multiply(b[ib[s:e]]).tocsr() @ d
        out[s:e] = shared.max(axis=1).toarray().ravel()
    return out


def _set_stats(idx: SplitIndex, key: str, ia, ib, prefix: str, with_idf: bool) -> dict:
    a, b = idx.mats[key]
    inter = rowwise_dot(a, b, ia, ib)
    na = np.asarray(a.sum(axis=1)).ravel()[ia]
    nb = np.asarray(b.sum(axis=1)).ravel()[ib]
    union = na + nb - inter
    both = (na > 0) & (nb > 0)
    out = {
        f"{prefix}_inter": np.where(both, inter, np.nan),
        f"{prefix}_jacc": np.where(both, inter / np.maximum(union, 1), np.nan),
        f"{prefix}_overlap": np.where(both, inter / np.maximum(np.minimum(na, nb), 1), np.nan),
    }
    if with_idf:
        idf = idx.idf[key]
        sum_a = (a @ idf)[ia]
        sum_b = (b @ idf)[ib]
        shared = rowwise_dot(a @ sp.diags(idf), b, ia, ib)
        out[f"{prefix}_idf_jacc"] = np.where(both, shared / np.maximum(sum_a + sum_b - shared, 1e-6), np.nan)
        out[f"{prefix}_max_shared_idf"] = np.where(both, _rowwise_max_shared_idf(a, b, ia, ib, idf), np.nan)
        out[f"{prefix}_unshared_idf"] = np.where(both, sum_a + sum_b - 2 * shared, np.nan)
    return out


def _group_context(df: pd.DataFrame, keys: List[str], score: np.ndarray, prefix: str) -> dict:
    """Rank of the pair inside its group, margin to the best *other* member, and group size."""
    gid = df.groupby(keys, sort=False).ngroup().to_numpy()
    n_groups = gid.max() + 1 if len(gid) else 0
    order = np.lexsort((-score, gid))
    sg, ss = gid[order], score[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sg)) + 1] if len(sg) else np.array([], int)
    sizes = np.diff(np.r_[starts, len(sg)])
    best = np.zeros(n_groups)
    second = np.zeros(n_groups)
    best[sg[starts]] = ss[starts]
    has2 = sizes > 1
    second[sg[starts[has2]]] = ss[starts[has2] + 1]
    rank_sorted = np.arange(len(sg)) - np.repeat(starts, sizes)
    rank = np.empty(len(sg), dtype=np.int32)
    rank[order] = rank_sorted + 1
    is_top = rank == 1
    best_other = np.where(is_top, second[gid], best[gid])
    size = np.zeros(n_groups, dtype=np.int32)
    size[sg[starts]] = sizes
    return {f"{prefix}_rank": rank, f"{prefix}_margin": score - best_other,
            f"{prefix}_gap_to_best": best[gid] - score, f"{prefix}_n": size[gid]}


def build_features(idx: SplitIndex, cand: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    t0 = time.time()
    if verbose:
        print(f"[features] Computing ~85 pairwise features for {len(cand):,} candidate pairs...", flush=True)
    ia, ib = cand["s1"].to_numpy(), cand["t"].to_numpy()
    s1, t = idx.s1, idx.t
    f: dict = {}

    def col(df, name):
        return df[name].to_numpy()[ia] if df is s1 else df[name].to_numpy()[ib]

    # ---------------------------------------------------------------- names
    if verbose:
        print("[features] 1/4: Name similarities & RapidFuzz distances...", flush=True)
    n1c, n2c = col(s1, "name_core"), col(t, "name_core")
    n_miss = col(s1, "name_missing") | col(t, "name_missing")
    f["name_char_cos"] = cand["name_cos"].to_numpy()
    f["name_word_cos"] = rowwise_dot(*idx.mats["name_word"], ia, ib)
    f.update(_set_stats(idx, "name_tok", ia, ib, "name_tok", with_idf=True))
    f["name_skel_jacc"] = _set_stats(idx, "name_skel", ia, ib, "name_skel", with_idf=False)["name_skel_jacc"]
    f["name_ratio"] = _cp(fuzz.ratio, n1c, n2c)
    f["name_partial"] = _cp(fuzz.partial_ratio, n1c, n2c)
    f["name_token_sort"] = _cp(fuzz.token_sort_ratio, n1c, n2c)
    f["name_token_set"] = _cp(fuzz.token_set_ratio, n1c, n2c)
    f["name_jaro_winkler"] = _cp(JaroWinkler.normalized_similarity, n1c, n2c, scale=1.0)
    f["name_nospace_ratio"] = _cp(fuzz.ratio, col(s1, "name_nospace"), col(t, "name_nospace"))
    f["name_clean_ratio"] = _cp(fuzz.ratio, col(s1, "name_clean"), col(t, "name_clean"))
    f["name_exact_core"] = (n1c == n2c).astype(np.float32)
    f["name_exact_nospace"] = (col(s1, "name_nospace") == col(t, "name_nospace")).astype(np.float32)
    l1 = np.fromiter((len(x) for x in n1c), np.float32, len(n1c))
    l2 = np.fromiter((len(x) for x in n2c), np.float32, len(n2c))
    f["name_len_ratio"] = np.minimum(l1, l2) / np.maximum(np.maximum(l1, l2), 1)
    k1 = np.fromiter((len(x) for x in col(s1, "name_tokens")), np.float32, len(ia))
    k2 = np.fromiter((len(x) for x in col(t, "name_tokens")), np.float32, len(ib))
    f["name_ntok_s1"], f["name_ntok_t"], f["name_ntok_diff"] = k1, k2, np.abs(k1 - k2)
    first1 = np.array([x[0] if x else "" for x in col(s1, "name_tokens")], dtype=object)
    first2 = np.array([x[0] if x else "" for x in col(t, "name_tokens")], dtype=object)
    f["name_first_token_eq"] = (first1 == first2).astype(np.float32)
    acr1, acr2 = col(s1, "name_acronym"), col(t, "name_acronym")
    ns1, ns2 = col(s1, "name_nospace"), col(t, "name_nospace")
    f["name_acronym_match"] = (((acr1 != "") & (acr1 == ns2)) | ((acr2 != "") & (acr2 == ns1))).astype(np.float32)
    leg1, leg2 = col(s1, "name_legal"), col(t, "name_legal")
    both_leg = (leg1 != "") & (leg2 != "")
    leg_j = np.array([len(set(a.split()) & set(b.split())) / len(set(a.split()) | set(b.split()))
                      if (a and b) else np.nan for a, b in zip(leg1, leg2)], dtype=np.float32)
    f["legal_form_jacc"] = np.where(both_leg, leg_j, np.nan)
    f["legal_form_one_side"] = ((leg1 != "") ^ (leg2 != "")).astype(np.float32)
    # DBA / parenthetical alternates: best token-set similarity over all name parts
    alts1, alts2 = col(s1, "name_alts"), col(t, "name_alts")
    alt_best = f["name_token_set"].copy()
    has_alt = np.flatnonzero(np.fromiter((bool(a) or bool(b) for a, b in zip(alts1, alts2)), bool, len(ia)))
    for i in has_alt:
        p1 = (alts1[i] or []) + [n1c[i]]
        p2 = (alts2[i] or []) + [n2c[i]]
        alt_best[i] = max(fuzz.token_set_ratio(a, b) for a in p1 for b in p2 if a and b) / 100.0 \
            if any(p1) and any(p2) else alt_best[i]
    f["name_alt_best"] = alt_best
    f["name_has_alt"] = np.zeros(len(ia), np.float32)
    f["name_has_alt"][has_alt] = 1.0
    for k in ["name_char_cos", "name_word_cos", "name_ratio", "name_partial", "name_token_sort", "name_token_set",
              "name_jaro_winkler", "name_nospace_ratio", "name_clean_ratio", "name_exact_core",
              "name_exact_nospace", "name_len_ratio", "name_first_token_eq", "name_alt_best"]:
        f[k] = np.where(n_miss, np.nan, f[k]).astype(np.float32)
    f["name_missing_s1"] = col(s1, "name_missing").astype(np.float32)
    f["name_missing_t"] = col(t, "name_missing").astype(np.float32)

    # ---------------------------------------------------------------- addresses
    if verbose:
        print("[features] 2/4: Address similarities & locality tails...", flush=True)
    a1c, a2c = col(s1, "addr_core"), col(t, "addr_core")
    a_miss = col(s1, "addr_missing") | col(t, "addr_missing")
    f["addr_char_cos"] = cand["addr_cos"].to_numpy()
    f["addr_word_cos"] = rowwise_dot(*idx.mats["addr_word"], ia, ib)
    f.update(_set_stats(idx, "addr_tok", ia, ib, "addr_tok", with_idf=True))
    f["addr_ratio"] = _cp(fuzz.ratio, a1c, a2c)
    f["addr_partial"] = _cp(fuzz.partial_ratio, a1c, a2c)
    f["addr_token_set"] = _cp(fuzz.token_set_ratio, a1c, a2c)
    f["addr_token_sort"] = _cp(fuzz.token_sort_ratio, a1c, a2c)
    num = _set_stats(idx, "addr_num", ia, ib, "addr_num", with_idf=False)
    f["addr_num_inter"], f["addr_num_jacc"] = num["addr_num_inter"], num["addr_num_jacc"]
    p = _set_stats(idx, "addr_postal", ia, ib, "postal", with_idf=False)
    both_postal = ~np.isnan(p["postal_inter"])
    f["postal_both_present"] = both_postal.astype(np.float32)
    f["postal_match"] = np.where(both_postal, (np.nan_to_num(p["postal_inter"]) > 0), np.nan).astype(np.float32)
    h1, h2 = col(s1, "addr_house"), col(t, "addr_house")
    both_house = (h1 != "") & (h2 != "")
    f["house_number_eq"] = np.where(both_house, h1 == h2, np.nan).astype(np.float32)
    tail1, tok2 = idx.mats["addr_tail"][0], idx.mats["addr_tok"][1]
    tok1, tail2 = idx.mats["addr_tok"][0], idx.mats["addr_tail"][1]
    nt1 = np.asarray(tail1.sum(axis=1)).ravel()[ia]
    nt2 = np.asarray(tail2.sum(axis=1)).ravel()[ib]
    cov1 = rowwise_dot(tail1, tok2, ia, ib) / np.maximum(nt1, 1)
    cov2 = rowwise_dot(tok1, tail2, ia, ib) / np.maximum(nt2, 1)
    both_tail = (nt1 > 0) & (nt2 > 0)
    f["addr_tail_cov_max"] = np.where(both_tail, np.maximum(cov1, cov2), np.nan)
    f["addr_tail_cov_min"] = np.where(both_tail, np.minimum(cov1, cov2), np.nan)
    la1 = np.fromiter((len(x) for x in a1c), np.float32, len(a1c))
    la2 = np.fromiter((len(x) for x in a2c), np.float32, len(a2c))
    f["addr_len_ratio"] = np.minimum(la1, la2) / np.maximum(np.maximum(la1, la2), 1)
    for k in ["addr_char_cos", "addr_word_cos", "addr_ratio", "addr_partial", "addr_token_set", "addr_token_sort",
              "addr_len_ratio"]:
        f[k] = np.where(a_miss, np.nan, f[k]).astype(np.float32)
    f["addr_missing_s1"] = col(s1, "addr_missing").astype(np.float32)
    f["addr_missing_t"] = col(t, "addr_missing").astype(np.float32)
    f["addr_postal_s1"] = (np.asarray(idx.mats["addr_postal"][0].sum(axis=1)).ravel()[ia] > 0).astype(np.float32)
    f["addr_postal_t"] = (np.asarray(idx.mats["addr_postal"][1].sum(axis=1)).ravel()[ib] > 0).astype(np.float32)
    f["landmark_s1"] = col(s1, "addr_landmark").astype(np.float32)
    f["landmark_t"] = col(t, "addr_landmark").astype(np.float32)

    # ---------------------------------------------------------------- metadata
    if verbose:
        print("[features] 3/4: Metadata & blocker features...", flush=True)
    c1, c2 = col(s1, "country_norm"), col(t, "country_norm")
    f["country_eq"] = np.where((c1 != "") & (c2 != ""), c1 == c2, np.nan).astype(np.float32)
    f["target_is_s3"] = (cand["src"].to_numpy() == "S3").astype(np.float32)
    for b in BLOCKERS:
        f[b] = cand[b].to_numpy().astype(np.float32)
    f["n_blocks"] = cand["n_blocks"].to_numpy().astype(np.float32)
    f["cheap_score"] = cand["cheap"].to_numpy().astype(np.float32)

    # ---------------------------------------------------------------- competition / context
    if verbose:
        print("[features] 4/4: Context & mutual competition rank features...", flush=True)
    base = cand[["s1", "t", "src"]].copy()
    cheap = cand["cheap"].to_numpy().astype(np.float64)
    # blended name similarity (token_set alone saturates at 1.0 for subset names like "Rao Pharma")
    with warnings.catch_warnings():   # rows where the name is missing are all-NaN -> 0 below
        warnings.simplefilter("ignore", category=RuntimeWarning)
        name_mix = np.nanmean(np.vstack([f["name_char_cos"], f["name_token_sort"], f["name_jaro_winkler"],
                                         f["name_alt_best"]]), axis=0) if len(ia) else np.zeros(0)
    name_mix = np.nan_to_num(name_mix, nan=0.0).astype(np.float64)
    for key, score, prefix in ((["s1", "src"], cheap, "ctx_s1_cheap"), (["s1", "src"], name_mix, "ctx_s1_name"),
                               (["t"], cheap, "ctx_t_cheap"), (["t"], name_mix, "ctx_t_name")):
        f.update(_group_context(base, key, score, prefix))
    f["ctx_mutual_best"] = ((f["ctx_s1_cheap_rank"] == 1) & (f["ctx_t_cheap_rank"] == 1)).astype(np.float32)

    X = pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in f.items()})
    if verbose:
        print(f"[features] {X.shape[0]:,} pairs x {X.shape[1]} features in {time.time() - t0:.1f}s")
    return X
