"""High-recall candidate generation.

Several independent blockers are run and their outputs are unioned. Each blocker is cheap and
catches a different failure mode of the others:

    b_name   character TF-IDF nearest neighbours on the normalised name     (typos, suffixes, order)
    b_addr   character TF-IDF nearest neighbours on the normalised address  (renamed / DBA businesses)
    b_comb   nearest neighbours on a weighted name+address vector           (common names, chains)
    b_rare   shared rare name token (inverted index)                        (heavy noise elsewhere)
    b_key    exact phonetic-skeleton bag or no-space name key               (transliteration, spacing)
    b_postal same postal code, best name similarity inside the postal bucket
    b_acr    acronym <-> full name ("IBM" vs "International Business Machines")

The three nearest-neighbour blockers are exact (all-pairs cosine, chunked, multi-threaded) and
share one pass over the similarity blocks. They run separately for every target source (S2, S3),
so one source can never crowd the other out of a Source-1 record's candidate list. The union is
then pruned per (Source-1 record, target source): a pair survives if it ranks in the top
`max_candidates_per_source` by a cheap name+address score, or the top `keep_top_name` by name, or
the top `keep_top_address` by address, or came from an exact key. The pruned set is exactly what
the matcher scores and what goes into candidate_pairs.tsv. Country is never used to filter.
"""
from __future__ import annotations

import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .config import BlockingConfig
from .index import SplitIndex

BLOCKERS = ["b_name", "b_addr", "b_comb", "b_rare", "b_key", "b_postal", "b_acr"]
BIT = {name: 1 << i for i, name in enumerate(BLOCKERS)}
PROTECTED_BITS = BIT["b_key"] | BIT["b_acr"]   # exact-key hits are never pruned


# ----------------------------------------------------------------------------- primitives
def _n_threads(n_jobs: int) -> int:
    n = os.cpu_count() or 1
    return max(1, n if n_jobs is None or n_jobs <= 0 else min(n_jobs, n))


def _dense_topk(block: np.ndarray, k: int, row_offset: int):
    """Top-k columns of every row of a dense block; returns (rows, cols, sims) with sims > 0."""
    n_rows, nd = block.shape
    k = min(k, nd)
    if k <= 0:
        return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.float32)
    top = np.argpartition(block, nd - k, axis=1)[:, nd - k:] if k < nd else np.tile(np.arange(nd), (n_rows, 1))
    val = np.take_along_axis(block, top, axis=1).ravel()
    keep = val > 0
    rows = np.repeat(np.arange(row_offset, row_offset + n_rows), top.shape[1])[keep]
    return rows, top.ravel()[keep].astype(np.int64), val[keep].astype(np.float32)


def topk_cosine(q: sp.csr_matrix, d: sp.csr_matrix, k: int, chunk_cells: int = 4_000_000, n_jobs: int = -1):
    """Exact top-k cosine neighbours of every row of q among the rows of d (rows L2-normalised).

    Chunked sparse x sparse product -> dense block -> argpartition, chunks run on a thread pool
    (SciPy sparse products and NumPy partitioning release the GIL). Memory ~ 12 * chunk_cells bytes per thread.
    """
    nq, nd = q.shape[0], d.shape[0]
    if nq == 0 or nd == 0 or k <= 0:
        return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.float32)
    dt = d.T.tocsr()
    step = max(1, int(chunk_cells // nd))

    def work(start):
        return _dense_topk((q[start:start + step] @ dt).toarray(), k, start)

    with ThreadPoolExecutor(_n_threads(n_jobs)) as ex:
        parts = list(ex.map(work, range(0, nq, step)))
    return tuple(np.concatenate([p[i] for p in parts]) for i in range(3))


def rowwise_dot(a: sp.csr_matrix, b: sp.csr_matrix, ia: np.ndarray, ib: np.ndarray,
                batch: int = 400_000) -> np.ndarray:
    """dot(a[ia[n]], b[ib[n]]) for every n, in batches (cosine when rows are L2-normalised)."""
    out = np.zeros(len(ia), dtype=np.float32)
    for s in range(0, len(ia), batch):
        e = s + batch
        out[s:e] = np.asarray(a[ia[s:e]].multiply(b[ib[s:e]]).sum(axis=1)).ravel()
    return out


# ----------------------------------------------------------------------------- blockers
def _knn_blockers(idx: SplitIndex, positions: np.ndarray, cfg: BlockingConfig):
    """Name, address and combined kNN blockers in ONE pass over the S1 x target similarity blocks.

    The combined score is w * name_cos + (1 - w) * address_cos, computed from the two blocks that are
    already in memory instead of a third (and densest) sparse product.
    """
    qn, dn = idx.mats["name_ret"]
    qa, da = idx.mats["addr_ret"]
    dn_t, da_t = dn[positions].T.tocsr(), da[positions].T.tocsr()
    n1, nd = qn.shape[0], len(positions)
    if n1 == 0 or nd == 0:
        empty = (np.empty(0, np.int64), np.empty(0, np.int64))
        return empty, empty, empty
    step = max(1, int(cfg.chunk_cells // nd))
    w = cfg.combined_name_weight

    def work(start):
        bn = (qn[start:start + step] @ dn_t).toarray()
        ba = (qa[start:start + step] @ da_t).toarray()
        out_name = _dense_topk(bn, cfg.name_k, start)
        out_addr = _dense_topk(ba, cfg.address_k, start)
        bn *= w
        ba *= (1.0 - w)
        bn += ba
        out_comb = _dense_topk(bn, cfg.combined_k, start)
        return out_name, out_addr, out_comb

    starts = list(range(0, n1, step))
    total_chunks = len(starts)
    print(f"[blocking] kNN multi-threaded search: {n1:,} S1 x {nd:,} targets | {total_chunks:,} chunks (chunk size {step})...", flush=True)
    t0_knn = time.time()
    last_log = t0_knn
    parts_dict = {}

    with ThreadPoolExecutor(_n_threads(cfg.n_jobs)) as ex:
        futures = {ex.submit(work, s): s for s in starts}
        for done_count, fut in enumerate(as_completed(futures), start=1):
            s = futures[fut]
            parts_dict[s] = fut.result()
            now = time.time()
            if done_count % 500 == 0 or done_count == total_chunks or (now - last_log >= 15.0):
                pct = (done_count / total_chunks) * 100
                elapsed = now - t0_knn
                rate = done_count / max(elapsed, 0.001)
                eta = (total_chunks - done_count) / max(rate, 0.001)
                print(f"[blocking] kNN progress: {done_count:,}/{total_chunks:,} chunks ({pct:.1f}%) | {rate:.1f} chunks/s | ETA: {eta:.0f}s", flush=True)
                last_log = now

    parts = [parts_dict[s] for s in starts]
    result = []
    for j in range(3):
        rows = np.concatenate([p[j][0] for p in parts])
        cols = positions[np.concatenate([p[j][1] for p in parts])]
        result.append((rows, cols))
    return tuple(result)


def _rare_token_blocker(idx: SplitIndex, positions: np.ndarray, cfg: BlockingConfig):
    t_tokens = idx.t["name_tokens"].values
    postings: Dict[str, List[int]] = defaultdict(list)
    for pos in positions:
        for tok in set(t_tokens[pos]):
            postings[tok].append(pos)
    max_df = max(cfg.rare_min_df, int(cfg.rare_df_frac * len(positions)))
    rows, cols = [], []
    for i, toks in enumerate(idx.s1["name_tokens"].values):
        usable = {t for t in toks if (len(t) >= 3 or t.isdigit()) and 0 < len(postings.get(t, ())) <= max_df}
        for tok in sorted(usable, key=lambda t: (len(postings[t]), t))[: cfg.rare_tokens_per_record]:
            hits = postings[tok]
            rows.extend([i] * len(hits))
            cols.extend(hits)
    return np.asarray(rows, np.int64), np.asarray(cols, np.int64)


def _exact_key_blocker(idx: SplitIndex, positions: np.ndarray, max_bucket: int = 200):
    def keys(df):
        skel = [" ".join(sorted(set(s))) for s in df["name_skel"]]
        return skel, list(df["name_nospace"])

    t_skel, t_nospace = keys(idx.t.iloc[positions])
    buckets: Dict[Tuple[int, str], List[int]] = defaultdict(list)
    for pos, k1, k2 in zip(positions, t_skel, t_nospace):
        if k1:
            buckets[(1, k1)].append(pos)
        if len(k2) >= 3:
            buckets[(2, k2)].append(pos)
    s_skel, s_nospace = keys(idx.s1)
    rows, cols = [], []
    for i, (k1, k2) in enumerate(zip(s_skel, s_nospace)):
        hits = set()
        for key in ((1, k1), (2, k2)):
            b = buckets.get(key)
            if b and len(b) <= max_bucket:
                hits.update(b)
        rows.extend([i] * len(hits))
        cols.extend(hits)
    return np.asarray(rows, np.int64), np.asarray(cols, np.int64)


def _postal_blocker(idx: SplitIndex, positions: np.ndarray, cfg: BlockingConfig):
    s1p = [(i, c) for i, codes in enumerate(idx.s1["addr_postal"]) for c in codes]
    tpos_codes = idx.t["addr_postal"].values
    tp = [(int(p), c) for p in positions for c in tpos_codes[p]]
    if not s1p or not tp:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    s1p = pd.DataFrame(s1p, columns=["s1", "code"])
    tp = pd.DataFrame(tp, columns=["t", "code"])
    size = tp.groupby("code")["t"].transform("size")
    tp = tp[size <= cfg.postal_max_bucket]
    pairs = s1p.merge(tp, on="code")[["s1", "t"]].drop_duplicates()
    if pairs.empty:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    a, b = idx.mats["name_char"]
    pairs["sim"] = rowwise_dot(a, b, pairs["s1"].values, pairs["t"].values)
    pairs = pairs[pairs["sim"] > 0.1]
    pairs = pairs.sort_values(["s1", "sim"], ascending=[True, False]).groupby("s1").head(cfg.postal_k)
    return pairs["s1"].to_numpy(np.int64), pairs["t"].to_numpy(np.int64)


def _acronym_blocker(idx: SplitIndex, positions: np.ndarray):
    t = idx.t.iloc[positions]
    t_single = defaultdict(list)   # single-token target names, keyed by the token
    t_acr = defaultdict(list)
    for pos, toks, acr in zip(positions, t["name_tokens"], t["name_acronym"]):
        if len(toks) == 1 and len(toks[0]) >= 2 and toks[0].isalpha():
            t_single[toks[0]].append(pos)
        if len(acr) >= 2:
            t_acr[acr].append(pos)
    rows, cols = [], []
    for i, (toks, acr) in enumerate(zip(idx.s1["name_tokens"], idx.s1["name_acronym"])):
        hits = []
        if len(acr) >= 2:
            hits += t_single.get(acr, [])[:50]
        if len(toks) == 1 and len(toks[0]) >= 2:
            hits += t_acr.get(toks[0], [])[:50]
        rows.extend([i] * len(hits))
        cols.extend(hits)
    return np.asarray(rows, np.int64), np.asarray(cols, np.int64)


# ----------------------------------------------------------------------------- union + prune
def generate_candidates(idx: SplitIndex, cfg: BlockingConfig, verbose: bool = True) -> pd.DataFrame:
    """Union of all blockers, cheap-scored and pruned per source. One row per candidate (s1, t) pair."""
    import gc
    t0 = time.time()
    name_s1, name_t = idx.mats["name_char"]
    addr_s1, addr_t = idx.mats["addr_char"]
    w = cfg.combined_name_weight

    timing = {}
    source_cands = []
    total_union = 0

    for src, pos in idx.source_positions.items():
        if verbose:
            print(f"[blocking] Processing target source {src} ({len(pos):,} records)...", flush=True)
        t1 = time.time()
        parts_rows, parts_cols, parts_bits = [], [], []

        def add(rows, cols, name):
            parts_rows.append(np.asarray(rows, np.int64))
            parts_cols.append(np.asarray(cols, np.int64))
            parts_bits.append(np.full(len(rows), BIT[name], np.int64))

        knn_name, knn_addr, knn_comb = _knn_blockers(idx, pos, cfg)
        add(*knn_name, "b_name")
        add(*knn_addr, "b_addr")
        add(*knn_comb, "b_comb")
        timing[f"knn_{src}"] = time.time() - t1
        t1 = time.time()
        if verbose:
            print(f"[blocking] Running rare token, phonetic key, postal & acronym blockers for {src}...", flush=True)
        add(*_rare_token_blocker(idx, pos, cfg), "b_rare")
        if cfg.use_exact_keys:
            add(*_exact_key_blocker(idx, pos), "b_key")
        if cfg.use_postal_block:
            add(*_postal_blocker(idx, pos, cfg), "b_postal")
        if cfg.use_acronym_block:
            add(*_acronym_blocker(idx, pos), "b_acr")
        timing[f"keys_{src}"] = time.time() - t1

        if not parts_rows or sum(len(x) for x in parts_rows) == 0:
            continue

        if verbose:
            print(f"[blocking {src}] Deduplicating candidate pairs...", flush=True)
        rows = np.concatenate(parts_rows)
        cols = np.concatenate(parts_cols)
        bits = np.concatenate(parts_bits)
        del parts_rows, parts_cols, parts_bits
        gc.collect()

        keys = rows * idx.n_t + cols
        order = np.argsort(keys, kind="stable")
        keys, bits = keys[order], bits[order]
        del order
        gc.collect()

        uniq, start = np.unique(keys, return_index=True)
        bits = np.bitwise_or.reduceat(bits, start) if len(keys) else bits
        s1, t = uniq // idx.n_t, uniq % idx.n_t
        del keys, uniq, start
        gc.collect()

        src_cand = pd.DataFrame({"s1": s1.astype(np.int64), "t": t.astype(np.int64), "bits": bits.astype(np.int64)})
        del s1, t, bits
        gc.collect()

        src_cand["src"] = src
        for name in BLOCKERS:
            src_cand[name] = ((src_cand["bits"].values & BIT[name]) > 0).astype(np.uint8)
        src_cand["n_blocks"] = src_cand[BLOCKERS].sum(axis=1).astype(np.uint8)
        n_src_union = len(src_cand)
        total_union += n_src_union

        # cheap score: name+address cosine
        ia, ib = src_cand["s1"].values, src_cand["t"].values
        src_cand["name_cos"] = rowwise_dot(name_s1, name_t, ia, ib)
        src_cand["addr_cos"] = rowwise_dot(addr_s1, addr_t, ia, ib)
        addr_missing = idx.s1["addr_missing"].values[ia] | idx.t["addr_missing"].values[ib]
        name_missing = idx.s1["name_missing"].values[ia] | idx.t["name_missing"].values[ib]
        comb = w * src_cand["name_cos"].values + (1 - w) * src_cand["addr_cos"].values
        cheap = np.where(addr_missing, src_cand["name_cos"].values, comb)
        src_cand["cheap"] = np.where(name_missing, src_cand["addr_cos"].values, cheap).astype(np.float32)

        if verbose:
            print(f"[blocking {src}] Pruning {n_src_union:,} candidates down to top-{cfg.max_candidates_per_source} per S1...", flush=True)

        # Chunked pruning to avoid huge Pandas groupby.rank memory spike
        kept_slices = []
        chunk_step = 500_000
        for s1_start in range(0, idx.n_s1, chunk_step):
            s1_end = min(s1_start + chunk_step, idx.n_s1)
            sub = src_cand[(src_cand["s1"] >= s1_start) & (src_cand["s1"] < s1_end)].copy()
            if len(sub) == 0:
                continue
            grp = sub.groupby("s1")
            rank_cheap = grp["cheap"].rank(method="first", ascending=False).values
            rank_name = grp["name_cos"].rank(method="first", ascending=False).values
            rank_addr = grp["addr_cos"].rank(method="first", ascending=False).values
            protected = (sub["bits"].values & PROTECTED_BITS) > 0
            keep = (protected
                    | ((rank_cheap <= cfg.max_candidates_per_source) & (sub["cheap"].values >= cfg.min_cheap_score))
                    | ((rank_name <= cfg.keep_top_name) & (sub["name_cos"].values >= cfg.min_cheap_score))
                    | ((rank_addr <= cfg.keep_top_address) & (sub["addr_cos"].values >= 0.5)))
            kept_slices.append(sub[keep].drop(columns=["bits"]))

        del src_cand
        gc.collect()

        if kept_slices:
            src_pruned = pd.concat(kept_slices, ignore_index=True)
            source_cands.append(src_pruned)
            if verbose:
                print(f"[blocking {src}] Retained {len(src_pruned):,} candidates ({len(src_pruned)/max(n_src_union, 1):.1%})", flush=True)
        del kept_slices
        gc.collect()

    if source_cands:
        cand = pd.concat(source_cands, ignore_index=True)
    else:
        cand = pd.DataFrame(columns=["s1", "t", "src", *BLOCKERS, "n_blocks", "name_cos", "addr_cos", "cheap"])

    cand["s1_id"] = idx.s1["entity_id"].values[cand["s1"].values]
    cand["t_id"] = idx.t["entity_id"].values[cand["t"].values]
    del source_cands
    gc.collect()

    if verbose:
        per = cand.groupby("s1").size() if len(cand) else pd.Series([0])
        print(f"[blocking] Total union={total_union:,} -> final kept={len(cand):,} pairs | per S1: mean={per.mean():.1f}, "
              f"p95={per.quantile(0.95):.0f}, max={per.max()} | S1 without candidates="
              f"{idx.n_s1 - cand['s1'].nunique()} | {time.time() - t0:.1f}s", flush=True)
    cand.attrs["n_union"] = total_union
    cand.attrs["timing"] = timing
    return cand
