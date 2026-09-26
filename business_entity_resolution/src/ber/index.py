"""Per-split record index: normalised views + sparse representations shared by blocking and features.

Everything here is *unsupervised* and fit on the records of the split being processed (train or
test). No labels are used, so the same code runs identically at test time, including for
countries that never appear in training.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import BlockingConfig
from .data import SplitData
from .normalize import normalize_records


def _split_words(text: str) -> List[str]:
    return text.split()


def binary_matrices(lists_a: List[List[str]], lists_b: List[List[str]]) -> Tuple[sp.csr_matrix, sp.csr_matrix, Dict[str, int]]:
    """Binary bag-of-tokens matrices for two record sets over one shared vocabulary."""
    vocab: Dict[str, int] = {}

    def encode(lists):
        indptr, indices = [0], []
        for toks in lists:
            for t in sorted(set(toks)):   # sorted: vocabulary ids (and float summation order) identical run to run
                indices.append(vocab.setdefault(t, len(vocab)))
            indptr.append(len(indices))
        return np.asarray(indptr, dtype=np.int64), np.asarray(indices, dtype=np.int64)

    pa, ia = encode(lists_a)
    pb, ib = encode(lists_b)
    n = max(len(vocab), 1)
    a = sp.csr_matrix((np.ones(len(ia), np.float32), ia, pa), shape=(len(lists_a), n))
    b = sp.csr_matrix((np.ones(len(ib), np.float32), ib, pb), shape=(len(lists_b), n))
    return a, b, vocab


def smooth_idf(a: sp.csr_matrix, b: sp.csr_matrix) -> np.ndarray:
    n = a.shape[0] + b.shape[0]
    df = np.asarray(a.sum(axis=0)).ravel() + np.asarray(b.sum(axis=0)).ravel()
    return (np.log((1.0 + n) / (1.0 + df)) + 1.0).astype(np.float32)


@dataclass
class SplitIndex:
    s1: pd.DataFrame                       # normalised Source 1 records (row position = s1 index)
    t: pd.DataFrame                        # normalised S2 + S3 records (row position = target index)
    source_positions: Dict[str, np.ndarray]
    mats: Dict[str, Tuple[sp.csr_matrix, sp.csr_matrix]] = field(default_factory=dict)
    idf: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def n_s1(self) -> int:
        return len(self.s1)

    @property
    def n_t(self) -> int:
        return len(self.t)


def _tfidf_pair(vec: TfidfVectorizer, a: pd.Series, b: pd.Series):
    vec.fit(pd.concat([a, b], ignore_index=True))
    return vec.transform(a).tocsr(), vec.transform(b).tocsr()


def build_index(data: SplitData, cfg: BlockingConfig, verbose: bool = True) -> SplitIndex:
    t0 = time.time()
    if verbose:
        print(f"[index] Step 1/6: Normalizing Source-1 records ({len(data.s1):,} rows)...", flush=True)
    s1n = normalize_records(data.s1, "S1")
    targets = data.targets
    if verbose:
        print(f"[index] Step 2/6: Normalizing Target records ({len(targets):,} rows across S2/S3)...", flush=True)
    tn = normalize_records(targets, targets["source"])
    positions = {src: np.flatnonzero(tn["source"].values == src) for src in sorted(tn["source"].unique())}
    idx = SplitIndex(s1=s1n, t=tn, source_positions=positions)

    char = dict(analyzer=cfg.char_analyzer, ngram_range=tuple(cfg.char_ngram_range), sublinear_tf=True,
                dtype=np.float32)
    if verbose:
        print("[index] Step 3/6: Fitting character TF-IDF on names and addresses...", flush=True)
    idx.mats["name_char"] = _tfidf_pair(TfidfVectorizer(**char), s1n["name_core"], tn["name_core"])
    idx.mats["addr_char"] = _tfidf_pair(TfidfVectorizer(**char), s1n["addr_core"], tn["addr_core"])
    ret = dict(char, ngram_range=tuple(cfg.retrieval_ngram_range), max_df=cfg.retrieval_max_df)
    if ret == char or (ret["ngram_range"] == char["ngram_range"] and cfg.retrieval_max_df >= 1.0):
        idx.mats["name_ret"], idx.mats["addr_ret"] = idx.mats["name_char"], idx.mats["addr_char"]
    else:
        idx.mats["name_ret"] = _tfidf_pair(TfidfVectorizer(**ret), s1n["name_core"], tn["name_core"])
        idx.mats["addr_ret"] = _tfidf_pair(TfidfVectorizer(**ret), s1n["addr_core"], tn["addr_core"])

    word = dict(analyzer=_split_words, sublinear_tf=True, dtype=np.float32)
    if verbose:
        print("[index] Step 4/6: Fitting word TF-IDF on names and addresses...", flush=True)
    idx.mats["name_word"] = _tfidf_pair(TfidfVectorizer(**word), s1n["name_core"], tn["name_core"])
    idx.mats["addr_word"] = _tfidf_pair(TfidfVectorizer(**word), s1n["addr_core"], tn["addr_core"])

    if verbose:
        print("[index] Step 5/6: Building name token & skeleton binary matrices...", flush=True)
    a, b, _ = binary_matrices(s1n["name_tokens"].tolist(), tn["name_tokens"].tolist())
    idx.mats["name_tok"], idx.idf["name_tok"] = (a, b), smooth_idf(a, b)
    a, b, _ = binary_matrices(s1n["name_skel"].tolist(), tn["name_skel"].tolist())
    idx.mats["name_skel"] = (a, b)

    if verbose:
        print("[index] Step 6/6: Building address token & postal matrices...", flush=True)
    # address tokens and "tail" tokens share one vocabulary so tail-vs-address coverage can be computed
    a_lists = s1n["addr_tokens"].tolist() + s1n["addr_tail"].tolist()
    b_lists = tn["addr_tokens"].tolist() + tn["addr_tail"].tolist()
    a, b, _ = binary_matrices(a_lists, b_lists)
    n1, n2 = len(s1n), len(tn)
    idx.mats["addr_tok"] = (a[:n1], b[:n2])
    idx.mats["addr_tail"] = (a[n1:], b[n2:])
    idx.idf["addr_tok"] = smooth_idf(a[:n1], b[:n2])
    a, b, _ = binary_matrices(s1n["addr_numbers"].tolist(), tn["addr_numbers"].tolist())
    idx.mats["addr_num"] = (a, b)
    a, b, _ = binary_matrices(s1n["addr_postal"].tolist(), tn["addr_postal"].tolist())
    idx.mats["addr_postal"] = (a, b)
    if verbose:
        sizes = ", ".join(f"{k}={len(v)}" for k, v in positions.items())
        print(f"[index] Completed in {time.time() - t0:.1f}s | S1={n1:,}, targets={n2:,} ({sizes}); vocab={idx.mats['name_char'][0].shape[1]:,}", flush=True)
    return idx
