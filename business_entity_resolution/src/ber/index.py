"""Per-split record index: normalised views + sparse representations shared by blocking and features.

Everything here is *unsupervised* and fit on the records of the split being processed (train or
test). No labels are used, so the same code runs identically at test time, including for
countries that never appear in training.
"""
from __future__ import annotations

import gc
import time
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


def binary_matrices(lists_a, lists_b, batch_size: int = 500_000) -> Tuple[sp.csr_matrix, sp.csr_matrix, Dict[str, int]]:
    """Binary bag-of-tokens matrices for two record sets over one shared vocabulary.

    Streams encoding in chunks to avoid generating tens of millions of Python list items.
    """
    vocab: Dict[str, int] = {}

    for toks in lists_a:
        for t in toks:
            if t not in vocab:
                vocab[t] = len(vocab)
    for toks in lists_b:
        for t in toks:
            if t not in vocab:
                vocab[t] = len(vocab)

    n_vocab = max(len(vocab), 1)

    def encode_chunks(lists):
        n = len(lists)
        chunks = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            sub_lists = lists[start:end] if isinstance(lists, list) else lists.iloc[start:end]
            indptr = [0]
            indices = []
            for toks in sub_lists:
                for t in sorted(set(toks)):
                    idx_val = vocab.get(t)
                    if idx_val is not None:
                        indices.append(idx_val)
                indptr.append(len(indices))
            pa = np.asarray(indptr, dtype=np.int32)
            ia = np.asarray(indices, dtype=np.int32)
            data = np.ones(len(ia), dtype=np.float32)
            chunk_csr = sp.csr_matrix((data, ia, pa), shape=(len(sub_lists), n_vocab), dtype=np.float32)
            chunks.append(chunk_csr)
        if len(chunks) == 1:
            return chunks[0]
        return sp.vstack(chunks, format="csr")

    a = encode_chunks(lists_a)
    b = encode_chunks(lists_b)
    gc.collect()
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


def _transform_batches(vec: TfidfVectorizer, s: pd.Series, batch_size: int = 500_000) -> sp.csr_matrix:
    chunks = []
    n = len(s)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        chunk_mat = vec.transform(s.iloc[start:end])
        chunks.append(chunk_mat)
    if len(chunks) == 1:
        return chunks[0].tocsr()
    return sp.vstack(chunks, format="csr")


def _tfidf_pair(vec: TfidfVectorizer, a: pd.Series, b: pd.Series, sample_size: int = 1_000_000):
    """Fits vocabulary & IDF on a representative sample (or full data if small),

    then transforms in batches of 500k rows to avoid massive Python list memory spikes.
    """
    total_len = len(a) + len(b)
    if total_len <= sample_size:
        full = pd.concat([a, b], ignore_index=True)
        vec.fit(full)
        del full
    else:
        # Fit on a stratified/proportional sample to stay well within RAM limits
        n_a = min(len(a), int(sample_size * (len(a) / total_len)))
        n_b = min(len(b), sample_size - n_a)
        sa = a.sample(n=n_a, random_state=42)
        sb = b.sample(n=n_b, random_state=42)
        sample = pd.concat([sa, sb], ignore_index=True)
        vec.fit(sample)
        del sample, sa, sb
    gc.collect()

    ma = _transform_batches(vec, a)
    mb = _transform_batches(vec, b)
    gc.collect()
    return ma, mb


def build_index(data: SplitData, cfg: BlockingConfig, verbose: bool = True) -> SplitIndex:
    t0 = time.time()
    if verbose:
        print(f"[index] Step 1/6: Normalizing Source-1 records ({len(data.s1):,} rows)...", flush=True)
    s1n = normalize_records(data.s1, "S1")

    n_targets = len(data.s2) + len(data.s3)
    if verbose:
        print(f"[index] Step 2/6: Normalizing Target records ({n_targets:,} rows: {len(data.s2):,} S2 + {len(data.s3):,} S3)...", flush=True)
    s2n = normalize_records(data.s2, "S2")
    s3n = normalize_records(data.s3, "S3")
    len_s2 = len(s2n)
    tn = pd.concat([s2n, s3n], ignore_index=True)
    del s2n, s3n
    gc.collect()

    # Free raw text columns from input dataframes to release 6-8 GB RAM
    data.s1 = data.s1[["entity_id", "country"]]
    data.s2 = data.s2[["entity_id", "country"]]
    data.s3 = data.s3[["entity_id", "country"]]
    gc.collect()

    positions = {
        "S2": np.arange(len_s2, dtype=np.int64),
        "S3": np.arange(len_s2, len(tn), dtype=np.int64)
    }
    idx = SplitIndex(s1=s1n, t=tn, source_positions=positions)

    char_min_df = getattr(cfg, "char_min_df", 5)
    char = dict(analyzer=cfg.char_analyzer, ngram_range=tuple(cfg.char_ngram_range),
                min_df=char_min_df, sublinear_tf=True, dtype=np.float32)
    if verbose:
        print(f"[index] Step 3/6: Fitting character TF-IDF on names and addresses (min_df={char_min_df})...", flush=True)
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
    a, b, _ = binary_matrices(s1n["name_tokens"], tn["name_tokens"])
    idx.mats["name_tok"], idx.idf["name_tok"] = (a, b), smooth_idf(a, b)
    del a, b
    gc.collect()

    a, b, _ = binary_matrices(s1n["name_skel"], tn["name_skel"])
    idx.mats["name_skel"] = (a, b)
    del a, b
    gc.collect()

    if verbose:
        print("[index] Step 6/6: Building address token & postal matrices...", flush=True)
    # address tokens and "tail" tokens share one vocabulary so tail-vs-address coverage can be computed
    a_lists = pd.concat([s1n["addr_tokens"], s1n["addr_tail"]], ignore_index=True)
    b_lists = pd.concat([tn["addr_tokens"], tn["addr_tail"]], ignore_index=True)
    a, b, _ = binary_matrices(a_lists, b_lists)
    del a_lists, b_lists
    gc.collect()

    n1, n2 = len(s1n), len(tn)
    idx.mats["addr_tok"] = (a[:n1], b[:n2])
    idx.mats["addr_tail"] = (a[n1:], b[n2:])
    idx.idf["addr_tok"] = smooth_idf(a[:n1], b[:n2])
    del a, b
    gc.collect()

    a, b, _ = binary_matrices(s1n["addr_numbers"], tn["addr_numbers"])
    idx.mats["addr_num"] = (a, b)
    del a, b
    gc.collect()

    a, b, _ = binary_matrices(s1n["addr_postal"], tn["addr_postal"])
    idx.mats["addr_postal"] = (a, b)
    del a, b
    gc.collect()

    # Free columns from s1n and tn that are never used downstream to save ~5 GB RAM
    unused_cols = ["addr_tokens", "addr_tail", "addr_numbers", "country_raw", "addr_clean"]
    s1n.drop(columns=[c for c in unused_cols if c in s1n.columns], inplace=True)
    tn.drop(columns=[c for c in unused_cols if c in tn.columns], inplace=True)
    gc.collect()

    if verbose:
        sizes = ", ".join(f"{k}={len(v)}" for k, v in positions.items())
        print(f"[index] Completed in {time.time() - t0:.1f}s | S1={n1:,}, targets={n2:,} ({sizes}); vocab={idx.mats['name_char'][0].shape[1]:,}", flush=True)
    return idx
