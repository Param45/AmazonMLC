"""All tunable settings in one place.

Nothing in here is country specific: the pipeline treats `country` as an open set of labels,
so an unseen country (France in the test set) flows through exactly the same code path.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class BlockingConfig:
    # character TF-IDF on normalised names / addresses used by the FEATURES (fit per split, unsupervised)
    char_analyzer: str = "char_wb"
    char_ngram_range: tuple = (3, 4)   # (3, 4) n-grams: cuts uninformative 2-grams by 45% while preserving matching power
    char_min_df: int = 5               # drops rare typo ngrams across 12.5M rows to keep matrix non-zeros sparse
    # optional separate TF-IDF for nearest-neighbour RETRIEVAL only
    retrieval_ngram_range: tuple = (3, 4)
    retrieval_max_df: float = 1.0
    # nearest neighbours kept per Source-1 record, *per target source* (S2 and S3 separately)
    name_k: int = 20
    address_k: int = 10
    combined_k: int = 20
    combined_name_weight: float = 0.6  # weight of name vs address in the combined retriever
    # rare-token inverted index: a token is "rare" if it occurs in <= max(rare_min_df, rare_df_frac * n_targets) targets
    rare_min_df: int = 30
    rare_df_frac: float = 0.002
    rare_tokens_per_record: int = 2
    # exact keys on phonetic skeleton / no-space name, postal code + name, acronyms
    use_exact_keys: bool = True
    use_postal_block: bool = True
    postal_max_bucket: int = 400
    postal_k: int = 5
    use_acronym_block: bool = True
    # final pruning: keep at most this many candidates per (Source-1 record, target source),
    # ranked by a cheap score. This pruned set is what the model scores (-> candidate_pairs.tsv).
    max_candidates_per_source: int = 20
    keep_top_name: int = 10        # ...plus the best few by name similarity alone
    keep_top_address: int = 5      # ...and by address similarity alone (only if address cosine >= 0.5)
    min_cheap_score: float = 0.05
    # exact top-k search: cells (S1 rows x targets) per chunk and per thread, and number of threads (-1 = all cores)
    chunk_cells: int = 5_000_000
    n_jobs: int = -1


@dataclass
class ModelConfig:
    n_folds: int = 5
    seed: int = 42
    num_boost_round: int = 3000
    early_stopping_rounds: int = 100
    params: dict = field(default_factory=lambda: {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 40,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "max_bin": 255,
        "verbose": -1,
        "num_threads": 0,          # 0 = all cores
        "deterministic": True,
        "force_row_wise": True,
        "seed": 42,
    })
    calibrate: bool = True        # isotonic calibration fit on out-of-fold predictions


@dataclass
class DecisionConfig:
    beta: float = 0.5
    threshold_grid: tuple = tuple(round(0.05 + 0.01 * i, 2) for i in range(91))       # 0.05 .. 0.95
    source_threshold_grid: tuple = tuple(round(0.10 + 0.05 * i, 2) for i in range(18))  # 0.10 .. 0.95
    expected_f_gamma_grid: tuple = (0.8, 1.0, 1.25, 1.5)   # p -> p ** gamma before expected-F
    min_prob: float = 1e-3                                  # candidates below this are ignored by expected-F
    try_exclusive: bool = True                              # also evaluate "one Source-1 per target" variants
    tolerance: float = 0.0005                               # prefer a simpler policy if within this macro-F0.5 of the best


@dataclass
class Config:
    blocking: BlockingConfig = field(default_factory=BlockingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)

    def to_dict(self) -> dict:
        return asdict(self)
