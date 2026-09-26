"""Business entity resolution pipeline (Amazon ML Challenge).

Stages
------
data       -> load the TSV files, parse and sanity-check the ground truth
normalize  -> country-agnostic text normalisation (names, addresses, country labels)
blocking   -> high-recall candidate generation (union of several independent blockers)
features   -> pairwise similarity + context features for every candidate pair
model      -> LightGBM matcher, grouped cross-validation, isotonic calibration
decision   -> per-Source-1 decision policies tuned for macro F0.5
metrics    -> macro F0.5 and diagnostic breakdowns, blocking recall
submission -> writers for matching_results.tsv / candidate_pairs.tsv and a local validator
pipeline   -> end-to-end train / predict orchestration and the command line interface
"""

__version__ = "1.0.0"
