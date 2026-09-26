# Business Entity Resolution — Amazon ML Challenge

For every Source 1 business, this pipeline finds all Source 2 / Source 3 records that describe the same real-world business. It writes `matching_results.tsv` (scored with macro F0.5) and `candidate_pairs.tsv` (the exact candidate set the model scored).

```
data → normalisation → blocking (7 blockers, union, pruning) → ~85 pair features
     → LightGBM matcher (grouped CV, calibrated) → macro-F0.5 decision policy → validated TSVs
```

## Folder layout

```
business_entity_resolution/
├── README.md
├── requirements.txt
└── src/
    ├── run_pipeline.py                    command-line entry point (train / predict / run / explore / validate / score)
    ├── business_entity_resolution.ipynb   step-by-step notebook: exploration → blocking → model → decision → submission
    └── ber/
        ├── config.py       every tunable setting (blocking K values, LightGBM params, decision grids)
        ├── data.py         TSV loading (explicit tab separator, no NA coercion), ground-truth parsing and checks
        ├── normalize.py    one normalisation path for every country (names, addresses, country labels)
        ├── index.py        per-split normalised views + TF-IDF and token matrices (unsupervised, fit per split)
        ├── blocking.py     candidate generation: 7 blockers, union, per-entity pruning
        ├── features.py     pair features: similarities, overlaps, agreement, missingness, provenance, competition
        ├── model.py        LightGBM, stratified group K-fold over S1 entities, isotonic calibration
        ├── decision.py     decision policies tuned for macro F0.5 with cross-fitting
        ├── metrics.py      macro F0.5, breakdowns, blocking recall and oracle upper bound
        ├── explore.py      exploration tables
        ├── plots.py        notebook charts
        ├── submission.py   output writers + a local re-implementation of every submission rule
        └── pipeline.py     orchestration, unseen-country stress test, CLI
```

## Setup

Python 3.11 or 3.12. On Windows (PowerShell) with conda:

```powershell
conda create -n ber python=3.11 -y
conda activate ber
pip install -r requirements.txt
```

Or with a plain virtual environment: `python -m venv .venv`, then `.venv\Scripts\Activate.ps1` (Windows) or `source .venv/bin/activate` (Linux/macOS), then `pip install -r requirements.txt`.

## Run it

`--data-dir` accepts either `student_resource/` or `student_resource/dataset/`. Without it, the code searches the current folder and its parents, and also reads the `BER_DATA_DIR` environment variable.

**Notebook.** Open `src/business_entity_resolution.ipynb` and run all cells. If the dataset is not found automatically, set `STUDENT_RESOURCE_DIR` in the first code cell. It writes `output/`, `artifacts/` and `analysis/` inside `student_resource/`.

**Command line (end to end, used to reproduce the submission):**

```powershell
# from the business_entity_resolution/ folder
python src/run_pipeline.py run --data-dir ../student_resource --out-dir ../student_resource/output --work-dir ../student_resource/artifacts
```

Individual stages:

```powershell
python src/run_pipeline.py explore  --data-dir ../student_resource --analysis-dir ../student_resource/analysis
python src/run_pipeline.py train    --data-dir ../student_resource --work-dir ../student_resource/artifacts
python src/run_pipeline.py predict  --data-dir ../student_resource --work-dir ../student_resource/artifacts --out-dir ../student_resource/output
python src/run_pipeline.py validate --data-dir ../student_resource --out-dir ../student_resource/output
python src/run_pipeline.py score    --pred some_predictions.tsv --truth some_ground_truth.tsv
```

Then run the official checker from `student_resource/`:

```powershell
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

### What each run writes

| Location | File | Contents |
|---|---|---|
| `output/` | `matching_results.tsv` | final matches: one row per test S1 entity, empty list for singletons |
| `output/` | `candidate_pairs.tsv` | the pruned candidate set the model scored; every match is a subset of it |
| `artifacts/` | `model_bundle.pkl` | fold models, calibrator, chosen decision policy, config |
| `artifacts/` | `validation_report.json` | cross-fitted macro F0.5, blocking recall, breakdowns, run metadata |
| `artifacts/` | `policy_summary.csv`, `feature_importance.csv`, `validation_*.csv` | tuning results and breakdowns |
| `artifacts/` | `oof_pairs.csv.gz`, `test_scored_pairs.csv.gz` | out-of-fold and test pair probabilities (for error analysis) |
| `analysis/` | `*.csv` | exploration tables (cardinality, completeness, country consistency and so on) |

## Method in brief

**Normalisation (`normalize.py`).** Unicode accents are folded (é → e), not deleted. Legal forms are recognised and moved out of the core name (Inc, LLC, Corp, Pvt Ltd, LLP, SARL, SAS, GmbH and similar). Common abbreviations are canonicalised (Mfg, Intl, Svcs; St, Rd, Ave, Blvd, Nr, Opp), with each concept mapped to one short form. DBA, "trading as" and parenthetical trade names are split into alternate names, and runs of initials are merged (M.G. → mg, L.L.C. → llc). Each name also gets a phonetic skeleton key, so Agarwal / Aggarwal / Agrawal collide. From each address the code extracts postal-like codes, the house number and the "tail" locality tokens. No code branches on the country value.

**Blocking (`blocking.py`).** Seven blockers run and their results are unioned:

- exact top-k nearest neighbours by character TF-IDF cosine on the name, the address, and a weighted name + address score. These run separately for S2 and S3 targets, in one fused and threaded pass;
- a shared rare name token;
- an exact phonetic-skeleton key and an exact no-space name key;
- the same postal code plus the best name similarity inside that postal bucket;
- acronym ↔ full name.

The union is then pruned per (S1 entity, target source). A pair survives if it is in the top 30 by a cheap score, the top 10 by name, the top 5 by address, or it came from an exact key. Country is never used as a filter. The pruned set is exactly what the model scores and what `candidate_pairs.tsv` contains.

**Features (`features.py`).** About 85 features per pair:

- character and word TF-IDF cosines;
- RapidFuzz ratio, partial, token-sort, token-set and Jaro–Winkler scores;
- token Jaccard, overlap and IDF-weighted overlap, plus the rarest shared token;
- phonetic-skeleton Jaccard, acronym match, legal-form agreement and the best score over DBA alternates;
- address overlaps, numeric-token and house-number agreement, postal match or conflict, and locality-tail coverage;
- missingness flags, country-label agreement (the country value itself is not a feature), target source, and which blockers produced the pair;
- competition features: the rank of a pair and its margin over the best alternative, both among the entity's candidates and among all entities that claim the same target.

**Matcher (`model.py`).** LightGBM binary classifier (MIT licence). It is trained on the full pruned candidate set, which is the same distribution it scores at test time. Validation uses stratified group K-fold over S1 entities, grouped by connected components of the ground-truth graph and stratified by country × match cardinality. Test scores average the fold models, and an isotonic calibrator is fit on the out-of-fold scores.

**Decision (`decision.py`).** Three policies are compared, each with and without the "one S1 per target" constraint:

- a global probability threshold;
- separate S2 and S3 thresholds;
- expected-F0.5 top-k: for each entity, the candidate set with the highest expected F0.5 under the calibrated probabilities, where k = 0 ("no match") is always allowed.

The policy is chosen by cross-fitting. Parameters are tuned on the out-of-fold predictions of four folds and scored on the fifth. When two policies are within 0.0005 macro F0.5 of each other, the simpler one wins. Every S1 entity counts, including singletons and entities without candidates.

**Unseen countries.** France appears only in the test set. `country_holdout_eval` (notebook section 8) trains and tunes on all but one training country, then scores the held-out one. This is the closest offline estimate of performance on a country the model has never seen.

## Configuration

All settings live in `src/ber/config.py`. To override them without editing code, pass a JSON file:

```json
{"blocking": {"name_k": 40, "combined_k": 40, "max_candidates_per_source": 40},
 "decision": {"try_exclusive": false}}
```

```powershell
python src/run_pipeline.py run --data-dir ../student_resource --config my_config.json
```

Useful knobs:

- **Recall vs size:** `name_k`, `address_k`, `combined_k`, `max_candidates_per_source`.
- **Speed and memory:** `n_jobs` (threads for the nearest-neighbour search) and `chunk_cells` (memory per thread ≈ 12 × chunk_cells bytes).

## Runtime and memory

Blocking cost grows with (#S1 × #targets) and uses every core. Everything after blocking grows with the number of candidate pairs, which is about 65 per S1 entity with the defaults. End-to-end timings (`run`) on a 2-core machine with synthetic data in the challenge format:

| Train S1 / targets | Test S1 / targets | Time | Peak RAM |
|---|---|---|---|
| 6,000 / 10,600 | 3,000 / 5,400 | ~1 min | 1.3 GB |
| 40,000 / 71,700 | 20,000 / 35,500 | ~10 min | 3.6 GB |

More cores shorten blocking and LightGBM training roughly in proportion. The notebook takes a little longer because it also runs the unseen-country stress test, which trains two extra models. Results are deterministic run to run: seeds are fixed, all tie-breaks are sorted, and LightGBM runs in deterministic mode.

## Fair play and licences

- Only the provided TSV files are used. There is no external database, API, geocoding or internet lookup, and no pretrained model.
- The lexicons in `normalize.py` (legal forms, street types, honorifics) are hand-written, generic language rules.
- The final model is LightGBM (MIT). Libraries: NumPy, pandas, SciPy and scikit-learn (BSD-3), RapidFuzz (MIT), Matplotlib (PSF-style).
- Everything is reproducible. Seeds are fixed, LightGBM runs in deterministic mode, versions are pinned, and the notebook and the CLI share the same code. The pipeline also ran unchanged with older library versions (NumPy 1.26, pandas 2.2, scikit-learn 1.5, LightGBM 4.5, RapidFuzz 3.9, Matplotlib 3.8). Exact scores can differ slightly across versions, so use the pinned versions to reproduce a submission.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Could not find the dataset folder` | pass `--data-dir` (or `STUDENT_RESOURCE_DIR` in the notebook) or set `BER_DATA_DIR` |
| Out of memory during blocking | lower `chunk_cells` or `n_jobs` |
| Out of memory during training | lower `max_candidates_per_source` (fewer candidate pairs) |
| Blocking recall too low (see `blocking_report` in the notebook) | raise `name_k` / `combined_k` / `address_k` and `max_candidates_per_source`; inspect the "missed" table |
