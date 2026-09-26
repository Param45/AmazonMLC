"""End-to-end orchestration and the command-line interface.

    python src/run_pipeline.py run      --data-dir <student_resource>   # train + predict + write + validate
    python src/run_pipeline.py train    --data-dir <student_resource> --work-dir artifacts
    python src/run_pipeline.py predict  --data-dir <student_resource> --work-dir artifacts --out-dir output
    python src/run_pipeline.py explore  --data-dir <student_resource> --analysis-dir analysis
    python src/run_pipeline.py validate --data-dir <student_resource> --out-dir output
    python src/run_pipeline.py score    --pred output/matching_results.tsv --truth <ground_truth.tsv>
"""
from __future__ import annotations

import argparse
import json
import pickle
import platform
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Optional, Set

import numpy as np
import pandas as pd

from . import __version__
from .blocking import generate_candidates
from .config import Config
from .data import SplitData, load_ground_truth, load_split, read_tsv, resolve_dataset_dir, truth_from_gt
from .decision import Policy, policy_mask, predictions_from_mask, tune_policy
from .explore import (cardinality, completeness, country_consistency, country_table, overview, save_tables,
                      target_usage)
from .features import build_features
from .index import SplitIndex, build_index
from .metrics import blocking_report, evaluation_report, macro_fbeta
from .model import Matcher, make_folds, train_matcher
from .normalize import normalize_country
from .submission import run_official_validator, validate_submission, write_submission


# ----------------------------------------------------------------------------- containers
@dataclass
class PreparedSplit:
    data: SplitData
    index: SplitIndex
    cand: pd.DataFrame
    X: pd.DataFrame
    y: Optional[np.ndarray] = None


@dataclass
class TrainedPipeline:
    config: Config
    matcher: Matcher
    policy: Policy
    validation: Dict[str, object] = field(default_factory=dict)
    version: str = __version__

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        return path

    @staticmethod
    def load(path) -> "TrainedPipeline":
        with open(path, "rb") as f:
            return pickle.load(f)


@dataclass
class TrainResult:
    trained: TrainedPipeline
    prepared: PreparedSplit
    oof: pd.DataFrame
    s1_fold: Dict[str, int]
    policy_summary: pd.DataFrame
    crossfit_f: float
    report: Dict[str, object]


@dataclass
class PredictResult:
    prepared: PreparedSplit
    scored: pd.DataFrame
    matches: Dict[str, Set[str]]
    candidates: Dict[str, Set[str]]


# ----------------------------------------------------------------------------- stages
def label_pairs(cand: pd.DataFrame, truth: Dict[str, Set[str]]) -> np.ndarray:
    pos = {(s, t) for s, ts in truth.items() for t in ts}
    return np.fromiter(((s, t) in pos for s, t in zip(cand["s1_id"], cand["t_id"])), dtype=np.int8, count=len(cand))


def prepare_split(data: SplitData, cfg: Config, verbose: bool = True) -> PreparedSplit:
    idx = build_index(data, cfg.blocking, verbose)
    cand = generate_candidates(idx, cfg.blocking, verbose)
    X = build_features(idx, cand, verbose)
    y = label_pairs(cand, data.truth) if data.truth is not None else None
    return PreparedSplit(data, idx, cand, X, y)


def country_map(data: SplitData) -> Dict[str, str]:
    return dict(zip(data.s1["entity_id"], data.s1["country"]))


def train(data: SplitData, cfg: Config, verbose: bool = True, prepared: Optional[PreparedSplit] = None) -> TrainResult:
    if data.truth is None:
        raise ValueError("training split has no ground truth")
    prep = prepared if prepared is not None else prepare_split(data, cfg, verbose)
    countries = country_map(data)
    s1_fold = make_folds(data.truth, {k: normalize_country(v) for k, v in countries.items()},
                         cfg.model.n_folds, cfg.model.seed)
    pair_fold = prep.cand["s1_id"].map(s1_fold).to_numpy()
    matcher, oof_raw, oof_cal = train_matcher(prep.X, prep.y, pair_fold, cfg.model, verbose)
    oof = prep.cand[["s1_id", "t_id", "src"]].copy()
    oof["p_raw"], oof["p"], oof["y"], oof["fold"] = oof_raw, oof_cal, prep.y, pair_fold
    policy, summary, crossfit = tune_policy(oof, data.truth, s1_fold, cfg.decision, verbose)
    pred = predictions_from_mask(oof, policy_mask(oof, policy, cfg.decision))
    report = evaluation_report(pred, data.truth, list(data.truth), countries, cfg.decision.beta)
    blk = blocking_report(prep.cand, data.truth, prep.index.n_s1, prep.index.n_t, countries, cfg.decision.beta)
    validation = {
        "crossfit_macro_F0.5": crossfit,
        "oof_macro_F0.5_selected_policy": float(report["headline"]["macro_F0.5"]),
        "policy": policy.describe(),
        "headline": report["headline"].to_dict(),
        "blocking": blk["headline"].to_dict(),
        "model": matcher.cv_info,
    }
    trained = TrainedPipeline(cfg, matcher, policy, validation)
    return TrainResult(trained, prep, oof, s1_fold, summary, crossfit, report)


def country_label_warnings(data: SplitData):
    """Country agreement is a model feature, so a label spelled differently across sources (e.g. "France"
    in S1 but "FR" in S2) would silently suppress that country's matches. Flag such labels."""
    s1_labels = set(data.s1["country"].map(normalize_country)) - {""}
    t_labels = set(data.s2["country"].map(normalize_country)) | set(data.s3["country"].map(normalize_country))
    return sorted(s1_labels - t_labels)


def predict(data: SplitData, trained: TrainedPipeline, verbose: bool = True,
            prepared: Optional[PreparedSplit] = None) -> PredictResult:
    orphan = country_label_warnings(data)
    if orphan:
        print(f"[predict] WARNING: S1 country labels never used in S2/S3: {orphan}. The 'country_eq' feature will be 0 "
              f"for every pair of these entities - check the label spellings (see explore.country_table).")
    prep = prepared if prepared is not None else prepare_split(data, trained.config, verbose)
    scored = prep.cand[["s1_id", "t_id", "src"]].copy()
    scored["p"] = trained.matcher.predict(prep.X) if len(scored) else np.zeros(0)
    mask = policy_mask(scored, trained.policy, trained.config.decision) if len(scored) else np.zeros(0, bool)
    matches = predictions_from_mask(scored, mask)
    candidates = scored.groupby("s1_id")["t_id"].apply(set).to_dict()
    if verbose:
        n_match = sum(bool(v) for v in matches.values())
        print(f"[predict] {len(data.s1)} S1 entities, {n_match} with >=1 match "
              f"({n_match / max(len(data.s1), 1):.1%}), {int(mask.sum())} matched pairs")
    return PredictResult(prep, scored, matches, candidates)


def write_outputs(data: SplitData, result: PredictResult, out_dir, dataset_dir, verbose: bool = True):
    s1_ids = data.s1["entity_id"].tolist()
    m_path, c_path = write_submission(out_dir, s1_ids, result.matches, result.candidates)
    errors, warnings = validate_submission(m_path, c_path, Path(dataset_dir) / "test")
    if verbose:
        print(f"[output] wrote {m_path} and {c_path}")
        print("[validate] " + ("PASS" if not errors else "FAIL:\n  " + "\n  ".join(errors)))
        for w in warnings:
            print(f"[validate] warning: {w}")
        code, msg = run_official_validator(Path(dataset_dir).parent, m_path, c_path, Path(dataset_dir) / "test")
        print(f"[validate] official validator: {msg if code is None else ('exit ' + str(code) + ' | ' + msg)}")
    return m_path, c_path, errors, warnings


def explore_split(data: SplitData, cfg: Config, analysis_dir=None, index: Optional[SplitIndex] = None) -> Dict[str, pd.DataFrame]:
    idx = index if index is not None else build_index(data, cfg.blocking, verbose=False)
    card = cardinality(data)
    tables = {"overview": overview(data), "country_by_source": country_table(data),
              "cardinality_distribution": card["distribution"], "cardinality_per_source": card["per_source"],
              "cardinality_stats": card["stats"].to_frame(), "target_usage": target_usage(data),
              "country_consistency_of_matches": country_consistency(data), "completeness": completeness(idx)}
    save_tables(tables, analysis_dir)
    return tables


def country_holdout_eval(prep: PreparedSplit, data: SplitData, cfg: Config, holdout: str, n_folds: int = 3,
                         verbose: bool = False) -> Dict[str, object]:
    """Unseen-country stress test: train + tune on the other countries only, score the held-out one.

    This is the closest offline proxy for France, which appears only in the test set.
    """
    countries = {k: normalize_country(v) for k, v in country_map(data).items()}
    hold = normalize_country(holdout)
    truth_in = {s: v for s, v in data.truth.items() if countries.get(s) != hold}
    truth_out = {s: v for s, v in data.truth.items() if countries.get(s) == hold}
    if not truth_in or not truth_out:
        raise ValueError(f"need entities both inside and outside {holdout!r}")
    in_mask = prep.cand["s1_id"].map(lambda s: countries.get(s) != hold).to_numpy()
    s1_fold = make_folds(truth_in, countries, n_folds, cfg.model.seed)
    cand_in = prep.cand.loc[in_mask, ["s1_id", "t_id", "src"]].reset_index(drop=True)
    X_in, y_in = prep.X[in_mask].reset_index(drop=True), prep.y[in_mask]
    matcher, _, oof_cal = train_matcher(X_in, y_in, cand_in["s1_id"].map(s1_fold).to_numpy(),
                                        replace(cfg.model, n_folds=n_folds), verbose)
    oof = cand_in.assign(p=oof_cal, y=y_in)
    policy, _, crossfit_in = tune_policy(oof, truth_in, s1_fold, cfg.decision, verbose)
    out = prep.cand.loc[~in_mask, ["s1_id", "t_id", "src"]].reset_index(drop=True)
    out["p"] = matcher.predict(prep.X[~in_mask].reset_index(drop=True)) if len(out) else np.zeros(0)
    pred = predictions_from_mask(out, policy_mask(out, policy, cfg.decision)) if len(out) else {}
    rep = evaluation_report(pred, truth_out, list(truth_out), None, cfg.decision.beta)["headline"]
    return {"held-out country": holdout, "train entities (other countries)": len(truth_in),
            "held-out entities": len(truth_out), "cross-fit F0.5 on training countries": crossfit_in,
            "F0.5 on held-out country": rep["macro_F0.5"], "held-out singleton accuracy": rep["singleton_accuracy (empty predicted)"],
            "held-out pair precision": rep["pair_precision"], "held-out pair recall": rep["pair_recall"],
            "policy tuned on training countries": policy.describe()}


# ----------------------------------------------------------------------------- CLI
def _load_config(path: Optional[str]) -> Config:
    cfg = Config()
    if path:
        overrides = json.loads(Path(path).read_text())
        for section, values in overrides.items():
            target = getattr(cfg, section)
            for k, v in values.items():
                if not hasattr(target, k):
                    raise KeyError(f"unknown config key {section}.{k}")
                current = getattr(target, k)
                if isinstance(current, dict):
                    current.update(v)            # e.g. {"model": {"params": {"num_leaves": 127}}} keeps other params
                else:
                    setattr(target, k, tuple(v) if isinstance(current, tuple) else v)
    return cfg


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return str(o)


def cmd_train(args) -> TrainedPipeline:
    cfg = _load_config(args.config)
    ds = resolve_dataset_dir(args.data_dir)
    work = Path(args.work_dir)
    t0 = time.time()
    data = load_split(ds, "train")
    print(f"[train] dataset={ds} sizes={data.summary()}")
    res = train(data, cfg)
    work.mkdir(parents=True, exist_ok=True)
    res.trained.save(work / "model_bundle.pkl")
    res.policy_summary.to_csv(work / "policy_summary.csv", index=False)
    res.trained.matcher.importance.to_csv(work / "feature_importance.csv")
    res.oof.to_csv(work / "oof_pairs.csv.gz", index=False)
    for name, table in res.report["tables"].items():
        table.to_csv(work / f"validation_{name}.csv")
    meta = dict(res.trained.validation, python=platform.python_version(), seconds=round(time.time() - t0, 1),
                config=cfg.to_dict())
    (work / "validation_report.json").write_text(json.dumps(meta, indent=2, default=_json_default))
    print(res.report["headline"].to_string())
    print(f"[train] done in {time.time() - t0:.0f}s -> {work / 'model_bundle.pkl'}")
    return res.trained


def cmd_predict(args, trained: Optional[TrainedPipeline] = None):
    ds = resolve_dataset_dir(args.data_dir)
    trained = trained or TrainedPipeline.load(Path(args.work_dir) / "model_bundle.pkl")
    t0 = time.time()
    data = load_split(ds, "test")
    print(f"[predict] dataset={ds} sizes={data.summary()} policy={trained.policy.describe()}")
    res = predict(data, trained)
    _, _, errors, _ = write_outputs(data, res, args.out_dir, ds)
    if args.work_dir:
        res.scored.to_csv(Path(args.work_dir) / "test_scored_pairs.csv.gz", index=False)
    by_country = (pd.DataFrame({"country": data.s1["country"],
                                "has_match": data.s1["entity_id"].map(lambda s: bool(res.matches.get(s)))})
                  .groupby("country")["has_match"].agg(entities="size", share_with_match="mean"))
    print("[predict] share of S1 entities with a match, by country (sanity check for unseen countries):")
    print(by_country.to_string())
    print(f"[predict] done in {time.time() - t0:.0f}s")
    return 1 if errors else 0


def cmd_run(args):
    trained = cmd_train(args)
    return cmd_predict(args, trained)


def cmd_explore(args):
    cfg = _load_config(args.config)
    ds = resolve_dataset_dir(args.data_dir)
    data = load_split(ds, "train")
    tables = explore_split(data, cfg, args.analysis_dir)
    for name, t in tables.items():
        print(f"\n== {name} ==\n{t.to_string()}")


def cmd_validate(args):
    ds = resolve_dataset_dir(args.data_dir)
    out = Path(args.out_dir)
    errors, warnings = validate_submission(out / "matching_results.tsv", out / "candidate_pairs.tsv", ds / "test")
    print("PASS" if not errors else "FAIL\n" + "\n".join(errors))
    for w in warnings:
        print("warning:", w)
    code, msg = run_official_validator(ds.parent, out / "matching_results.tsv", out / "candidate_pairs.tsv", ds / "test")
    print("official validator:", msg)
    return 1 if errors else 0


def cmd_score(args):
    gt = load_ground_truth(args.truth)
    truth = truth_from_gt(gt, gt["source1_entity_id"])
    pred_df = read_tsv(args.pred)
    pred = {s: set(x for x in str(ids).split(",") if x) for s, ids in zip(pred_df.iloc[:, 0], pred_df.iloc[:, 1])}
    print(f"macro F0.5 = {macro_fbeta(pred, truth, list(truth)):.5f} over {len(truth)} Source-1 entities")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="run_pipeline.py", description="Business entity resolution pipeline")
    sub = ap.add_subparsers(dest="command", required=True)

    def common(p, out=False, work=False, analysis=False):
        p.add_argument("--data-dir", default=None, help="student_resource/ or student_resource/dataset/")
        p.add_argument("--config", default=None, help="optional JSON file with config overrides")
        if out:
            p.add_argument("--out-dir", default="output")
        if work:
            p.add_argument("--work-dir", default="artifacts")
        if analysis:
            p.add_argument("--analysis-dir", default="analysis")

    common(sub.add_parser("run", help="train + predict + write outputs + validate"), out=True, work=True)
    common(sub.add_parser("train", help="fit on the training split"), work=True)
    common(sub.add_parser("predict", help="score the test split with a saved model"), out=True, work=True)
    common(sub.add_parser("explore", help="write exploration tables"), analysis=True)
    common(sub.add_parser("validate", help="validate existing output files"), out=True)
    sc = sub.add_parser("score", help="macro F0.5 of a predictions file against a ground-truth file")
    sc.add_argument("--pred", required=True)
    sc.add_argument("--truth", required=True)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {"run": cmd_run, "train": lambda a: (cmd_train(a), 0)[1], "predict": cmd_predict,
                "explore": cmd_explore, "validate": cmd_validate, "score": cmd_score}
    rc = handlers[args.command](args)
    return int(rc or 0)


if __name__ == "__main__":
    sys.exit(main())
