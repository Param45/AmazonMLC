"""Loading the challenge TSV files and parsing / checking the ground truth."""
from __future__ import annotations

import csv
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import pandas as pd

RECORD_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GT_COLUMNS = ["source1_entity_id", "matched_entity_ids"]


def read_tsv(path: os.PathLike) -> pd.DataFrame:
    """Read a challenge TSV robustly.

    * explicit tab separator (commas live inside addresses and ID lists)
    * everything as str, and "NA"/"null" are NOT turned into NaN (a business may be called "NA Traders")
    * QUOTE_NONE: a stray double quote inside a name can never swallow the following lines
    """
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=[],
                     quoting=csv.QUOTE_NONE, encoding="utf-8")
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    with open(path, "r", encoding="utf-8") as f:
        n_lines = sum(1 for line in f if line.strip())
    if n_lines - 1 != len(df):
        raise ValueError(f"{path}: parsed {len(df)} rows but the file has {n_lines - 1} non-empty data lines")
    return df


def load_records(path: os.PathLike, prefix: str) -> pd.DataFrame:
    df = read_tsv(path)
    missing = [c for c in RECORD_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}; found {list(df.columns)}")
    df = df[RECORD_COLUMNS].copy()
    for c in RECORD_COLUMNS:
        df[c] = df[c].fillna("").astype(str).str.strip()
    if df["entity_id"].duplicated().any():
        dups = df.loc[df["entity_id"].duplicated(), "entity_id"].head().tolist()
        raise ValueError(f"{path}: duplicated entity_id values, e.g. {dups}")
    bad = ~df["entity_id"].str.startswith(prefix)
    if bad.any():
        raise ValueError(f"{path}: {int(bad.sum())} ids do not start with {prefix!r}")
    return df.reset_index(drop=True)


def load_ground_truth(path: os.PathLike) -> pd.DataFrame:
    gt = read_tsv(path)
    missing = [c for c in GT_COLUMNS if c not in gt.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    gt = gt[GT_COLUMNS].copy()
    for c in GT_COLUMNS:
        gt[c] = gt[c].fillna("").astype(str).str.strip()
    return gt


def split_ids(value: str) -> List[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def truth_from_gt(gt: pd.DataFrame, s1_ids: Iterable[str]) -> Dict[str, Set[str]]:
    """source1_entity_id -> set of matched S2/S3 ids. Every S1 id gets an entry (empty = singleton)."""
    truth: Dict[str, Set[str]] = {sid: set() for sid in s1_ids}
    for s1_id, ids in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        truth.setdefault(s1_id, set()).update(split_ids(ids))
    return truth


def positive_pairs(truth: Dict[str, Set[str]]) -> pd.DataFrame:
    rows = [(s, t) for s, ts in truth.items() for t in ts]
    return pd.DataFrame(rows, columns=["s1_id", "t_id"])


@dataclass
class SplitData:
    name: str
    s1: pd.DataFrame
    s2: pd.DataFrame
    s3: pd.DataFrame
    gt: Optional[pd.DataFrame] = None
    truth: Optional[Dict[str, Set[str]]] = None

    @property
    def targets(self) -> pd.DataFrame:
        """S2 and S3 stacked (S2 first). `source` is derived from the id prefix."""
        t = pd.concat([self.s2.assign(source="S2"), self.s3.assign(source="S3")], ignore_index=True)
        return t

    def summary(self) -> Dict[str, int]:
        out = {"S1": len(self.s1), "S2": len(self.s2), "S3": len(self.s3)}
        if self.truth is not None:
            out["positive_pairs"] = sum(len(v) for v in self.truth.values())
            out["singletons"] = sum(1 for v in self.truth.values() if not v)
        return out


def resolve_dataset_dir(path: Optional[os.PathLike] = None) -> Path:
    """Accept `student_resource/`, `student_resource/dataset/` or `BER_DATA_DIR`; return the dataset dir."""
    candidates = []
    if path is not None:
        candidates.append(Path(path))
    if os.environ.get("BER_DATA_DIR"):
        candidates.append(Path(os.environ["BER_DATA_DIR"]))
    here = Path.cwd()
    for base in [here, *here.parents][:5]:
        candidates += [base / "dataset", base / "student_resource" / "dataset"]
    # Auto-detect Kaggle input directory if running on Kaggle
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for p in kaggle_input.rglob("train_source1.tsv"):
            candidates.append(p.parent.parent)
            break
    for c in candidates:
        for d in (c, c / "dataset"):
            if (d / "train" / "train_source1.tsv").exists() or (d / "test" / "test_source1.tsv").exists():
                return d.resolve()
    raise FileNotFoundError("Could not find the dataset folder (expected dataset/train/train_source1.tsv). "
                            "Pass it explicitly or set the BER_DATA_DIR environment variable.")


def load_split(dataset_dir: os.PathLike, split: str) -> SplitData:
    base = Path(dataset_dir) / split
    s1 = load_records(base / f"{split}_source1.tsv", "S1-")
    s2 = load_records(base / f"{split}_source2.tsv", "S2-")
    s3 = load_records(base / f"{split}_source3.tsv", "S3-")
    gt = truth = None
    gt_path = base / f"{split}_ground_truth.tsv"
    if gt_path.exists():
        gt = load_ground_truth(gt_path)
        truth = truth_from_gt(gt, s1["entity_id"])
    return SplitData(split, s1, s2, s3, gt, truth)


def check_ground_truth(data: SplitData) -> pd.DataFrame:
    """Data-contract checks on the ground truth. Returns a table of (check, value, ok)."""
    assert data.gt is not None and data.truth is not None
    gt, s1_ids = data.gt, set(data.s1["entity_id"])
    t_ids = set(data.s2["entity_id"]) | set(data.s3["entity_id"])
    all_matched = [t for ids in gt["matched_entity_ids"] for t in split_ids(ids)]
    usage = defaultdict(set)
    for s, ts in data.truth.items():
        for t in ts:
            usage[t].add(s)
    dup_within = sum(len(split_ids(v)) != len(set(split_ids(v))) for v in gt["matched_entity_ids"])
    rows = [
        ("S1 ids missing from ground truth", len(s1_ids - set(gt["source1_entity_id"])), True),
        ("ground-truth rows with unknown S1 id", int((~gt["source1_entity_id"].isin(s1_ids)).sum()), True),
        ("duplicated S1 rows in ground truth", int(gt["source1_entity_id"].duplicated().sum()), True),
        ("matched ids not found in S2/S3", sum(t not in t_ids for t in all_matched), True),
        ("self matches / S1 ids in matched lists", sum(t.startswith("S1-") for t in all_matched), True),
        ("rows with duplicated ids inside the list", dup_within, True),
        ("targets linked to >1 S1 entity", sum(len(v) > 1 for v in usage.values()), None),
    ]
    out = pd.DataFrame(rows, columns=["check", "value", "must_be_zero"])
    out["ok"] = [(v == 0) if must else True for v, must in zip(out["value"], out["must_be_zero"])]
    return out.drop(columns="must_be_zero")
