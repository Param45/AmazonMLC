"""Writing matching_results.tsv / candidate_pairs.tsv and validating them locally.

The files are written by hand (not with DataFrame.to_csv) so the format is exactly:
    header line, then  <S1 id> TAB <comma-separated ids, no spaces, no quotes>  per line
with an empty second field for entities without matches / candidates.
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

MATCH_HEADER = ("source1_entity_id", "matched_entity_ids")
CAND_HEADER = ("source1_entity_id", "candidate_entity_ids")


def _sort_ids(ids: Iterable[str]) -> List[str]:
    return sorted(set(ids), key=lambda x: (x[:3], len(x), x))


def write_id_lists(path: os.PathLike, header: Tuple[str, str], s1_ids: Iterable[str],
                   mapping: Dict[str, Iterable[str]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{header[0]}\t{header[1]}\n")
        for s in s1_ids:
            f.write(f"{s}\t{','.join(_sort_ids(mapping.get(s, ())))}\n")
    return path


def write_submission(out_dir: os.PathLike, s1_ids: List[str], matches: Dict[str, Set[str]],
                     candidates: Dict[str, Set[str]]) -> Tuple[Path, Path]:
    """Matches are forced to be a subset of candidates (a matched id outside the candidate list is a bug)."""
    for s, ids in matches.items():
        missing = set(ids) - set(candidates.get(s, ()))
        if missing:
            raise AssertionError(f"{s}: matched ids {sorted(missing)[:3]} are not in its candidate list")
    out_dir = Path(out_dir)
    m = write_id_lists(out_dir / "matching_results.tsv", MATCH_HEADER, s1_ids, matches)
    c = write_id_lists(out_dir / "candidate_pairs.tsv", CAND_HEADER, s1_ids, candidates)
    return m, c


def _read_rows(path: Path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE))


def _ids_from_source(path: Path) -> List[str]:
    rows = _read_rows(path)
    col = rows[0].index("entity_id")
    return [r[col].strip() for r in rows[1:] if r]


def validate_submission(matching: os.PathLike, candidate: os.PathLike, test_dir: os.PathLike):
    """Re-implements every rule from the problem statement. Returns (errors, warnings)."""
    test_dir = Path(test_dir)
    s1 = _ids_from_source(test_dir / "test_source1.tsv")
    targets = set(_ids_from_source(test_dir / "test_source2.tsv")) | set(_ids_from_source(test_dir / "test_source3.tsv"))
    errors: List[str] = []
    warnings: List[str] = []
    parsed = {}
    for path, header in ((Path(matching), MATCH_HEADER), (Path(candidate), CAND_HEADER)):
        name = path.name
        if not path.exists():
            errors.append(f"{name}: file not found")
            continue
        rows = _read_rows(path)
        if not rows or tuple(c.strip() for c in rows[0]) != header:
            errors.append(f"{name}: header must be {header[0]}<TAB>{header[1]}, got {rows[0] if rows else None}")
            continue
        seen, lists = set(), {}
        for ln, r in enumerate(rows[1:], start=2):
            if len(r) == 1:
                r = r + [""]
            if len(r) != 2:
                errors.append(f"{name}:{ln}: expected 2 tab-separated columns, got {len(r)}")
                continue
            sid, ids = r[0].strip(), r[1].strip()
            if sid in seen:
                errors.append(f"{name}:{ln}: duplicate source1_entity_id {sid}")
            seen.add(sid)
            items = [x for x in ids.split(",")] if ids else []
            if any(x != x.strip() or not x for x in items):
                errors.append(f"{name}:{ln}: empty id or whitespace inside the id list")
            items = [x.strip() for x in items if x.strip()]
            if len(items) != len(set(items)):
                errors.append(f"{name}:{ln}: duplicate ids in the list of {sid}")
            bad_prefix = [x for x in items if not x.startswith(("S2-", "S3-"))]
            if bad_prefix:
                errors.append(f"{name}:{ln}: non S2/S3 ids {bad_prefix[:3]}")
            unknown = [x for x in items if x not in targets and x not in bad_prefix]
            if unknown:
                errors.append(f"{name}:{ln}: ids not present in the test S2/S3 files {unknown[:3]}")
            lists[sid] = set(items)
        missing = set(s1) - seen
        extra = seen - set(s1)
        if missing:
            errors.append(f"{name}: {len(missing)} test Source-1 entities missing, e.g. {sorted(missing)[:3]}")
        if extra:
            errors.append(f"{name}: {len(extra)} unknown source1_entity_id values, e.g. {sorted(extra)[:3]}")
        parsed[name] = lists
    if len(parsed) == 2:
        mres, cres = parsed.values()
        outside = sum(len(v - cres.get(k, set())) for k, v in mres.items())
        if outside:
            warnings.append(f"{outside} matched ids do not appear in the candidate list of their entity")
    return errors, warnings


def run_official_validator(student_resource_dir: os.PathLike, matching: os.PathLike, candidate: os.PathLike,
                           test_dir: os.PathLike):
    """Run utils/validate_submission.py if the challenge's helper script is present."""
    script = Path(student_resource_dir) / "utils" / "validate_submission.py"
    if not script.exists():
        return None, f"official validator not found at {script}"
    proc = subprocess.run([sys.executable, str(script), "--matching", str(matching), "--candidate", str(candidate),
                           "--test-dir", str(test_dir)], capture_output=True, text=True)
    return proc.returncode, (proc.stdout + proc.stderr).strip()
