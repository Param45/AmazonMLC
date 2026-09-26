"""Country-agnostic text normalisation.

Design rules
------------
* One code path for every country. Nothing branches on the `country` value, so unseen labels
  (France in the test set) are processed exactly like US / India records.
* Accents are *folded* (é -> e, ç -> c) instead of deleted. A plain `[^a-z0-9]` filter would turn
  "Société" into "soci t" and silently wreck every French record.
* Abbreviations are mapped to one canonical form per concept, applied to both sides of a pair.
  Canonical forms are the *short* forms ("street", "str", "st" -> "st"), which also makes the
  English "St" (street) and French "St" (saint) collide harmlessly instead of mismatching.
* The lexicons below are generic language knowledge (legal forms, street types, honorifics)
  written by hand; no external database or API is consulted.
* Several views are kept per field (clean string, core tokens, phonetic skeleton, numbers, postal
  codes, tail tokens) because different features need different amounts of normalisation.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, List

import pandas as pd

# ----------------------------------------------------------------------------- basics
MISSING_VALUES = {"", "nan", "none", "null", "na", "n/a", "n.a", "n.a.", "-", "--", "?", "unknown",
                  "not available", "not applicable", "nil"}

_SPECIAL_CHARS = str.maketrans({
    "ß": "ss", "æ": "ae", "Æ": "ae", "œ": "oe", "Œ": "oe", "ø": "o", "Ø": "o", "đ": "d", "Đ": "d",
    "ł": "l", "Ł": "l", "ı": "i", "þ": "th", "’": "'", "‘": "'", "`": "'", "´": "'", "ʼ": "'",
    "–": "-", "—": "-", "‐": "-", "‑": "-", "＆": "&",
})


def is_missing(value) -> bool:
    return value is None or str(value).strip().lower() in MISSING_VALUES


def fold(text) -> str:
    """Lower-case, fold accents and compatibility characters; keep non-Latin letters intact."""
    if is_missing(text):
        return ""
    s = unicodedata.normalize("NFKD", str(text).translate(_SPECIAL_CHARS))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.lower()


_RE_ELISION = re.compile(r"\b(l|d|qu)'")                       # French elision: l'eglise -> l eglise
_RE_NON_WORD = re.compile(r"[\W_]+")
_RE_ALPHA_DIGIT = re.compile(r"(?<=[^\W\d_])(?=\d)|(?<=\d)(?=[^\W\d_])")
_RE_ORDINAL = re.compile(r"\b(\d+)(?:st|nd|rd|th|er|ere|eme|ieme|e)\b")
_RE_SPACED_PIN = re.compile(r"(?<!\d)(\d{3})[\s-]+(\d{3})\W*$")  # "560 034" at the very end -> "560034"
_RE_PAREN = re.compile(r"\(([^()]*)\)")
_RE_DBA = re.compile(
    r"\b(?:d ?/ ?b ?/ ?a|d b a|dba|doing business as|trading as|t ?/ ?a|a ?/ ?k ?/ ?a|aka|"
    r"f ?/ ?k ?/ ?a|fka|formerly known as|formerly)\b")


def merge_initials(tokens: List[str]) -> List[str]:
    """Join runs of single letters: 'l l c' -> 'llc', 'm g road' -> 'mg road', 's a r l' -> 'sarl'."""
    out: List[str] = []
    run: List[str] = []
    for t in tokens:
        if len(t) == 1 and t.isalpha():
            run.append(t)
            continue
        if run:
            out.append("".join(run))
            run = []
        out.append(t)
    if run:
        out.append("".join(run))
    return out


def tokenize(folded: str) -> List[str]:
    s = _RE_ELISION.sub(r"\1 ", folded)
    s = s.replace("'", "").replace("&", " and ")
    s = _RE_NON_WORD.sub(" ", s)
    s = _RE_ALPHA_DIGIT.sub(" ", s)
    return merge_initials(s.split())


def stem(token: str) -> str:
    """Very light plural folding, applied identically to both sides of a pair."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


_SKELETON_SUBS = [("ph", "f"), ("ck", "k"), ("q", "k"), ("x", "ks"), ("z", "s"), ("v", "w"),
                  ("c", "k"), ("ee", "i"), ("oo", "u")]


def skeleton(token: str) -> str:
    """Language-agnostic phonetic key: keep the first letter, drop later vowels/h/w/y, collapse repeats.

    agarwal / aggarwal / agrawal -> agrl ; lakshmi / laxmi -> lksm ; shree / shri / sri -> sr
    """
    if not token or token.isdigit():
        return token
    t = token
    for a, b in _SKELETON_SUBS:
        t = t.replace(a, b)
    head, rest = t[0], re.sub(r"[aeiouyhw]", "", t[1:])
    out = head
    for ch in rest:
        if ch != out[-1]:
            out += ch
    return out


# ----------------------------------------------------------------------------- names
# legal forms -> canonical token (only stripped when they appear in the trailing run of a name)
LEGAL_CANON: Dict[str, str] = {
    "inc": "inc", "incorporated": "inc", "incorp": "inc",
    "corp": "corp", "corporation": "corp", "corpn": "corp",
    "co": "co", "company": "co", "cos": "co", "cie": "co", "compagnie": "co",
    "ltd": "ltd", "limited": "ltd", "ltda": "ltd",
    "llc": "llc", "lc": "llc", "llp": "llp", "lp": "lp", "plc": "plc", "pllc": "pllc", "pc": "pc",
    "pvt": "pvt", "private": "pvt", "pte": "pvt", "prvt": "pvt", "p": "pvt", "opc": "opc",
    "gmbh": "gmbh", "ag": "ag", "kg": "kg", "bv": "bv", "nv": "nv", "srl": "srl", "spa": "spa",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa", "eurl": "eurl", "snc": "snc",
    "sci": "sci", "selarl": "selarl", "scp": "scp", "pty": "pty",
}
# generic business-word abbreviations -> canonical (pre-stemming) form
NAME_CANON: Dict[str, str] = {
    "intl": "international", "internatl": "international", "mfg": "manufacturing",
    "mfrs": "manufacturers", "mfr": "manufacturer", "svc": "services", "svcs": "services",
    "srvcs": "services", "servs": "services", "assoc": "associates", "assocs": "associates",
    "bros": "brothers", "natl": "national", "mgmt": "management", "mgt": "management",
    "dept": "department", "univ": "university", "hosp": "hospital", "ctr": "center",
    "centre": "center", "cntr": "center", "grp": "group", "sys": "systems", "tech": "technologies",
    "techs": "technologies", "technology": "technologies", "soln": "solutions",
    "solns": "solutions", "ent": "enterprises", "ents": "enterprises", "entp": "enterprises",
    "ind": "industries", "inds": "industries", "engg": "engineering", "engr": "engineering",
    "pharma": "pharmaceuticals", "pharmaceutical": "pharmaceuticals", "labs": "laboratories",
    "lab": "laboratories", "laboratory": "laboratories", "mktg": "marketing",
    "distr": "distributors", "distrib": "distributors", "constr": "construction",
    "const": "construction", "med": "medical", "jewelers": "jewellers", "jeweler": "jewellers",
    "jeweller": "jewellers", "shree": "sri", "shri": "sri", "sree": "sri", "shre": "sri",
    "restaurants": "restaurant", "rest": "restaurant", "hldgs": "holdings", "hldg": "holdings",
    "invts": "investments", "inv": "investments", "fin": "financial", "finl": "financial",
    "agcy": "agency", "ins": "insurance", "trdg": "trading", "trd": "trading", "bldrs": "builders",
}
NAME_STOP = {"the", "and", "of", "et", "de", "la", "le", "les", "du", "des", "d", "l", "a", "an",
             "und", "for", "at", "in", "on", "by"}
NAME_LEADING_STOP = {"ms", "messrs", "mr", "mrs", "the"}   # "M/s Sharma Traders" -> "sharma traders"


def _canon_name_tokens(tokens: List[str]) -> List[str]:
    return [LEGAL_CANON.get(t, NAME_CANON.get(t, t)) for t in tokens]


_LEGAL_VALUES = set(LEGAL_CANON.values())


def _core_and_legal(tokens: List[str]):
    toks = list(tokens)
    while toks and toks[0] in NAME_LEADING_STOP:
        toks = toks[1:]
    legal: List[str] = []
    while toks and (toks[-1] in _LEGAL_VALUES or toks[-1] in NAME_STOP):
        t = toks.pop()
        if t in _LEGAL_VALUES:
            legal.append(t)
    core = [stem(t) for t in toks if t not in NAME_STOP]
    if not core:  # the name was nothing but legal words / stop words: keep what we have
        core = [stem(t) for t in tokens if t not in NAME_STOP] or list(tokens)
    return core, legal


def normalize_name(raw) -> dict:
    s = fold(raw)
    empty = {"name_clean": "", "name_core": "", "name_tokens": [], "name_skel": [], "name_nospace": "",
             "name_acronym": "", "name_legal": "", "name_alts": [], "name_missing": True}
    if not s.strip():
        return empty
    # possessives are joined ("joe's" -> "joes"), French elisions are split ("l'atelier" -> "l atelier")
    s = _RE_ELISION.sub(r"\1 ", s).replace("'", "").replace("&", " and ")
    # alternates: multi-word parenthetical parts and DBA / trading-as splits
    parts: List[str] = []
    main = s
    for inner in _RE_PAREN.findall(s):
        if len(tokenize(inner)) >= 2:
            parts.append(inner)
            main = main.replace(f"({inner})", " ")
    main = main.replace("(", " ").replace(")", " ")
    main_norm = re.sub(r"[^\w/]+", " ", main)
    parts = [p for p in _RE_DBA.split(main_norm) if p.strip()] + parts
    if not parts:
        return empty

    all_clean: List[str] = []
    part_cores: List[List[str]] = []
    legal: List[str] = []
    for p in parts:
        toks = _canon_name_tokens(tokenize(p))
        if not toks:
            continue
        all_clean += toks
        core, leg = _core_and_legal(toks)
        part_cores.append(core)
        legal += leg
    if not part_cores:
        return empty
    core = [t for pc in part_cores for t in pc]
    alts = [" ".join(pc) for pc in part_cores] if len(part_cores) > 1 else []
    return {
        "name_clean": " ".join(all_clean),
        "name_core": " ".join(core),
        "name_tokens": core,
        "name_skel": [skeleton(t) for t in core],
        "name_nospace": "".join(core),
        "name_acronym": "".join(t[0] for t in core if t[0].isalpha()) if len(core) >= 2 else "",
        "name_legal": " ".join(sorted(set(legal))),
        "name_alts": alts,
        "name_missing": False,
    }


# ----------------------------------------------------------------------------- addresses
ADDR_CANON: Dict[str, str] = {
    "street": "st", "str": "st", "saint": "st", "suite": "ste", "sainte": "ste",
    "road": "rd", "marg": "rd", "avenue": "ave", "av": "ave", "aven": "ave",
    "boulevard": "blvd", "bd": "blvd", "bld": "blvd", "boul": "blvd", "bvd": "blvd",
    "drive": "dr", "lane": "ln", "court": "ct", "crt": "ct", "place": "pl", "plaza": "plz",
    "square": "sq", "highway": "hwy", "hiway": "hwy", "parkway": "pkwy", "circle": "cir",
    "floor": "fl", "flr": "fl", "building": "bldg", "bldng": "bldg", "apartment": "apt",
    "appt": "apt", "near": "nr", "opposite": "opp", "opst": "opp", "behind": "bhd", "beside": "bsd",
    "besides": "bsd", "adjacent": "adj", "north": "n", "south": "s", "east": "e", "west": "w",
    "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw", "sector": "sec",
    "phase": "ph", "district": "dist", "distt": "dist", "bazaar": "bazar", "chauk": "chowk",
    "ngr": "nagar", "col": "colony", "chemin": "ch", "allee": "all", "impasse": "imp",
    "route": "rte", "faubourg": "fbg", "mount": "mt", "fort": "ft", "center": "ctr",
    "centre": "ctr", "cntr": "ctr", "galli": "gali", "taluka": "taluk", "tq": "taluk",
    "number": "no", "num": "no", "nos": "no", "hno": "no", "r": "rue",
}
ADDR_DROP = {"cedex"}
ADDR_STOP = {"no", "plot", "shop", "house", "h", "flat", "door", "unit", "ste", "apt", "fl", "bldg",
             "the", "of", "and", "et", "de", "la", "le", "les", "du", "des", "d", "l", "at", "in",
             "on", "nr", "opp", "bhd", "bsd", "adj", "next", "to", "facing"}
LANDMARK_TOKENS = {"nr", "opp", "bhd", "bsd", "adj", "facing"}
_COMPOUND_SUFFIXES = ("nagar",)   # "shivajinagar" -> "shivaji nagar" (spacing variants)


def _canon_addr_tokens(tokens: List[str]) -> List[str]:
    out: List[str] = []
    for t in tokens:
        t = ADDR_CANON.get(t, t)
        if t in ADDR_DROP:
            continue
        for suf in _COMPOUND_SUFFIXES:
            if t.endswith(suf) and len(t) > len(suf) + 2 and not t.isdigit():
                out.append(t[: -len(suf)])
                t = suf
                break
        out.append(t)
    return out


def normalize_address(raw) -> dict:
    s = fold(raw)
    empty = {"addr_clean": "", "addr_core": "", "addr_tokens": [], "addr_numbers": [], "addr_postal": [],
             "addr_house": "", "addr_tail": [], "addr_landmark": False, "addr_missing": True}
    if not s.strip():
        return empty
    s = _RE_SPACED_PIN.sub(r"\1\2", s)
    s = _RE_ORDINAL.sub(r"\1", s)
    components = [c for c in re.split(r"[,;|\n]+", s) if c.strip()]
    comp_tokens = [_canon_addr_tokens(tokenize(c)) for c in components]
    tokens = [t for ct in comp_tokens for t in ct]
    if not tokens:
        return empty
    numbers = [t for t in tokens if t.isdigit()]
    postal = sorted({t for t in numbers if 5 <= len(t) <= 6})
    house = next((t for t in numbers if t not in postal), "")
    core = [t for t in tokens if t not in ADDR_STOP]
    tail: List[str] = []
    alpha_comps = [[t for t in ct if not t.isdigit() and t not in ADDR_STOP] for ct in comp_tokens]
    alpha_comps = [c for c in alpha_comps if c]
    for c in alpha_comps[-2:]:
        tail += c
    return {
        "addr_clean": " ".join(tokens),
        "addr_core": " ".join(core),
        "addr_tokens": core,
        "addr_numbers": numbers,
        "addr_postal": postal,
        "addr_house": house,
        "addr_tail": sorted(set(tail)),
        "addr_landmark": any(t in LANDMARK_TOKENS for t in tokens),
        "addr_missing": False,
    }


def normalize_country(raw) -> str:
    """Open-set label normalisation: case / accent / punctuation only. No alias table, no filtering."""
    return " ".join(merge_initials(_RE_NON_WORD.sub(" ", fold(raw)).split()))


def _normalize_series(fn, series: pd.Series, desc: str, log_interval: int = 250_000) -> List[dict]:
    n = len(series)
    out = []
    t0 = time.time()
    last_log = t0
    for i, x in enumerate(series):
        out.append(fn(x))
        now = time.time()
        if (i + 1) % log_interval == 0 or (i + 1) == n or (now - last_log >= 15.0):
            elapsed = now - t0
            speed = (i + 1) / max(elapsed, 0.001)
            pct = ((i + 1) / n) * 100
            eta = (n - (i + 1)) / max(speed, 1.0)
            print(f"[normalize] {desc}: {i+1:,}/{n:,} ({pct:.1f}%) | {speed:,.0f} rows/s | ETA: {eta:.0f}s", flush=True)
            last_log = now
    return out


# ----------------------------------------------------------------------------- tables
def normalize_records(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Return one row per record with every normalised view used downstream."""
    src_label = source if isinstance(source, str) else "mixed"
    names_list = _normalize_series(normalize_name, df["business_name"], f"{src_label} names")
    addrs_list = _normalize_series(normalize_address, df["business_address"], f"{src_label} addresses")
    names = pd.DataFrame(names_list, index=df.index)
    addrs = pd.DataFrame(addrs_list, index=df.index)
    out = pd.concat([df[["entity_id"]].copy(), names, addrs], axis=1)
    out["source"] = source if isinstance(source, str) else list(source)
    out["country_raw"] = df["country"].values
    out["country_norm"] = [normalize_country(c) for c in df["country"]]
    out["name_raw"] = df["business_name"].values
    out["addr_raw"] = df["business_address"].values
    return out.reset_index(drop=True)


def examples_table(values, kind: str = "name") -> pd.DataFrame:
    """Small helper for the notebook: show raw -> normalised views side by side."""
    fn = normalize_name if kind == "name" else normalize_address
    rows = []
    for v in values:
        r = fn(v)
        if kind == "name":
            rows.append({"raw": v, "clean": r["name_clean"], "core": r["name_core"], "legal": r["name_legal"],
                         "skeleton": " ".join(r["name_skel"]), "alternates": " | ".join(r["name_alts"])})
        else:
            rows.append({"raw": v, "core": r["addr_core"], "numbers": " ".join(r["addr_numbers"]),
                         "postal": " ".join(r["addr_postal"]), "house": r["addr_house"],
                         "tail": " ".join(r["addr_tail"]), "landmark": r["addr_landmark"]})
    return pd.DataFrame(rows)
