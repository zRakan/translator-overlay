"""Speaker labels in Arabic: 'Michael:' would put Latin letters in the middle of right-to-left text, so a known name is written in Arabic
(data/speaker_names_ar.tsv). Unknown names stay as they are."""
import difflib
import os
import re
from functools import lru_cache

PATH = os.path.join(os.path.dirname(__file__), "data", "speaker_names_ar.tsv")


@lru_cache(maxsize=1)
def table() -> dict:
    out = {}
    with open(PATH, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or "\t" not in line:
                continue
            en, ar = line.rstrip("\n").split("\t", 1)
            if en.strip() and ar.strip():
                out[en.strip().lower()] = ar.strip()
    return out


def arabic_name(name: str):
    """'Lucia' -> 'لوسيا'; 'The Man' -> 'الرجل'; 'Guard 2' -> 'الحارس 2'; an OCR slip ('Luqia') -> the nearest listed name; unknown -> None."""
    t = table()
    low = re.sub(r"\s+", " ", name.strip().lower())
    for cand in (low, low.removeprefix("the ")):
        if cand in t:
            return t[cand]
    m = re.fullmatch(r"(.+?) (\d{1,3})", low)  # "Guard 2": the role, then its number
    if m and (m.group(1) in t or m.group(1).removeprefix("the ") in t):
        return f"{t.get(m.group(1)) or t[m.group(1).removeprefix('the ')]} {m.group(2)}"
    if len(low) >= 5:  # a one-letter OCR slip in a longer name
        close = difflib.get_close_matches(low, [k for k in t if abs(len(k) - len(low)) <= 1], n=1, cutoff=0.84)
        if close:
            return t[close[0]]
    return None


def is_listed(name: str) -> bool:
    """Exactly one of the listed speakers (no fuzzy matching)."""
    low = re.sub(r"\s+", " ", name.strip().lower())
    t = table()
    return low in t or low.removeprefix("the ") in t
