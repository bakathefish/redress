"""Cleanliness scan for the public tree: fail loudly on anything that should not ship.

Three checks, all fatal:

1. PROHIBITED TERMS in text files (case-insensitive). Internal project names, the author's
   local paths, review-tool names, and the vocabulary the model card is not allowed to use
   (a rank statistic is not a probability, a selection rate is not a precision).
2. PROHIBITED FILE TYPES anywhere in the tree: .pdf, .tex, .docx, .png, .jpg, .jpeg.
3. FILE SIZE: no file over 50 MB.

Every allowance is enumerated in ALLOWANCES below, applied by masking the allowed span out
of the text before the term scan runs, and COUNTED and PRINTED in the report. Nothing is
skipped silently.

Run from the repository root:  python tools/scan_public_tree.py
Exit status is non-zero if anything fails, so this can gate a commit.
"""

from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --------------------------------------------------------------- the term list
# (label, regex). Matched case-insensitively. Word-start anchors are used where the
# term is a name, so "ember" fires on "src/ember/" and on "EMBER_V4_R1" but not on
# "member" or "remembered".
TERMS: list[tuple[str, str]] = [
    ("internal project name", r"\bEMBER"),
    ("internal project name", r"ISEF@26"),
    ("local path", r"C:\\Users"),
    ("local path", r"/c/Users"),
    ("author's account name", r"\brudra"),
    ("build-tree path", r"scratch[/\\]"),
    ("forbidden vocabulary", r"\bp_lrd\b"),
    ("forbidden vocabulary", r"confirmed LRD candidate"),
    ("forbidden vocabulary", r"\bdiscover"),
    ("forbidden vocabulary", r"precision"),
    ("forbidden vocabulary", r"\bcensus_vshaped\b"),
    ("forbidden vocabulary", r"32 compact"),
    ("forbidden vocabulary", r"seven misses"),
]

# ---------------------------------------------------------------- allowances
# (id, description, file predicate, pattern marking an allowed span).
# A span matching the pattern is blanked out before the term scan, so terms inside it
# cannot fire. Every application is counted and printed in the report.
ALLOWANCES: list[tuple[str, str, str, str]] = [
    (
        "A1",
        'the negated disclaimers "not a precision" and "never precision"',
        r".*",
        r"(?:not a|never)\s+precision",
    ),
    (
        "A2",
        "the scikit-learn metric identifier average_precision_score",
        r".*",
        r"average_precision_score",
    ),
    (
        "A4",
        "the licence names its copyright holder",
        r"^LICENSE$",
        r"Rudra",
    ),
    (
        "A5",
        "the README names its author once",
        r"^README\.md$",
        r"Author: Rudra",
    ),
    (
        "A6",
        "the release script's own vocabulary guard, marked in place with # scan-allow",
        r"^recovery[\\/]v4_release\.py$",
        r"(?m)^.*#\s*scan-allow:.*$",
    ),
    (
        "A7",
        "this scanner carries the term list it enforces",
        r"^tools[\\/]scan_public_tree\.py$",
        r"(?s).*",
    ),
    # A9 exists because every table of record under recovery/ is released byte-identical to
    # the run that produced it. That file is data, not prose: rewriting a word inside a
    # provenance string would make the released table a different table from the one the
    # model card quotes. The term is allowed in that one file and nowhere else, and is
    # named here rather than covered by a blanket rule.
    (
        "A9",
        "recovery/feature_manifest_v34.json is a table of record, released "
        "byte-identical to the run, and its provenance string names the host the "
        "shape columns were extracted on",
        r"^recovery[\\/]feature_manifest_v34\.json$",
        r"\bember-emit-01",
    ),
]

TEXT_EXT = {
    ".py",
    ".md",
    ".json",
    ".csv",
    ".txt",
    ".ini",
    ".cfg",
    ".toml",
    ".yml",
    ".yaml",
}
TEXT_NAMES = {"LICENSE", ".gitattributes", ".gitignore"}
BAD_EXT = {".pdf", ".tex", ".docx", ".png", ".jpg", ".jpeg"}
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".hypothesis", ".venv"}
MAX_BYTES = 50 * 1024 * 1024


def walk(root: str):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for name in sorted(filenames):
            full = os.path.join(dirpath, name)
            yield os.path.relpath(full, root), full


def is_text(rel: str) -> bool:
    base = os.path.basename(rel)
    return base in TEXT_NAMES or os.path.splitext(base)[1].lower() in TEXT_EXT


def mask(rel: str, text: str, counts: dict[str, int]) -> str:
    """Blank out every allowed span, counting each application."""
    out = text
    norm = rel.replace("/", os.sep).replace("\\", os.sep)
    for aid, _desc, filepat, span in ALLOWANCES:
        if not re.search(filepat, norm):
            continue
        flags = 0 if span.startswith("(?") else re.I | re.M
        hits = list(re.finditer(span, out, flags))
        if not hits:
            continue
        counts[aid] = counts.get(aid, 0) + len(hits)
        chars = list(out)
        for m in hits:
            for i in range(m.start(), m.end()):
                if chars[i] != "\n":
                    chars[i] = " "
        out = "".join(chars)
    return out


def line_index(text: str) -> list[int]:
    line_of: list[int] = []
    line = 1
    for ch in text:
        line_of.append(line)
        if ch == "\n":
            line += 1
    line_of.append(line)
    return line_of


def main() -> int:
    failures: list[str] = []
    allow_counts: dict[str, int] = {}
    inventory: list[tuple[str, int]] = []
    total = 0

    for rel, full in walk(ROOT):
        size = os.path.getsize(full)
        inventory.append((rel, size))
        total += size

        ext = os.path.splitext(rel)[1].lower()
        if ext in BAD_EXT:
            failures.append(f"PROHIBITED FILE TYPE  {rel}  ({ext})")
        if size > MAX_BYTES:
            failures.append(
                f"FILE TOO LARGE        {rel}  ({size / 1e6:.1f} MB > 50 MB)"
            )
        if not is_text(rel):
            continue

        with open(full, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        scanned = mask(rel, text, allow_counts)
        line_of = line_index(scanned)

        for label, pat in TERMS:
            for m in re.finditer(pat, scanned, re.I):
                ln = line_of[m.start()]
                ctx = text[max(0, m.start() - 45) : m.end() + 45].replace("\n", " ")
                failures.append(
                    f"PROHIBITED TERM       {rel}:{ln}  [{label}] "
                    f"{scanned[m.start() : m.end()]!r}  ...{ctx}..."
                )

    print("=" * 78)
    print("FILE INVENTORY")
    print("=" * 78)
    for rel, size in inventory:
        print(f"{size:>12,}  {rel}")
    print("-" * 78)
    print(f"{total:>12,}  TOTAL over {len(inventory)} files")
    print()

    print("=" * 78)
    print("ALLOWANCES APPLIED")
    print("=" * 78)
    for aid, desc, filepat, _span in ALLOWANCES:
        n = allow_counts.get(aid, 0)
        scope = "any file" if filepat == r".*" else filepat
        print(f"{aid}  {n:>5} span(s)  [{scope}]  {desc}")
    print()

    print("=" * 78)
    print("RESULT")
    print("=" * 78)
    if failures:
        for f in failures:
            print(f)
        print()
        print(f"FAIL: {len(failures)} finding(s)")
        return 1
    print("PASS: no prohibited term, no prohibited file type, no file over 50 MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
