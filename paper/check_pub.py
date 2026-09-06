"""Checks on the publication rewrite (paper/main_flat.tex) against the deposited draft
(paper/main_reference_2026-09-04.tex).

  1. every macro used exists in numbers.tex or is defined in the preamble (a name the
     reference never used is listed so it can be looked at);
  2. numeric literals typed into the body are a subset of the literals the reference typed
     (a new one is a number that bypassed numbers.tex, or a caption constant to justify);
  3. no em dash (U+2014) and no "---" outside the keyword line;
  4. prohibited words (precision, purity, probability, confirmed candidate, discovered,
     discovery, p_lrd, new LRD) appear only in sentences that negate them;
  5. every \\ref and \\autoref target is defined, and every figure or table label is
     referenced in the prose at least once;
  6. citation keys of the reference that the rewrite no longer cites are listed (a
     dropped citation is a decision, not an accident);
  7. word count and numerals per 100 words for each section, against the literature
     calibration (13 to 17 numerals per 100 words in the bodies of the papers studied).

Run after flatten.py:

    python paper/flatten.py && python paper/check_pub.py
"""

from __future__ import annotations

import collections
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
AFTER = os.path.join(HERE, "main_flat.tex")
BEFORE = os.path.join(HERE, "reviews", "main_reference_2026-09-04.tex")
NUMBERS = os.path.join(HERE, "numbers.tex")

# literals typed into the prose that quote a published source rather than a table of
# record; each carries its source, and the director's record names the ruling
ALLOWED_LITERALS = {
    "60": "Pan et al. 2026 abstract: overall purity of about 60 per cent (round 5, A11)",
    "1.2": "Barro et al. 2024b Fig. 2 caption: stellar locus of the aperture ratio near 1.2 (eighth comparator, 2026-09-06)",
    "16": "Barro et al. 2026 Section 4.1: 19 of 20 Hviding LRDs, three added by hand (round 5, A2)",
    "19": "Barro et al. 2026 Section 4.1: 19 of the 20 Hviding LRDs in their catalogue (round 5, A2)",
    "20": "Barro et al. 2026 Section 4.1: the 20 Hviding LRDs in their catalogue (round 5, A2)",
    # JWST program identifiers and the STScI contract number quoted in the
    # acknowledgments; identifiers, not measurements (revision 2026-09-05)
    "03127": "NASA contract NAS 5-03127 (MAST acknowledgment text)",
    "1345": "CEERS, ERS 1345 (MAST)",
    "1837": "PRIMER, GO 1837 (MAST)",
    "1180": "JADES, GTO 1180 (MAST)",
    "1181": "JADES, GTO 1181 (MAST)",
    "1210": "JADES, GTO 1210 (MAST)",
    "1286": "JADES, GTO 1286 (MAST)",
    "3215": "JADES, GO 3215 (MAST)",
    "1895": "FRESCO, GO 1895 (MAST)",
    "1963": "JEMS, GO 1963 (MAST)",
    "2079": "NGDEEP, GO 2079 (MAST)",
    "2514": "PANORAMIC, GO 2514 (MAST)",
    "3577": "GO 3577 (MAST)",
    "4233": "RUBIES, GO 4233 (MAST)",
    "249": "Perez-Gonzalez et al. 2026 abstract (arXiv:2602.20247): 249 LRDs with NIRSpec prism spectra",
    # column widths of the scope table (tab:scope); layout constants, not measurements
    # (revision 2026-09-06, round 9)
    "5.2": "tab:scope column width in cm (layout)",
    "6.6": "tab:scope column width in cm (layout)",
    "4.6": "tab:scope column width in cm (layout)",
}

PROHIBITED = [
    "precision",
    "purity",
    "probabilit",
    "confirmed candidate",
    "discovered",
    "discovery",
    "p_lrd",
    "new lrd",
    "new little red dot",
]
NEGATIONS = (
    " not ",
    " no ",
    " none ",
    " never ",
    " nothing ",
    " neither ",
    " rather than ",
    " cannot ",
    " nor ",
)
LATEX_MACROS = {
    "documentclass",
    "usepackage",
    "makeatletter",
    "makeatother",
    "ifundefined",
    "def",
    "let",
    "relax",
    "unexpanded",
    "expandafter",
    "begin",
    "end",
    "input",
    "newcommand",
    "renewcommand",
    "setcounter",
    "setlength",
    "providecommand",
    "graphicspath",
    "url",
    "href",
    "nolinkurl",
    "title",
    "shorttitle",
    "shortauthors",
    "author",
    "affiliation",
    "email",
    "keywords",
    "section",
    "subsection",
    "paragraph",
    "label",
    "ref",
    "cite",
    "citep",
    "citet",
    "citealt",
    "citealp",
    "citeauthor",
    "citeyear",
    "item",
    "centering",
    "includegraphics",
    "caption",
    "textwidth",
    "columnwidth",
    "toprule",
    "midrule",
    "bottomrule",
    "cmidrule",
    "multicolumn",
    "raggedright",
    "arraybackslash",
    "footnotesize",
    "small",
    "scriptsize",
    "tabcolsep",
    "csname",
    "endcsname",
    "appendix",
    "FloatBarrier",
    "bibliographystyle",
    "bibliography",
    "facilities",
    "software",
    "textsc",
    "texttt",
    "textit",
    "emph",
    "mathrm",
    "sinh",
    "ln",
    "log",
    "tfrac",
    "frac",
    "chi",
    "beta",
    "sigma",
    "mu",
    "AA",
    "le",
    "ge",
    "approx",
    "sim",
    "times",
    "dagger",
    "rm",
    "hskip",
    "null",
    "par",
    "endgroup",
    "begingroup",
    "interlinepenalty",
    "hangfrom",
    "tempskipa",
    "z",
    "M",
    "xsect",
    "svsechd",
    "penalty",
    "ApjSectionpenalty",
    "MakeUppercase",
    "AddToNoCaseChangeList",
    "sec",
    "subsec",
    "startpbox",
    "endpbox",
    "ar",
    "do",
    "array",
    "cell",
    "tableftsep",
    "tabmidsep",
    "tabrightsep",
    "ssect",
    "mkpream",
    "ifdim",
    "fi",
    "else",
    "dodoi",
    "doeprint",
    "doarXiv",
    "bibinfo",
    "repourl",
    "lrd",
    "cff",
    "ctt",
    "noindent",
    "textbf",
    "vspace",
    "hspace",
    "newline",
    "enumerate",
    "itemize",
    "acknowledgments",
    "abstract",
    "document",
    "figure",
    "table",
    "tabular",
    "emergencystretch",
    "dbltopfraction",
    "dblfloatpagefraction",
    "floatpagefraction",
    "dbltopnumber",
    "gdef",
    "edef",
    "x",
    "the",
    "ifnum",
    "ifx",
    "ifcase",
    "or",
    "phantom",
    "big",
    "left",
    "right",
    "quad",
    "qquad",
    "text",
    "ldots",
    "dots",
    "cdot",
    "pm",
    "sqrt",
    "leq",
    "geq",
    "neq",
    "infty",
    "star",
    "prime",
    "hat",
    "bar",
    "vec",
}


def body_lines(text: str) -> str:
    return "\n".join(l for l in text.split("\n") if not l.lstrip().startswith("%"))


def strip_comments(text: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", text)


def number_macros() -> set:
    with open(NUMBERS, encoding="utf-8") as f:
        t = f.read()
    return set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", t))


def preamble_macros(text: str) -> set:
    return set(re.findall(r"\\(?:new|renew|provide)command\{\\([A-Za-z]+)\}", text))


def macros(text: str) -> collections.Counter:
    return collections.Counter(re.findall(r"\\([A-Za-z]+)", strip_comments(text)))


def numbers(text: str) -> dict:
    out = {}
    for line in strip_comments(text).split("\n"):
        # citation keys, labels, cross-references, verbatim file names and commit
        # hashes carry digits that are not quantities; drop their arguments first
        line = re.sub(
            r"\\(?:cite[a-zA-Z]*\*?|label|ref|eqref|texttt|url|href|bibliography)"
            r"(?:\[[^\]]*\])*\{[^}]*\}",
            " ",
            line,
        )
        stripped = re.sub(r"\\[A-Za-z@]+", " ", line)
        for m in re.finditer(r"\d[\d,\.]*", stripped):
            lit = m.group(0).rstrip(".,")
            if lit:
                out.setdefault(
                    lit, stripped[max(0, m.start() - 45) : m.end() + 30].strip()
                )
    return out


def document_body(text: str) -> str:
    m = re.search(r"\\begin\{document\}(.*)\\end\{document\}", text, re.S)
    return m.group(1) if m else text


def sentences(text: str) -> list:
    body = strip_comments(document_body(text))
    body = re.sub(r"\s+", " ", body)
    return [s.strip() for s in re.split(r"(?<=[.!?]) +", body) if s.strip()]


def norm(s: str) -> str:
    return " " + re.sub(r"[^a-z_ ]", " ", s.lower()) + " "


def cite_keys(text: str) -> collections.Counter:
    out = collections.Counter()
    for m in re.finditer(
        r"\\cite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}", strip_comments(text)
    ):
        for k in m.group(1).split(","):
            k = k.strip()
            if k:
                out[k] += 1
    return out


def labels_and_refs(text: str):
    t = strip_comments(text)
    labels = re.findall(r"\\label\{([^}]*)\}", t)
    refs = []
    for m in re.finditer(r"\\(?:auto)?ref\{([^}]*)\}", t):
        refs.extend(k.strip() for k in m.group(1).split(","))
    return labels, refs


def drop_floats_and_captions(text: str) -> str:
    """Prose only: remove figure/table environments and captions, for the density and the
    label-in-prose checks."""
    t = strip_comments(text)
    t = re.sub(r"\\begin\{(figure\*?|table\*?)\}.*?\\end\{\1\}", " ", t, flags=re.S)
    return t


def split_sections(text: str) -> list:
    body = drop_floats_and_captions(document_body(text))
    parts = re.split(r"(\\section\*?\{[^}]*\})", body)
    out = []
    head = "front matter (abstract)"
    buf = parts[0]
    for i in range(1, len(parts), 2):
        out.append((head, buf))
        head = re.sub(r"\\section\*?\{|\}$", "", parts[i])
        buf = parts[i + 1] if i + 1 < len(parts) else ""
    out.append((head, buf))
    return out


def words(s: str) -> int:
    s = re.sub(r"\\[A-Za-z]+\{?", " ", s)
    return len(re.findall(r"[A-Za-z][A-Za-z\-']+", s))


def numerals(s: str) -> int:
    # a macro use counts as one numeral, a typed number as one numeral
    n = len(re.findall(r"\\[a-zA-Z]+\{\}", s))
    s2 = re.sub(r"\\[A-Za-z]+\{?\}?", " ", s)
    n += len(re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?", s2))
    return n


def main() -> int:
    with open(BEFORE, encoding="utf-8") as f:
        before = f.read()
    with open(AFTER, encoding="utf-8") as f:
        after = f.read()
    fails = 0
    # A18 (round 5): the flat file under review must be the flattening of main.tex
    sys.path.insert(0, HERE)
    from flatten import MAIN as _MAIN, flatten as _flatten

    if _flatten(_MAIN) != after:
        print("0. FAIL main_flat.tex is not the flattening of main.tex; run flatten.py")
        fails += 1
    else:
        print("0. flat file matches the section files")

    print("1. macros")
    known = number_macros() | preamble_macros(after) | LATEX_MACROS
    used = macros(after)
    unknown = sorted(k for k in used if k not in known and not k.startswith("@"))
    if unknown:
        print("   LOOK: names not in numbers.tex or the preamble (LaTeX or a typo):")
        print("   " + ", ".join(unknown))
    ref_used = macros(before)
    new_names = sorted(k for k in used if k in number_macros() and k not in ref_used)
    print(
        "   number macros used: %d distinct; new relative to the reference: %s"
        % (sum(1 for k in used if k in number_macros()), new_names or "none")
    )
    unused = sorted(k for k in number_macros() if k not in used and k in ref_used)
    print(
        "   number macros the reference used and the rewrite does not: %d" % len(unused)
    )

    print("2. numeric literals typed in the body")
    nb, na = numbers(document_body(before)), numbers(document_body(after))
    extra = sorted(set(na) - set(nb) - set(ALLOWED_LITERALS))
    for k in sorted(set(na) & set(ALLOWED_LITERALS)):
        print("   allowed literal %r: %s" % (k, ALLOWED_LITERALS[k]))
    if extra:
        print("   FAIL new literals (%d):" % len(extra))
        for k in extra:
            print("      %r in: %s" % (k, na[k]))
        fails += 1
    else:
        print(
            "   PASS subset of the reference's literals (%d before, %d after)"
            % (len(nb), len(na))
        )

    print("3. em dashes")
    bad = []
    for i, line in enumerate(after.split("\n"), 1):
        if line.lstrip().startswith("%") or "\\keywords" in line:
            continue
        if "\u2014" in line or "---" in line:
            bad.append((i, line.strip()[:90]))
    if bad:
        print("   FAIL %d lines" % len(bad))
        for i, l in bad:
            print("      line %d: %s" % (i, l))
        fails += 1
    else:
        print("   PASS")

    print("4. prohibited vocabulary")
    ok = True
    for s in sentences(after):
        ns = norm(s)
        for w in PROHIBITED:
            if w in ns:
                if any(n in ns for n in NEGATIONS):
                    continue
                print("      FAIL %r without a negation: %s" % (w, s[:140]))
                ok = False
    print(
        "   PASS every occurrence sits in a negated sentence" if ok else "   see above"
    )
    fails += 0 if ok else 1

    print("5. labels and references")
    labels, refs = labels_and_refs(after)
    dup = [k for k, v in collections.Counter(labels).items() if v > 1]
    undefined = sorted(set(refs) - set(labels))
    if dup:
        print("   FAIL duplicate labels: %s" % dup)
        fails += 1
    if undefined:
        print("   FAIL undefined references: %s" % undefined)
        fails += 1
    prose = drop_floats_and_captions(document_body(after))
    prose_refs = set()
    for m in re.finditer(r"\\ref\{([^}]*)\}", prose):
        prose_refs.update(k.strip() for k in m.group(1).split(","))
    floats = [l for l in labels if l.startswith(("fig:", "tab:"))]
    unref = [l for l in floats if l not in prose_refs]
    if unref:
        print("   FAIL floats never referenced from the prose: %s" % unref)
        fails += 1
    if not dup and not undefined and not unref:
        print(
            "   PASS %d labels, %d refs, every float referenced in prose"
            % (len(labels), len(refs))
        )
    # order of first reference against order of definition
    first = {}
    for m in re.finditer(r"\\ref\{([^}]*)\}", strip_comments(document_body(after))):
        for k in m.group(1).split(","):
            k = k.strip()
            first.setdefault(k, m.start())
    for kind in ("fig:", "tab:"):
        defined = [l for l in labels if l.startswith(kind)]
        by_first = sorted(defined, key=lambda l: first.get(l, 10**9))
        if by_first != defined:
            print(
                "   LOOK: %s definition order differs from first-reference order" % kind
            )
            print("      defined : %s" % defined)
            print("      by ref  : %s" % by_first)

    print("6. citations")
    cb, ca = cite_keys(before), cite_keys(after)
    dropped = sorted(k for k in cb if k not in ca)
    added = sorted(k for k in ca if k not in cb)
    print(
        "   keys: %d in the reference, %d in the rewrite; dropped %s; added %s"
        % (len(cb), len(ca), dropped or "none", added or "none")
    )

    print("7. length and density by section (prose only, floats and captions removed)")
    total_w = total_n = 0
    for head, body in split_sections(after):
        w, n = words(body), numerals(body)
        total_w += w
        total_n += n
        if w:
            print(
                "   %-58s %6d words  %5.1f numerals per 100 words"
                % (head[:58], w, 100.0 * n / w)
            )
    print(
        "   %-58s %6d words  %5.1f numerals per 100 words"
        % ("TOTAL", total_w, 100.0 * total_n / max(total_w, 1))
    )
    ref_w = sum(words(b) for _, b in split_sections(before))
    print("   reference draft prose: %d words" % ref_w)

    print()
    print("CHECK", "OK" if fails == 0 else "FAILED (%d)" % fails)
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
