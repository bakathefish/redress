"""The three appendix tables of the recovery paper, written from the tables of record in
recovery to paper/tables/*.tex.

Each output file holds only the BODY of a table: one line per row, cells separated by "&",
every line ending in "\\\\". The paper supplies the \\begin{tabular} wrapper, the column
specification and the rules, so a column can be re-styled in the tex without touching this
script, and a number can be audited by diffing this file's output against recovery.

Three tables:
  recovered_rule_missed.tex  the rule-missed positives the classifier recovers
  still_missed.tex           the positives still missed after recovery
  followup_top.tex           the top of the follow-up tier of the candidate list

Run from the repository root: python paper/build_tables.py
"""

import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V = os.path.join(ROOT, "recovery")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables")
os.chdir(ROOT)

# Columns joined in from the feature table of record, keyed on (field, id). Round 3,
# Reviewer A: the two appendix tables were set across both columns only to carry their
# coordinates, at the cost of a nearly empty page; the coordinates of every labelled object
# are released in label_audit.csv, so those two tables no longer print RA and Dec and no
# longer join them. The follow-up table of the main text keeps its coordinates and reads
# them from candidates.csv, not from this join.
JOIN_COLS = ["c_f277w_f444w", "z_phot"]

# The archive status as printed in the follow-up table. The catalogue wording is too wide for
# the printed column, so it is set in the paper's notation here and nowhere else; the
# catalogue itself is unchanged. These strings are already LaTeX and are not escaped again.
STATUS_TEX = {
    "untested": "untested",
    "secure z>3 (V untested)": r"secure $z>3$",
    "secure low-z": r"secure $z\le 3$",
}


def tex_escape(s):
    """Escape the LaTeX specials that can appear in the catalogue strings this script
    writes: underscores in field names, and the comparison signs in the archive status
    text ("secure z>3 (V untested)"), which are only safe in text mode under T1."""
    s = str(s)
    for a, b in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("<", r"$<$"),
        (">", r"$>$"),
    ):
        s = s.replace(a, b)
    return s


def yesno(v):
    return "yes" if bool(v) else "no"


def num(fmt, v):
    """Format a number, setting a leading minus as a true minus sign.

    The hyphen-minus a %f produces prints in text mode as a hyphen, which is shorter and
    sits higher than the minus of the surrounding mathematics. Every numeric column that
    can go negative is written through this (round 4, Reviewer D minor 6): the declination
    of the follow-up table and the F277W-F444W colour of all three. Columns that cannot go
    negative pass through unchanged, so the change is visible only where a value is
    actually negative."""
    s = fmt % float(v)
    return "$-$" + s[1:] if s.startswith("-") else s


def write_rows(name, rows):
    """Write one table body and return its row count."""
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(" & ".join(r) + " \\\\\n")
    print("wrote %-32s %3d rows" % (name, len(rows)))
    return len(rows)


def load_features():
    return pd.read_parquet(
        os.path.join(V, "features.parquet"), columns=["field", "id"] + JOIN_COLS
    )


def join(df, feat):
    """Attach the colour and the photo-z from the feature table. Every (field, id) in the
    evaluation lists must be present exactly once, so a silent miss cannot reach the paper."""
    n = len(df)
    m = df.merge(feat, on=["field", "id"], how="left", validate="one_to_one")
    assert len(m) == n, "join changed the row count: %d -> %d" % (n, len(m))
    miss = m[JOIN_COLS].isna().any(axis=1)
    assert not miss.any(), "unjoined rows: %s" % m.loc[miss, ["field", "id"]].to_dict(
        "records"
    )
    return m


def table_recovered(ev, feat):
    """The rule-missed positives the classifier recovers, richest score first.

    Round 3, Reviewer A: RA and Dec were dropped so the table sets in one column; the
    coordinates of these objects are in the released label_audit.csv."""
    df = join(pd.DataFrame(ev["recovered_rule_missed_list"]), feat)
    df = df.sort_values("score", ascending=False)
    rows = []
    for _, r in df.iterrows():
        lists = "".join(
            code
            for code, flag in (
                ("H", r["in_hviding25_A1"]),
                ("B", r["in_barro25"]),
                ("G", r["in_degraaff26"]),
            )
            if bool(flag)
        )
        rows.append(
            [
                tex_escape(r["field"]),
                str(int(r["id"])),
                num("%.2f", r["mag_f444w"]),
                num("%.2f", r["c_f277w_f444w"]),
                num("%.2f", r["z_phot"]),
                lists,
                num("%.3f", r["score"]),
            ]
        )
    return rows


def table_still_missed(ev, feat):
    """The positives still missed after recovery, richest score first.

    Round 3, Reviewer A: RA and Dec were dropped so the table sets in one column; the
    coordinates of these objects are in the released label_audit.csv."""
    df = join(pd.DataFrame(ev["still_missed_list"]), feat)
    df = df.sort_values("score", ascending=False)
    rows = []
    for _, r in df.iterrows():
        rows.append(
            [
                tex_escape(r["field"]),
                str(int(r["id"])),
                num("%.2f", r["mag_f444w"]),
                num("%.2f", r["c_f277w_f444w"]),
                num("%.2f", r["z_phot"]),
                yesno(r["picked"]),
                num("%.3f", r["score"]),
                # score_rank_pct is already 100 x the fraction of the catalogue scoring at
                # least as high (recovery/v4_evaluate.py), so it is printed as it stands.
                num("%.2f", r["score_rank_pct"]),
            ]
        )
    return rows


def table_followup(n=20):
    """The head of the follow-up tier of the candidate list, best rank first.

    Round 2, Reviewer A finding 9: a catalogue photometric redshift above 10 is a
    catastrophic template failure, and the table's second row carries 17.91 with nothing to
    mark it, so those values are daggered and the caption says what the dagger means. The
    compact column was dropped in the same round (Reviewer D minor 5): it is "yes" in every
    row by the tier's own definition, so it carries no information and costs width."""
    c = pd.read_csv(os.path.join(V, "candidates.csv"))
    ft = c[c["followup_tier"].astype(bool)].sort_values("rank").head(n)
    unknown = sorted(set(ft["status"]) - set(STATUS_TEX))
    assert not unknown, "unmapped archive status: %s" % unknown
    assert ft["compact_labbe"].astype(bool).all(), (
        "a follow-up-tier row is not compact, so the dropped column did carry information"
    )
    rows = []
    for _, r in ft.iterrows():
        zp = num("%.2f", r["z_phot"])
        if float(r["z_phot"]) > 10:
            zp += r"$^{\dagger}$"
        rows.append(
            [
                str(int(r["rank"])),
                tex_escape(r["field"]),
                str(int(r["id"])),
                num("%.5f", r["ra"]),
                num("%.5f", r["dec"]),
                num("%.2f", r["mag_f444w"]),
                num("%.2f", r["c_f277w_f444w"]),
                zp,
                num("%.3f", r["ranking_score"]),
                STATUS_TEX[r["status"]],
            ]
        )
    return rows


def main():
    ev = json.load(open(os.path.join(V, "evaluation.json"), encoding="utf-8"))
    feat = load_features()

    n1 = write_rows("recovered_rule_missed.tex", table_recovered(ev, feat))
    n2 = write_rows("still_missed.tex", table_still_missed(ev, feat))
    n3 = write_rows("followup_top.tex", table_followup())

    for got, want, what in (
        (n1, 20, "recovered_rule_missed.tex"),
        (n2, 24, "still_missed.tex"),
        (n3, 20, "followup_top.tex"),
    ):
        assert got == want, "%s: %d rows, expected %d" % (what, got, want)
    print("row counts: %d + %d + %d = %d" % (n1, n2, n3, n1 + n2 + n3))


if __name__ == "__main__":
    main()
