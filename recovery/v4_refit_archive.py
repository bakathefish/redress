"""v4 build step 9: the archival test of every candidate that has an eligible spectrum.

Eligible record = DJA v4.4 index row within 0.5 arcsec, PRISM, grade >= 3, z_best > 3 (the
paper's mask, verbatim). Each eligible spectrum is fetched from the public DJA extraction
store and fitted with the FROZEN engine of record (selections/spectra/census3.py
+ slopefit.py, unmodified): guarded linear-space power laws on both sides of the Balmer
break with one-sided delta-chi-squared = 4 bounds, the V test written on the bounds.
Per-object verdict, conservative in the same direction as the paper's pass-2 rule:
  consistent    at least one eligible spectrum passes the frozen V test (c_uv & c_opt & c_diff)
  disfavoured   no spectrum passes and at least one testable spectrum fails even with the
                V-favourable bounds: beta_uv_lo2 > -0.2 or beta_opt_hi2 < 0 or
                (beta_opt_hi2 - beta_uv_lo2) < 0.5
  inconclusive  testable spectra exist but neither of the above
  untestable    every eligible spectrum was refused by the guards (or failed to download)
Nothing here feeds back into the model. Usage: python v4_refit_archive.py [suffix]
Outputs: archive_refits{SUF}.csv (one row per spectrum), candidates{SUF}.csv updated in place
(archive_verdict, status), archive_refit_counts{SUF}.json
"""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = "recovery"
SUF = sys.argv[1] if len(sys.argv) > 1 else ""
ENGINE = os.path.join(
    "selections", "spectra"
)  # the frozen refit engine
sys.path.insert(0, ENGINE)
import census3  # noqa: E402
from slopefit import fit_powerlaw  # noqa: E402

SPEC = os.path.join(OUT, "archive_spectra")
os.makedirs(SPEC, exist_ok=True)
census3.OUT = SPEC
BASE = "https://s3.amazonaws.com/msaexp-nirspec/extractions"

cand = pd.read_csv(os.path.join(OUT, f"candidates{SUF}.csv"))
rec = pd.read_csv(os.path.join(OUT, f"candidate_archive_records{SUF}.csv"))
todo = rec[rec.eligible.astype(bool)].drop_duplicates(["cand_rank", "file"]).copy()
print("candidates:", len(cand), "| eligible spectra to refit:", len(todo), flush=True)


class Row:
    def __init__(self, file, root):
        self.file, self.url = file, f"{BASE}/{root}/{file}"


with ThreadPoolExecutor(max_workers=8) as pool:
    paths = list(
        pool.map(census3.fetch, [Row(r.file, r.root) for r in todo.itertuples()])
    )


def both_bounds(path, z):
    """The two one-sided bounds the pass-2 rule needs, from the frozen fitter, unchanged."""
    from astropy.io import fits

    with fits.open(path, memmap=False) as h:
        d = h[1].data
        w_um = np.asarray(d["wave"], float)
        fl = np.asarray(d["flux"], float)
        er = np.asarray(d["err"], float)
    lam = 1e4 * w_um / (1.0 + z)
    flam, elam = fl / w_um**2, er / w_um**2
    km = census3.line_mask(lam)
    uv = fit_powerlaw(lam, flam, elam, census3.UV_LO, census3.BREAK, mask=km)
    op = fit_powerlaw(lam, flam, elam, census3.BREAK, census3.OPT_HI, mask=km)
    return uv, op


rows = []
for r, p in zip(todo.itertuples(), paths):
    base = dict(
        cand_rank=int(r.cand_rank),
        file=r.file,
        root=r.root,
        grating=r.grating,
        grade=float(r.grade),
        z_best=float(r.z_best),
        sep_arcsec=float(r.sep_arcsec),
    )
    if p is None:
        rows.append(
            dict(
                base,
                ok=False,
                testable=False,
                note="download failed",
                spectrum_verdict="untestable",
            )
        )
        continue
    c = census3.classify(p, float(r.z_best))
    out = dict(base)
    for k, v in c.items():
        out[k] = (
            None
            if (isinstance(v, float) and not np.isfinite(v))
            else (bool(v) if isinstance(v, (bool, np.bool_)) else v)
        )
    if c.get("testable"):
        uv, op = both_bounds(p, float(r.z_best))
        out["beta_uv_lo2"] = float(uv["beta_lo2"])
        out["beta_opt_hi2"] = float(op["beta_hi2"])
        if c.get("v_shaped"):
            out["spectrum_verdict"] = "consistent"
        elif (
            (uv["beta_lo2"] > -0.2)
            or (op["beta_hi2"] < 0)
            or ((op["beta_hi2"] - uv["beta_lo2"]) < 0.5)
        ):
            out["spectrum_verdict"] = "disfavoured"
        else:
            out["spectrum_verdict"] = "inconclusive"
    else:
        out["spectrum_verdict"] = "untestable"
    rows.append(out)
    print(
        f"  rank {base['cand_rank']:4d} {r.file}: {out['spectrum_verdict']} (uv {out.get('beta_uv')}, opt {out.get('beta_opt')})",
        flush=True,
    )
ref = pd.DataFrame(rows)
if ref.empty:
    ref = pd.DataFrame(columns=["cand_rank", "file", "root", "grating", "grade", "z_best", "sep_arcsec", "spectrum_verdict"])
ref.to_csv(os.path.join(OUT, f"archive_refits{SUF}.csv"), index=False)

order = {"consistent": 0, "inconclusive": 1, "disfavoured": 2, "untestable": 3}


def object_verdict(g):
    v = set(g.spectrum_verdict)
    if "consistent" in v:
        return "consistent"
    if "disfavoured" in v:
        return "disfavoured"
    if "inconclusive" in v:
        return "inconclusive"
    return "untestable"


ov = (ref.groupby("cand_rank").apply(object_verdict).rename("archive_verdict") if not ref.empty else pd.Series(dtype=object, name="archive_verdict"))
cand = cand.drop(columns=["archive_verdict"]).merge(
    ov, left_on="rank", right_index=True, how="left"
)
cand["archive_verdict"] = cand.archive_verdict.fillna("")
cand.loc[cand.status == "archive pending", "status"] = (
    cand.loc[cand.status == "archive pending", "archive_verdict"]
    .map(
        {
            "consistent": "archive consistent",
            "disfavoured": "disfavoured",
            "inconclusive": "inconclusive",
            "untestable": "untestable",
        }
    )
    .fillna("archive pending")
)
cand.to_csv(os.path.join(OUT, f"candidates{SUF}.csv"), index=False)
counts = {
    "eligible_spectra": int(len(todo)),
    "spectrum_verdicts": ref.spectrum_verdict.value_counts().to_dict(),
    "object_verdicts": ov.value_counts().to_dict(),
    "status_counts": cand.status.value_counts().to_dict(),
    "status_counts_equal_burden": cand[cand.tier == "equal_burden"]
    .status.value_counts()
    .to_dict(),
    "engine": "selections/spectra/census3.py + slopefit.py (frozen)",
    "eligibility": "PRISM, grade>=3, z_best>3 (paper mask)",
}
json.dump(
    counts, open(os.path.join(OUT, f"archive_refit_counts{SUF}.json"), "w"), indent=1
)
print(json.dumps(counts, indent=1))
