"""Fetch raw three-band FITS cutouts (F150W, F277W, F444W; DJA grizli cutout service) for
the objects shown in the paper's image figure: rule-missed spectroscopic LRDs
the model recovers, spectroscopic LRDs the model still misses, and follow-up-tier
candidates. Objects are chosen by fixed rules (ranked lists, no hand picking) and written
to paper/figures/stamps/<group>_<field>_<id>.fits with a manifest.

The service's size=5 in the URL below is a radius, not a width: it returns a 10 arcsec
field as 200 x 200 pixels at 0.05 arcsec per pixel, which is the CD1_1 = -1.388889e-05 deg
carried by every stamp header. Anything that converts these pixels to angle must use 0.05;
reading size=5 as the full width is what put the wrong scale in the figure code.

Run from the repository root: python paper/fetch_stamps.py
"""

import json
import os
import time
import urllib.request

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V = os.path.join(ROOT, "recovery")
OUT = os.path.join(ROOT, "paper", "figures", "stamps")
os.makedirs(OUT, exist_ok=True)
URL = (
    "https://grizli-cutout.herokuapp.com/thumb?ra={ra:.6f}&dec={dec:.6f}"
    "&size=5&filters=f150w-clear,f277w-clear,f444w-clear&output=fits"
)

ev = json.load(open(os.path.join(V, "evaluation.json")))
feat = pd.read_parquet(
    os.path.join(V, "features.parquet"),
    columns=[
        "field",
        "id",
        "ra",
        "dec",
        "mag_f444w",
        "c_f277w_f444w",
        "z_phot",
        "picked",
    ],
)
cand = pd.read_csv(os.path.join(V, "candidates.csv"))

rec = pd.DataFrame(ev["recovered_rule_missed_list"]).merge(
    feat, on=["field", "id"], suffixes=("", "_f")
)
rec = rec.sort_values("score", ascending=False)
# fixed rule: the three highest-scoring recovered misses redder than F277W-F444W = 1 and the
# three highest-scoring bluer than 0.5, so the figure shows both ends of the colour range
rec_red = rec[rec.c_f277w_f444w > 1.0].head(3)
rec_blue = rec[rec.c_f277w_f444w < 0.5].head(3)
still = pd.DataFrame(ev["still_missed_list"]).merge(
    feat, on=["field", "id"], suffixes=("", "_f")
)
# fixed rule (round 2, Reviewer C finding 3): the three confirmed LRDs that neither
# selection recovers whose out-of-fold score falls closest below the equal-burden threshold
# of their own fold. The five folds carry thresholds from 0.68 to 0.89, so ranking on the
# raw score is not the same set as ranking on the distance to the cut, and the caption
# promises the second. t_burden is the per-row fold threshold in oof_scores.parquet.
oof = pd.read_parquet(
    os.path.join(V, "oof_scores.parquet"), columns=["field", "id", "t_burden"]
)
still = still.merge(oof, on=["field", "id"], how="left")
still = still[~still.picked.astype(bool)]
still = still.assign(gap=still.t_burden - still.score)
assert (still.gap > 0).all(), "a still-missed object is above its own fold threshold"
still = still.sort_values("gap").head(3)
fu = (
    cand[cand.followup_tier].sort_values("rank").head(6)
)  # the six highest-ranked follow-up-tier candidates

rows = []
for g, d in (
    ("recovered", pd.concat([rec_red, rec_blue])),
    ("missed", still),
    ("candidate", fu),
):
    for r in d.itertuples():
        rows.append(
            dict(
                group=g,
                field=r.field,
                id=int(r.id),
                ra=float(r.ra),
                dec=float(r.dec),
                mag_f444w=float(r.mag_f444w),
                c_f277w_f444w=float(r.c_f277w_f444w),
                z_phot=float(r.z_phot),
                score=float(
                    getattr(r, "score", getattr(r, "ranking_score", float("nan")))
                ),
            )
        )
man = pd.DataFrame(rows)


def valid(p):
    try:
        with open(p, "rb") as fh:
            return fh.read(6) == b"SIMPLE" and os.path.getsize(p) > 50000
    except OSError:
        return False


ok = 0
for r in man.itertuples():
    p = os.path.join(OUT, f"{r.group}_{r.field}_{r.id}.fits")
    if valid(p):
        ok += 1
        continue
    url = URL.format(ra=r.ra, dec=r.dec)
    for attempt in range(4):
        try:
            urllib.request.urlretrieve(url, p)
            if valid(p):
                ok += 1
                break
        except Exception as e:  # noqa: BLE001
            print("retry", r.field, r.id, e, flush=True)
        time.sleep(3 * (attempt + 1))
    else:
        print("FAILED", r.field, r.id)
man.to_csv(os.path.join(OUT, "manifest.csv"), index=False)
print(f"{ok} of {len(man)} stamps on disk")
print(man.to_string(index=False))
