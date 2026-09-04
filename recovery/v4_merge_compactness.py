"""v4 build step 10b: join the frozen aperture compactness measured on the VM
(vm_compactness.py output) onto the candidate catalog, and count the published compactness
criteria as tiers (never as a redefinition of the model's result):
  Akins+24   0.5 < f444(0.2")/f444(0.5") <= 0.7   -> compact_akins
  Labbe+23   f444(0.4")/f444(0.2") < 1.7          -> compact_labbe
Usage: python v4_merge_compactness.py [suffix]  (reads compactness{SUF}.csv)
"""

import json
import os
import sys

import numpy as np
import pandas as pd

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = "recovery"
SUF = sys.argv[1] if len(sys.argv) > 1 else ""
cand = pd.read_csv(os.path.join(OUT, f"candidates{SUF}.csv"))
comp = pd.read_csv(os.path.join(OUT, f"compactness{SUF}.csv"))
keep = [
    "source_id",
    "compactness_f444w",
    "labbe_compactness",
    "flux_d020",
    "flux_d040",
    "flux_d050",
    "measurable",
    "reason",
    "mosaic",
    "pixscale",
]
comp = comp[[c for c in keep if c in comp.columns]].drop_duplicates("source_id")
comp = comp.rename(
    columns={
        "measurable": "compactness_measurable",
        "reason": "compactness_reason",
        "mosaic": "compactness_mosaic",
        "pixscale": "compactness_pixscale",
    }
)
for c in [c for c in comp.columns if c != "source_id"]:
    if c in cand.columns:
        cand = cand.drop(columns=[c])
cand = cand.merge(comp, on="source_id", how="left")
cand["compactness_measurable"] = cand.compactness_measurable.fillna(False).astype(bool)
c444 = cand.compactness_f444w.values.astype(float)
lab = cand.labbe_compactness.values.astype(float)
cand["compact_akins"] = np.isfinite(c444) & (c444 > 0.5) & (c444 <= 0.7)
cand["compact_labbe"] = np.isfinite(lab) & (lab < 1.7)
# follow-up tier (a post-hoc tiering, stated as such): equal-burden, no archival verdict against
# it, inside the model's support range, compact by the Labbe criterion, catalog photo-z >= 3
cand["followup_tier"] = (
    (cand.tier == "equal_burden")
    & cand.status.isin(["untested", "archive consistent", "inconclusive", "secure z>3 (V untested)"])
    & ~cand.ood_flag.astype(bool)
    & cand.compact_labbe
    & (cand.z_phot >= 3)
)
cand.to_csv(os.path.join(OUT, f"candidates{SUF}.csv"), index=False)
eb = cand[cand.tier == "equal_burden"]
counts = {
    "candidates": int(len(cand)),
    "measured": int(cand.compactness_measurable.sum()),
    "measured_equal_burden": int(eb.compactness_measurable.sum()),
    "compact_labbe_equal_burden": int(eb.compact_labbe.sum()),
    "compact_akins_equal_burden": int(eb.compact_akins.sum()),
    "compact_labbe_all": int(cand.compact_labbe.sum()),
    "unmeasurable_reasons": cand.loc[~cand.compactness_measurable, "compactness_reason"]
    .fillna("not run")
    .value_counts()
    .to_dict(),
    "followup_tier": int(cand.followup_tier.sum()),
    "followup_tier_per_region": cand[cand.followup_tier].field.value_counts().to_dict(),
    "c444_median_equal_burden": float(
        np.nanmedian(c444[(cand.tier == "equal_burden").values])
    ),
}
json.dump(
    counts, open(os.path.join(OUT, f"compactness_counts{SUF}.json"), "w"), indent=1
)
print(json.dumps(counts, indent=1))
