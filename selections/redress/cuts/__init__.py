"""U3 — the published LRD selections, re-implemented verbatim (milestone M0).

One module per paper, thresholds transcribed from
the author's verbatim excerpts of the primary sources and ASSERTED against the quoted text in the test suite. The six cuts
plus the plan's seventh adjunct ("Akins wing" = Akins et al. 2024, plan-text
resolution 2026-08-02):

=================  ==========================================================
key                selection
=================  ==========================================================
``labbe23``        Labbé+23 (UNCOVER): (red1|red2) & compact, SNR/mag gate
``kokorev24``      Kokorev+24: Labbé's red1 + LOOSENED red2 + BD cut +
                   per-color detection/upper-limit semantics
``kocevski24``     Kocevski+24: continuum-slope selection (β_UV/β_opt) with
                   redshift-dependent bandpasses + size + line-boost guards
``perezgonzalez24``Pérez-González+24 (SMILES): 2 colors + mag, no compactness
``barro23``        Barro+23: single color F277W−F444W>1.5 + mag
``greene24``       Greene+24 §5.2 UPDATED v-shape (their §3.1 target cut IS
                   labbe23 — see the module docstring)
``akins24``        Akins+24 (COSMOS-Web): single color + C444 window
=================  ==========================================================

Shared semantics (documented per module where a paper says otherwise):

* Inputs are DATA_CONTRACTS harmonized-photometry tables (fluxes in µJy with
  the cov/NaN missingness rule); magnitudes/colors derive through
  ``redress.conventions`` at call time — never stored, one source of truth.
* Every comparison is FAIL-CLOSED: a row whose required quantity is missing,
  non-finite, or non-positive-flux cannot pass that criterion (the one
  documented exception is Kokorev's explicit upper-limit substitution rule,
  implemented from the quoted sentence).
* Inequalities are exactly as printed — strict where the paper prints strict
  (boundary rows FAIL), inclusive only where the paper prints ``≤`` (the one
  case: Pérez-González's F444W ≤ 28; its Fig-1 caption discrepancy is
  documented in that module).
* Each ``select(...)`` returns a dict of named per-criterion boolean arrays
  plus the composite ``"selected"`` — the audit trail IS the return value.
* Post-photometric refinements that need external products (Labbé's GALFIT
  PSF-dominance, Akins's SED-χ² brown-dwarf grid) enter as OPTIONAL
  caller-supplied masks, documented per module; they are never silently
  skipped — absent mask ⇒ absent key in the result.

Deployment reality check (Hviding+25, the M1 cross-check): down to
F444W < 26.5 these selections recover only 35–62% of spectroscopic LRDs —
that incompleteness is the headline target here, and re-implementing the cuts
exactly is what makes the completeness atlas mean something.
"""

from redress.cuts import (
    akins24,
    barro23,
    barro24b,
    greene24,
    kocevski24,
    kokorev24,
    labbe23,
    perezgonzalez24,
)

#: module key -> the module's ``select`` callable (signatures differ by paper —
#: the M0 runner supplies each cut's documented extra inputs by key).
CUTS = {
    "labbe23": labbe23.select,
    "kokorev24": kokorev24.select,
    "kocevski24": kocevski24.select,
    "perezgonzalez24": perezgonzalez24.select,
    "barro23": barro23.select,
    "greene24": greene24.select,
    "akins24": akins24.select,
}

#: Comparators added AFTER the benchmark was fixed (revision, 2026-09-06). Kept out of
#: ``CUTS`` on purpose: the seven above are the pre-stated benchmark of record and every
#: released union, burden and paired count is computed over them alone. Anything here is
#: reported separately, as an eighth line, never folded into the union of record.
COMPARATORS = {
    "barro24b": barro24b.select,
}
