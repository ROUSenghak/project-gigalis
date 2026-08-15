# Regional Reference Datasheet

Generated: `2026-08-15T15:50:58`

## What This Reference Is

The active reference for successor linkage is a stratified review of
`120` awarded digital procurement anchors drawn from the
study region itself: Grand Ouest (Bretagne, Pays de la Loire, Normandie). Each anchor was reviewed
against the real BOAMP notices and official notice URLs of its candidates on
`2026-08-11`, before the linkage methods compared below existed.

- source: `data/reference/regional_link_benchmark/BOAMP_Internship_Reference_120.csv`;
- version: `internship_v1.0`;
- construction: single-pass LLM-assisted evidence review by the project owner against real BOAMP notices and official notice URLs;
- anchor award dates: `2015-04-14` to `2023-05-07`;
- observation cutoff: `2025-12-31`;
- independent of the linkage algorithms: `True`;
- independent human specialist review: `False`.

It is a **regional reference sample**, not ground truth, and not proof of legal
renewal.

## What It Replaced And Why

It replaced a France-level benchmark whose two annotation passes were both
emitted by deterministic rules in a single script, built from the same text,
CPV, and date evidence the linkage methods consume. A method could score well
there only by agreeing with that rule, so the numbers measured rule agreement
rather than correctness. Those artifacts have been removed from the repository
in full; their history remains in version control.

## Current Materialised State

- reviewed anchors: `120`;
- resolved onto the current episode reconstruction: `112`;
- pilot split: `16` usable anchors, `5` positive, `11` negative;
- locked split: `72` usable anchors, `18` positive, `54` negative;
- pair rows: `5,221` pilot, `20,917` locked;
- candidate-generation recall ceiling: `0.9130`.

## Label Definitions

- `OBSERVED_SUCCESSOR`: a later procurement in the reviewed candidate set that
  plausibly replaces or continues the anchor's need.
- `NO_OBSERVED_SUCCESSOR_IN_SCOPE`: none among the candidates the reviewer saw.
  This is corpus-relative, not proof that no renewal occurred.
- `OUTSIDE_SCOPE` / `INSUFFICIENT_INFORMATION`: the reviewer declined to decide;
  these anchors are excluded from evaluation rather than counted as negatives.

## Current Method Comparison On The Locked Split

| Method | Threshold | Precision | 95% CI | Recall | 95% CI | FPR | Accepted |
|---|---:|---:|---|---:|---|---:|---:|
| `M_A_deterministic` | 70.0 | 0.533 | 0.301-0.752 | 0.444 | 0.246-0.663 | 0.130 | 15 |
| `M_B_text_ranking` | 70.0 | 0.875 | 0.529-0.978 | 0.389 | 0.203-0.614 | 0.000 | 8 |
| `M_C_weighted_gated` | 70.0 | 0.522 | 0.330-0.708 | 0.667 | 0.438-0.837 | 0.185 | 23 |
| `M_D_fellegi_sunter` | 65.0 | 0.200 | 0.036-0.625 | 0.056 | 0.010-0.258 | 0.018 | 5 |

Intervals are Wilson score intervals. They overlap heavily: this reference
separates the methods only coarsely, and any claim that one method beats another
must survive that overlap.

## Decision Rule

`M_B_text_ranking @ 0.70` remains the frozen primary event definition. It was
fixed before this reference was consulted and has not been moved since, which is
what allows the locked split to be reported as held out. Choosing a threshold
from these rows now would convert the locked split into a tuning set. A
replacement requires a pre-specified selection rule, direct review of the
incremental links, and fresh evidence.

## Known Limitations

- Labels are a single-pass LLM-assisted review, not an independent multi-reviewer human panel.
- Negatives are corpus-relative: the reviewer saw roughly 25 candidates per anchor, not the full pool, so a negative means 'no successor among those shown'.
- The reference records one successor relationship per anchor and does not distinguish renewal from next-phase, so only the primary event set exists.
- Design weights come from the v1 sampling frame, whose stratum populations were computed on the earlier episode reconstruction.
- Sample size is small; every metric needs its interval read alongside it.

## What It May Legitimately Be Used For

Comparing linkage methods on the same exposed candidate pairs, reading the
frozen operating point held out, and bounding recall through candidate
generation. It may not be used to claim externally validated accuracy, national
prevalence, or legal renewal status.
