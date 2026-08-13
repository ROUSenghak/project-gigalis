# Independent Link Review Protocol

Generated: `2026-08-13T23:10:59`

Pairs prepared: `60`

## Purpose

This review is the minimum independent check needed before reporting linkage
precision as externally validated. The reviewer file is blinded: it contains
descriptions, buyer fields, CPVs, and dates, but no bootstrap label, algorithm
decision, score, or sampling stratum.

## Sample Composition

| Hidden audit stratum | Rows |
|---|---:|
| BUYER_DECLARED_RESOLVED | 20 |
| HIGH_SIMILARITY_STRUCTURAL_NEGATIVE | 20 |
| PRIMARY_ACCEPTED | 20 |

The strata deliberately mix likely matches, difficult non-matches, and
buyer-declared relationships. They estimate performance on this challenge set,
not national prevalence. The audit key must remain hidden until labels are
finalised.

## Review Fields

For each row, decide:

1. `same_legal_buyer_Y_N_UNCERTAIN`: whether the notices concern the same legal
   buyer. A matching name is not enough when legal form or SIREN evidence conflicts.
2. `relationship_label`: one of `RENEWAL_OF_EXPIRING`,
   `NEXT_PHASE_OF_PROGRAMME`, `RETENDER_AFTER_FAILURE`, `PARALLEL_LOT`,
   `EXTENSION_SAME_CONTRACT`, `UNRELATED`, or `UNUSABLE`.
3. `observable_successor_Y_N_UNCERTAIN`: `Y` only for a later procurement that
   plausibly replaces or continues the same need. Use `UNCERTAIN` when the visible
   evidence cannot support a defensible decision.
4. Confidence and short evidence-based notes.

Timing alone must not decide the label. Early successor publication is possible,
and missing duration must not be replaced with an assumed four-year term.

## Files

- Reviewer file: `data/review/independent_link_review_sample.csv`
- Hidden audit key: `data/review/independent_link_review_audit_key.csv`
- Review provenance: `data/review/review_provenance.json`

Reviewer identity is not stored row by row. The review source and whether it is
independent human validation must instead be recorded truthfully in the
provenance file.

## Acceptance Rule

Freeze `M_B_text_ranking @ 0.70` while reviewing. After review, compute exact
binomial confidence intervals for accepted-link precision overall and by sample
stratum. Do not tune the threshold on these labels and then report the same rows
as validation.
