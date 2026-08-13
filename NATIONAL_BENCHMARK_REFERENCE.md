# National Benchmark Reference

Generated: `2026-08-13T21:25:33`

## Purpose

The current national benchmark is the France-wide reference for calibrating,
validating, and evaluating observable-successor linkage algorithms.

It does not prove legal renewal. It tests whether a method can identify a
plausible later successor procurement from the same buyer.

## Current Materialised State

- labelled anchors: `252`;
- labelled pairs: `7,031`;
- dev split: `134` anchors, `3,638` pair rows, `136` primary-positive pairs;
- validation split: `60` anchors, `1,763` pair rows, `62` primary-positive pairs;
- sealed test: closed for method selection.

## Current Method Comparison

The current benchmark evaluation includes all four methods. `M_D_fellegi_sunter` is no
longer skipped because `fs_match_probability` is now computed from the fitted
Fellegi-Sunter model when the current benchmark exposure is evaluated.

| Method | Threshold | Validation precision | Validation recall | Validation FPR | Accepted |
|---|---:|---:|---:|---:|---:|
| `M_A_deterministic` | 70.0 | 0.167 | 0.045 | 0.105 | 6 |
| `M_B_text_ranking` | 70.0 | 0.800 | 0.182 | 0.000 | 5 |
| `M_C_weighted_gated` | 70.0 | 0.600 | 0.409 | 0.105 | 15 |
| `M_D_fellegi_sunter` | 65.0 | 0.000 | 0.000 | 0.026 | 1 |

## Decision Rule

The incumbent `M_B_text_ranking @ 0.70` remains the primary method because it
has the strongest precision-first validation profile. A replacement should only
be promoted if it preserves or improves precision and false-positive control
without an unacceptable recall loss.

## Caveat

The current labels are protocol-generated development evidence with double-pass
validation and adjudication. They should not be described as official legal
renewal truth or independent human inter-annotator ground truth.
