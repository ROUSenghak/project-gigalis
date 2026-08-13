# Linkage Quality Evidence

Generated from the held-out split of the current national development reference. These labels come from deterministic bootstrap rules, not independent specialist review.

## Academic Basis And Evidence Boundary

The metric choice is supported by [Davis and Goadrich (2006)](https://doi.org/10.1145/1143844.1143874), who analyse the relationship between ROC and precision-recall curves for skewed binary decisions, and [Saito and Rehmsmeier (2015)](https://doi.org/10.1371/journal.pone.0118432), who show why precision-recall analysis is more informative for imbalanced data. The classical probabilistic linkage comparator follows [Fellegi and Sunter (1969)](https://doi.org/10.1080/01621459.1969.10501049). These sources justify methods and diagnostics; they do not validate this project's labels, numerical results, or `0.70` threshold.

The figures below are generated from this project's data and code. Generic web or presentation illustrations are explanatory aids only and are not used as academic evidence.

## What Each Diagnostic Means

- **Confusion matrix:** anchor-level evidence, matching the actual pipeline decision: one accepted successor or abstention.
- **Exact-successor accounting:** stricter project metric; a wrong successor is both a false accepted link and a missed true successor.
- **ROC curve:** pair-level score-ranking diagnostic over exposed candidate pairs. Useful, but less important than precision-recall because positives are rare.
- **Precision-recall curve:** pair-level score-ranking diagnostic. This is the better curve for this project because validation has only 62 positive pairs out of 1,763 candidate pairs.
- **Threshold trade-off:** anchor-level sweep of the actual `M_B` top-1 decision. It shows how strict acceptance changes precision, recall, false-positive rate, and link volume.

## Held-Out Internal Event-Detection Confusion Matrix

Rows are actual anchor status; columns are predicted link/abstention. Here, a wrong candidate on a positive anchor still counts as detecting that the anchor has a successor.

| method | threshold | tp | fp | fn | tn |
| --- | --- | --- | --- | --- | --- |
| M_A_deterministic | 70.0000 | 2 | 4 | 20 | 34 |
| M_B_text_ranking | 70.0000 | 5 | 0 | 17 | 38 |
| M_C_weighted_gated | 70.0000 | 11 | 4 | 11 | 34 |
| M_D_fellegi_sunter | 65.0000 | 0 | 1 | 22 | 37 |

## Held-Out Internal Exact-Successor Accounting

This is the stricter accounting behind project precision and recall. Cells do not necessarily sum to the number of anchors because a wrong successor contributes one FP and one FN.

| method | threshold | tp | fp | fn | tn |
| --- | --- | --- | --- | --- | --- |
| M_A_deterministic | 70.0000 | 1 | 5 | 21 | 34 |
| M_B_text_ranking | 70.0000 | 4 | 1 | 18 | 38 |
| M_C_weighted_gated | 70.0000 | 9 | 6 | 13 | 34 |
| M_D_fellegi_sunter | 65.0000 | 0 | 1 | 22 | 37 |

## Held-Out Internal Pair-Level ROC and Precision-Recall Metrics

| method | pair_roc_auc | pair_average_precision | positive_pairs | negative_pairs |
| --- | --- | --- | --- | --- |
| M_A_deterministic | 0.5129 | 0.0390 | 62 | 1701 |
| M_B_text_ranking | 0.9453 | 0.5354 | 62 | 1701 |
| M_C_weighted_gated | 0.7900 | 0.3257 | 62 | 1701 |
| M_D_fellegi_sunter | 0.6573 | 0.0533 | 62 | 1701 |

## M_B Anchor-Level Threshold Trade-Off

| threshold | accepted_links | true_positive | precision | recall | false_positive_rate | coverage |
| --- | --- | --- | --- | --- | --- | --- |
| 0.5000 | 16 | 11 | 0.6875 | 0.5000 | 0.0789 | 0.2667 |
| 0.5500 | 11 | 9 | 0.8182 | 0.4091 | 0.0000 | 0.1833 |
| 0.6000 | 9 | 8 | 0.8889 | 0.3636 | 0.0000 | 0.1500 |
| 0.6500 | 6 | 5 | 0.8333 | 0.2273 | 0.0000 | 0.1000 |
| 0.7000 | 5 | 4 | 0.8000 | 0.1818 | 0.0000 | 0.0833 |
| 0.7500 | 4 | 4 | 1.0000 | 0.1818 | 0.0000 | 0.0667 |
| 0.8000 | 2 | 2 | 1.0000 | 0.0909 | 0.0000 | 0.0333 |

At `0.60`, the internal validation split has 8 correct successors among 9 accepted links (precision `0.8889`) and recall `0.3636`. At the frozen `0.70`, it has 4 correct successors among 5 accepted links (precision `0.8000`) and recall `0.1818`. Thus `0.60` empirically dominates `0.70` on this particular validation sample. The development evidence points the other way: `0.70` has precision `0.8750` and FPR `0.0345`, compared with precision `0.8000` and FPR `0.0575` at `0.60`. The production-link diagnostic at `0.70` also confirmed only 14 of 20 links conservatively. Therefore `0.70` is retained as the previously frozen conservative baseline, not described as optimal; `0.60` remains a required sensitivity arm. Promoting the unreviewed lower threshold after observing four favourable incremental validation cases would be post-hoc tuning.

## Interpretation

`M_B_text_ranking @ 0.70` is the frozen conservative operating baseline, not a claim of threshold optimality. Its role is to provide one reproducible primary event definition while `0.60`, `0.80`, `M_C`, and expiry-aware variants quantify sensitivity. The bootstrap labels support development comparisons but cannot establish external accuracy.

## Plot Files

- `reports/figures/benchmark_validation_confusion_matrices.png`
- `reports/figures/benchmark_validation_pair_roc.png`
- `reports/figures/benchmark_validation_pair_precision_recall.png`
- `reports/figures/benchmark_validation_m_b_threshold_tradeoff.png`
