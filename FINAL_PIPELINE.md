# Final Defensible Pipeline

Generated: `2026-08-13T19:42:07`

## Current Decision

The current primary method remains `M_B_text_ranking @ 0.70`.

On the latest v3 validation reference it gives:

- precision@1: `0.800`;
- recall@1: `0.182`;
- false-positive rate on negative anchors: `0.000`;
- accepted validation links: `5`.

`M_C_weighted_gated` has higher recall but also higher false-positive risk.
`M_D_fellegi_sunter` is now evaluated on v3, but it does not outperform `M_B`.

## End-to-End Workflow

```text
Official BOAMP API, 2015-2025
  -> schema-aware standardisation
  -> procurement episode reconstruction
  -> Grand Ouest digital study cohort
  -> broad same-buyer candidate generation
  -> four linkage algorithms compared on v3
  -> M_B primary successor selection
  -> survival dataset and expiry-aware sensitivity audit
```

The event remains an **observable successor procurement**, not a confirmed legal
renewal.

## Latest Benchmark State

- labelled anchors: `252`;
- labelled pairs: `7,031`;
- dev: `134` anchors and `3,638` pair rows;
- validation: `60` anchors and `1,763` pair rows;
- sealed test: not used for method selection.

## Current Source of Truth

- `data/processed/boamp_v2/linkage_evaluation_summary_v3_dev_primary.json`
- `data/processed/boamp_v2/linkage_evaluation_summary_v3_validation_primary.json`
- `data/processed/boamp_v2/benchmark_v3/modeling/benchmark_v3_modeling_summary.json`
- `reports/boamp_methodology_chapter.pdf`
- `notebooks/12_successor_linkage_and_evaluation.ipynb`

## Refresh Command

```bash
PYTHONPATH=. python3 scripts/run_final_pipeline.py --with-benchmark-v3-evaluation --force
```
