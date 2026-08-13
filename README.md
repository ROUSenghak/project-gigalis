# BOAMP Observable Successor Procurement Study

This repository has one current analytical pipeline and one primary result.
Older labels such as `v1`, `v2`, and `v3` may still appear in internal paths or
schema/version fields for reproducibility, but they are not competing project
versions.

## Current Claim

The project identifies **observable successor procurements** in BOAMP data and
uses them to study time-to-visible-reprocurement for awarded digital public
procurement episodes.

It does **not** claim to certify legal contract renewals.

## Current Source Of Truth

- Main technical report: `reports/boamp_methodology_chapter.pdf`
- Final pipeline summary: `FINAL_PIPELINE.md`
- Benchmark summary: `NATIONAL_BENCHMARK_REFERENCE.md`
- Linkage quality evidence: `QUALITY_EVIDENCE.md`
- Primary survival dataset: `data/processed/boamp_v2/survival_dataset.parquet`
- Primary accepted links: `data/processed/boamp_v2/accepted_successor_links.parquet`
- Expiry-aware audit queue: `data/processed/boamp_v2/expiry_link_review.csv`

## Current Pipeline

```text
Official BOAMP notices, 2015-2025
  -> schema-aware standardised notices
  -> procurement episode reconstruction
  -> Grand Ouest awarded digital cohort
  -> same-buyer candidate generation
  -> linkage method comparison on the current national benchmark
  -> primary successor selection with M_B_text_ranking @ 0.70
  -> right-censored survival dataset
  -> survival analysis and sensitivity checks
```

## Current Results

- Cohort: `3,800` awarded digital procurement episodes.
- Candidate pairs: `763,417`.
- Primary accepted successor links: `544`.
- Primary event rate: `0.1432`.
- Median observed successor time among linked events: `31.82` months.
- Expiry-aware audit accepted links: `504`.
- Expiry-aware changed anchors for review: `42`.

Current validation benchmark result for the primary method:

- method: `M_B_text_ranking @ 0.70`;
- validation precision: `0.800`;
- validation recall: `0.1818`;
- validation false-positive rate: `0.000`;
- accepted validation links: `5`.

## Why This Is The Primary Method

The selected method is precision-first. In the survival dataset, a false
positive link creates a false event and a false event time. An abstention is
handled as right-censoring, which is more conservative. Therefore the current
project prioritises credible links over maximum coverage.

## Run The Current Pipeline

The normal current refresh is:

```bash
PYTHONPATH=. python3 scripts/run_final_pipeline.py --with-current-benchmark-evaluation
```

Use `--force` only when intentionally rebuilding existing artifacts:

```bash
PYTHONPATH=. python3 scripts/run_final_pipeline.py --with-current-benchmark-evaluation --force
```

Run validation tests with:

```bash
PYTHONPATH=. pytest -q
```

## Internal Version Labels

The repository keeps some internal lineage labels because they protect
reproducibility:

- `data/processed/boamp_v2/` is the current processed data layer.
- `benchmark_v3` is the current national benchmark directory.
- artifact fields such as `*_version` identify schema or artifact contracts.

These labels should not be interpreted as separate final project versions. For
writing and presentation, use the wording **current pipeline**, **current
national benchmark**, and **primary observable-successor method**.

## Boundaries

- Do not describe accepted links as confirmed legal renewals.
- Do not treat missing duration as an assumed four-year contract.
- Do not promote the expiry-aware audit arm unless the changed links are
  reviewed and the primary precision-first decision is intentionally changed.
- Do not over-tune thresholds on the current validation split.
