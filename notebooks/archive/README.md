# Archived Exploration Notebooks

These nine notebooks are from the project's early exploration phase (all dated
2026-08-11, before the current canonical pipeline was built on 2026-08-12
through 2026-08-14). They are retained for provenance and to show the
methodological path that led to the current design — not deleted, but not
part of the active pipeline either.

## What They Are

- `boamp_full_raw_eda.ipynb`, `boamp_preprocessing_data_engineering.ipynb`:
  early, unnumbered drafts of the raw-data EDA and preprocessing steps that
  became `notebooks/10_standardized_notice_and_episode_evidence_audit.ipynb`
  and `notebooks/11_cohort_and_data_quality.ipynb`. Their output tables and
  figures are archived separately at `archive/full_raw_eda/`.
- `03_boamp_grand_ouest_preprocessing_data_engineering.ipynb` through
  `09_linkage_parameter_selection_diagnostics.ipynb`: the original
  preprocessing → linkage-baseline → error-analysis → high-precision-iteration
  → parameter-diagnostics sequence, built against the **regional reference**
  (the same 120-anchor Grand Ouest review that is canonical again today, now
  at `data/reference/regional_link_benchmark/`) and an old processed-data
  layout (`data/processed/boamp_grand_ouest/`) that no longer exists.

## Why They're Archived, Not Active

They predate the current pipeline (`notebooks/10` through `notebooks/14`,
driven by `scripts/run_final_pipeline.py`) and were written against an episode
reconstruction whose identifiers no longer exist. Running them today would fail
immediately: their input paths are not in the current data layout. Note that
their use of the regional reference is not a reason to trust their numbers —
the reference has since been re-resolved onto current episode identifiers, and
the linkage methods themselves changed. They are not
executed by `scripts/run_final_pipeline.py` and their numbers/results are
**not** current evidence — do not cite figures or tables from these notebooks
in the final report.

## Current Equivalents

| Archived notebook | Current equivalent |
|---|---|
| `boamp_full_raw_eda.ipynb` | `notebooks/10_standardized_notice_and_episode_evidence_audit.ipynb` |
| `boamp_preprocessing_data_engineering.ipynb`, `03_...` | `notebooks/11_cohort_and_data_quality.ipynb` |
| `04_reference_benchmark_preparation.ipynb` | superseded by `scripts/build_regional_benchmark.py` |
| `05_...` through `09_...` | `notebooks/12_successor_linkage_and_evaluation.ipynb` |
