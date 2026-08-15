# BOAMP Observable Successor Procurement Study

This repository has one active analytical workflow, one processed-data root,
one reference sample, and one primary linkage policy.

## Research Claim

The project identifies **observable successor procurements** in official BOAMP
notices and studies time to visible reprocurement for awarded digital public
procurement episodes in Grand Ouest. An accepted link is not proof of a legal
contract renewal.

## Canonical Workflow

```text
Official BOAMP notices, 2015-2025
  -> schema-aware notice standardisation
  -> procurement episode reconstruction
  -> awarded Grand Ouest digital cohort
  -> same-buyer candidates 90 days to 8 years after award
  -> four linkage methods evaluated on the Grand Ouest regional reference
  -> primary M_B_text_ranking selection at 0.70
  -> right-censored survival dataset
  -> survival analysis plus threshold and borderline-link sensitivity checks
```

The frozen primary rule is `M_B_text_ranking @ 0.70`. It is a conservative
operating point, not a claim of mathematical optimality. Event-definition
sensitivity is carried by four arms — `M_B @ 0.80`, `M_B @ 0.70`, `M_B @ 0.60`,
and the `M_C_weighted_gated @ 0.70` contrast — plus a fixed borderline band
around the threshold. A duration-conditioned arm was built and evaluated during
development and has been removed in full; see `PROJECT_WORK_PROTOCOL.md` §3.6.

## One Active Layout

- Processed root: `data/processed/boamp/`
- Reference source data: `data/reference/regional_link_benchmark/`
- Materialised reference: `data/processed/boamp/regional_benchmark/`
- Primary accepted links: `data/processed/boamp/accepted_successor_links.parquet`
- Primary survival data: `data/processed/boamp/survival_dataset.parquet`
- Linkage configuration: `data/processed/boamp/linkage_config.json`
- Final workflow summary: `FINAL_PIPELINE.md`
- Technical report: `reports/boamp_methodology_chapter.pdf`
- Active notebooks: `notebooks/10` through `notebooks/14`, executed by
  `scripts/run_final_pipeline.py --with-notebooks`

There are no active project directories named by competing project versions.
Fields ending in `_schema` are data-contract identifiers, not alternative
analytical results. `notebooks/` contains only the five active,
pipeline-executed notebooks; nine earlier exploration notebooks (dated
2026-08-11) are isolated under `notebooks/archive/` and are never read by the
active pipeline — see `notebooks/archive/README.md`.

## One Reference, And Why The Other Was Retired

- **Grand Ouest regional reference (active, canonical).** A stratified review of
  `120` awarded digital procurement anchors across Bretagne, Pays de la Loire,
  and Normandie, carried out on 2026-08-11 against real BOAMP notices and
  official notice URLs, before the linkage methods in this repository existed.
  `112` anchors re-resolve onto the current episode reconstruction and `88` are
  usable. Its labels are independent of every method it scores. They are a
  single LLM research pass over the notices, their official URLs, and wider
  public sources, spot-checked on a subset by the project owner rather than
  verified anchor-by-anchor, so it is a **reference sample**, not ground truth. See `REGIONAL_BENCHMARK_REFERENCE.md`.
- **France-level benchmark (retired, removed).** Both of its annotation passes
  were emitted by deterministic rules built from the same text, CPV, and date
  evidence the linkage methods consume, so a method could score well on it only
  by agreeing with a hand-written rule made from its own features. Its numbers
  measured rule agreement, not correctness. Its data, construction scripts,
  annotation tooling, and the library modules that served only it have been
  deleted; `scripts/validate_canonical_state.py` fails the build if any of it
  reappears. The history remains in version control.

Only the regional reference is used for method comparison and the reported
precision/recall figures below.

## Current Materialised Results

- Study cohort: `3,800` awarded digital procurement episodes.
- Candidate pairs: `763,417`.
- Primary accepted links: `544`.
- Primary event rate: `0.1432`.
- Median observed successor time among linked events: `31.82` months.
- Excluding the `280` episodes whose best candidate scores within `±0.05` of the
  threshold leaves both headline hazard ratios pointing the same way, while the
  absolute KM level falls: comparative claims are robust to borderline links,
  absolute probabilities are not.
- Locked-split precision of `M_B @ 0.70`: `0.875` (95% CI `0.529`-`0.978`) on
  `8` accepted links; recall `0.389` (95% CI `0.203`-`0.614`); false-positive
  rate `0.000` on reviewed negative anchors.
- Candidate generation caps recall at `0.913`: it reaches `21` of the `23`
  reviewed successors.

Accuracy values come only from the regional reference. They are reference-sample
estimates, not independently validated accuracy: the labels are a single-pass
LLM-assisted review rather than an independent panel of procurement specialists,
and reference negatives are corpus-relative, so the reported false-positive rate
is an upper bound. The completed 60-pair production review is a separate frozen
diagnostic and does not estimate recall.

## Run Everything

From the repository root:

```bash
PYTHONPATH=. python3 scripts/run_final_pipeline.py --with-notebooks --with-tests
```

The command applies the primary pipeline, rebuilds the regional reference,
refreshes its evaluation, regenerates reader-facing artifacts, executes
notebooks, runs the test suite, and writes
`data/processed/boamp/final_pipeline_manifest.json`. Completed stages are
skipped. Use `--force` only when intentionally rebuilding all materialised
outputs from their inputs.

## Supporting Evidence

- `EXECUTIVE_SUMMARY.md`: one-page status for non-technical stakeholders.
- `REGIONAL_BENCHMARK_REFERENCE.md`: reference datasheet and method metrics.
- `QUALITY_EVIDENCE.md`: confusion matrices and threshold/curve evidence.
- `DATA_QUALITY_REPORT.md`: completeness, identity, and integrity checks.
- `TREND_ANALYSIS_REPORT.md`: descriptive temporal analysis.
- `SURVIVAL_ANALYSIS_REPORT.md`: current KM, Cox, parametric, prediction, censoring, and linkage-sensitivity evidence.
- `REVIEW_AUDIT_RESULTS.md`: frozen independent-link-review diagnostic.
- `METHODOLOGICAL_REFERENCES.md`: primary external methodological sources.
- `INTERNSHIP_GUIDE_COMPLIANCE.md`: mapping to the internship requirements.

## Boundaries

- Do not call accepted links confirmed legal renewals.
- Do not impute a missing duration or assume a four-year contract.
- Do not promote a threshold after inspecting the locked split; it is held out
  only because the operating point was frozen before that split was read.
- Do not describe the reference labels as independent human ground truth.
- Do not call the survival probabilities lower bounds. Missed successors push
  the measured level down and residual false links push it up, so the net
  direction is not identified.
- Do not reintroduce the France-level benchmark or the duration-conditioned
  linkage arm; both were removed, not paused.
