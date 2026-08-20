# BOAMP Observable Successor Procurement Study

This repository has one active analytical workflow, one processed-data root,
two reference samples, and one primary linkage policy.

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
  -> supervised business technology taxonomy over procurement text, layered on top
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
- Technology taxonomy source data: `data/reference/tech classification/`
- Technology taxonomy code: `boamp_pipeline/technology_{taxonomy,models,evidence}.py`,
  run by `scripts/build_technology_taxonomy.py`
- Technology taxonomy outputs: `data/processed/boamp/technology/`
- Frozen classifier configuration: `data/processed/boamp/technology/final_model_config.json`
- Episode-level technology predictions:
  `data/processed/boamp/technology/episode_technology_predictions.csv`
- Final workflow summary: `FINAL_PIPELINE.md`
- Technical report: `reports/boamp_methodology_chapter.pdf`
- Active notebooks: `notebooks/10` through `notebooks/15`, executed by
  `scripts/run_final_pipeline.py --with-notebooks`
- Pipeline entry point: `scripts/run_final_pipeline.py`. Every other script under
  `scripts/` is a stage it invokes, with three exceptions that are **raw-export
  utilities, not analysis stages**: `build_boamp_raw_csv.py`,
  `create_boamp_grand_ouest_raw_csv.py` and
  `create_boamp_grand_ouest_annual_parquet.py` serialise the acquired JSONL into
  CSV/Parquet views for external consumption without cleaning, deduplicating or
  filtering. Nothing in the analysis reads their output; the analysis reads the
  JSONL directly through `build_standardized_notices.py`
- Library code: `boamp_pipeline/` (imported by both the scripts and the notebooks,
  so a notebook runs the same code the pipeline runs)
- Tests: `tests/`, collected by `pytest.ini`, which restricts collection to that
  directory
- Run record: `data/processed/boamp/final_pipeline_manifest.json` (git state,
  dependency versions, stage status, input checksums, headline counts) and
  `data/processed/boamp/canonical_state_validation.json` (cross-artifact gate)
- Archives, read by nothing: `archive/` (retired France-level benchmark material
  and the raw-data EDA) and `notebooks/archive/`

There are no active project directories named by competing project versions.
Fields ending in `_schema` are data-contract identifiers, not alternative
analytical results. `notebooks/` contains only the six active,
pipeline-executed notebooks; nine earlier exploration notebooks (dated
2026-08-11) are isolated under `notebooks/archive/` and are never read by the
active pipeline — see `notebooks/archive/README.md`.

## Two Reference Samples

The two references answer different questions and are never mixed. The regional
link reference scores *linkage*: whether an accepted successor is real. The
technology annotation scores *classification*: what a procurement is for. Below,
"the reference" always means the linkage one.

## The Technology Annotation Corpus

`500` BOAMP notices spanning 2015-2025, manually labelled into `11` business
technology classes. It is the training and evaluation data for the supervised
classifier and has no role in linkage. It was delivered as a single labelled
file with no annotator identifier and no second pass, so no inter-annotator
agreement statistic exists, and its class counts are quotas rather than
prevalence. See `TECHNOLOGY_TAXONOMY_REPORT.md`.

## The Linkage Reference, And Why The Other Was Retired

- **Grand Ouest regional reference (active, canonical).** A stratified review of
  `120` awarded digital procurement anchors across Bretagne, Pays de la Loire,
  and Normandie, carried out on 2026-08-11 against real BOAMP notices and
  official notice URLs, before the linkage methods in this repository existed.
  `112` anchors re-resolve onto the current episode reconstruction and `88` are
  usable. Its **labels** are independent of every method it scores -- none of them
  existed when the review was carried out. They are a
  single LLM research pass over the notices, their official URLs, and wider
  public sources, spot-checked on a subset by the project owner rather than
  verified anchor-by-anchor, so it is a **reference sample**, not ground truth.
  A narrower gap sits beside that one: the rule that chose the ~25 candidates
  exported per anchor for review was not recorded and cannot be reconstructed
  from this repository, so **recall** and the candidate-reachability ceiling are
  not fully independent of the text score they evaluate. **Precision** is
  unaffected. See `REGIONAL_BENCHMARK_REFERENCE.md`.
- **France-level benchmark (retired, removed).** Both of its annotation passes
  were emitted by deterministic rules built from the same text, CPV, and date
  evidence the linkage methods consume, so a method could score well on it only
  by agreeing with a hand-written rule made from its own features. Its numbers
  measured rule agreement, not correctness. Its data, construction scripts,
  annotation tooling, and the library modules that served only it have been
  deleted; `scripts/validate_canonical_state.py` fails the build if any of it
  reappears. The history remains in version control.

Only the regional reference is used for linkage method comparison and the
reported precision/recall figures below.


## Current Materialised Results

- Study cohort: `3,800` awarded Grand Ouest procurement episodes carrying at
  least one CPV code in divisions 32/35/48/72. The digital filter is an
  **any-code** rule at episode level, not a rule about the main CPV, so
  `1,176` of them (`30.9%`) are multi-lot procurements whose main CPV lies
  outside those divisions. `digital_segment`, the stratifying variable, is the
  lowest-numbered digital division present. See `DATA_QUALITY_REPORT.md`.
- Candidate pairs: `763,417`.
- Primary accepted links: `544`.
- Primary event rate: `0.1432`.
- Median observed successor time among linked events: `31.82` months.
- Excluding the `280` episodes whose best candidate scores within `±0.05` of the
  threshold leaves both headline hazard ratios pointing the same way, while the
  absolute KM level falls: comparative claims are robust to borderline links,
  absolute probabilities are not. (A separate `280` anchors generated no
  candidate at all — two different sets, and the borderline check does not touch
  the second.)
- The largest linked-versus-censored imbalance is candidate-pool size
  (SMD `+0.470` on the log scale). One sensitivity Cox model adds it: CPV-35
  barely moves (`1.553` → `1.512`) while the framework hazard ratio attenuates
  (`1.751` → `1.617`), so part of the framework association is differential
  detectability rather than re-procurement behaviour. The main Cox model is
  unchanged. See `SURVIVAL_ANALYSIS_REPORT.md`.
- Re-censoring the `173` accepted links that carry the documented false-positive
  signatures — word-level similarity below `0.50`, or a successor shared with
  another anchor — gives the same verdict from the other direction: CPV-35 moves
  `1.553` to `1.541` and framework `1.751` to `1.692`, while the 12-month KM
  level falls from `4.62%` to `2.64%`. The framework association is not an
  artefact of shared framework boilerplate.
- Locked-split precision of `M_B @ 0.70`: `0.875` (95% CI `0.529`-`0.978`) on
  `8` accepted links; recall `0.389` (95% CI `0.203`-`0.614`); false-positive
  rate `0.000` on reviewed negative anchors.
- Candidate generation exposed `21` of the `23` reviewed successors in the
  regional reference, so recall against this sample is capped at `0.913`. Both
  unreachable cases are attributed to a named blocking condition — one anchor
  never entered the cohort because its award notice carries no structured Grand
  Ouest address, and one buyer changed legal form from CCAS to CIAS with no
  shared SIREN. Neither is an implementation defect.
- Technology classifier: out-of-fold macro-F1 `0.744` (95% family-bootstrap CI
  `0.682`-`0.791`) on the `500`-notice annotated corpus, against `0.473`
  (`0.413`-`0.526`) for the best CPV/descriptor benchmark on identical folds and
  the same regularisation range. The paired difference is `0.271`
  (`0.201`-`0.340`), excluding zero. Applied to all `3,800` cohort episodes;
  every episode keeps a prediction and `235` (`6.2%`) clear the `0.70`
  operational confidence cutoff. That confidence is the **raw** class score:
  Platt scaling was evaluated inside the same grouped splits and rejected by the
  pre-specified rule, because its macro-F1 cost exceeded the `0.02` budget even
  though its calibration gain cleared the `0.02` requirement. `AI` has `7`
  labelled notices and is reported as a rare-class limitation, not as a measured
  capability.
- Technology-level survival and trend enrichment is gated twice: a class must be
  a substantive technology the classifier separates (out-of-fold F1 `>= 0.65` on
  at least `10` annotated notices) *and* carry enough episodes and events. Five
  classes qualify for survival. Including the fallback residuals instead would
  make the log-rank result look markedly stronger (`p = 0.0001` against
  `p = 0.036`), which is why the classifier gate is applied.
- Accepted links stay inside one CPV division in `351` of the `538` cases where
  both divisions are observed (`0.652`); the reviewed reference successors cross
  divisions at a comparable rate (`9` of `23`). Hard same-CPV blocking is
  therefore not imposed: it would discard those `9` reviewed successors and cut
  the attainable recall ceiling to `0.609`.

Accuracy values come only from the regional reference. They are reference-sample
estimates, not independently validated accuracy: the labels are a single-pass
LLM-assisted review rather than an independent panel of procurement specialists,
and reference negatives are corpus-relative, so the reported false-positive rate
is conservative by construction rather than a population-wide rate. The
completed 60-pair production review is a separate frozen
diagnostic and does not estimate recall.

## Run Everything

From the repository root:

```bash
PYTHONPATH=. python3 scripts/run_final_pipeline.py --with-notebooks --with-tests
```

The command runs every stage in dependency order:

```text
base -> episodes -> cohort -> linkage -> survival
     -> technology
     -> evidence and reader-artifact refresh
     -> canonical state validation
     -> manifest
```

The technology layer runs **before** any artifact or validator that quotes it.
Reader-facing documents (`EXECUTIVE_SUMMARY.md`, `FINAL_PIPELINE.md`,
`REGIONAL_BENCHMARK_REFERENCE.md`, the methodology chapter) and
`canonical_state_validation.json` all read technology numbers, so running them
first would silently republish the previous run's results.

Completed upstream stages are skipped when their outputs are newer than the code
that writes them; everything downstream of the data recomputes every run. Use
`--force` only when intentionally rebuilding all materialised outputs from their
inputs.

The run writes `data/processed/boamp/final_pipeline_manifest.json`, which records
the git commit and working-tree state, Python and library versions, the status of
every stage, SHA-256 checksums of the reference inputs and the canonical outputs,
and the headline counts read back off the artifacts.

## Supporting Evidence

- `EXECUTIVE_SUMMARY.md`: one-page status for non-technical stakeholders.
- `REGIONAL_BENCHMARK_REFERENCE.md`: reference datasheet and method metrics.
- `QUALITY_EVIDENCE.md`: confusion matrices and threshold/curve evidence.
- `DATA_QUALITY_REPORT.md`: completeness, identity, and integrity checks.
- `TREND_ANALYSIS_REPORT.md`: descriptive temporal analysis.
- `SURVIVAL_ANALYSIS_REPORT.md`: current KM, Cox, parametric, operational
  12/24-month probabilities, censoring, and linkage/borderline/template-risk
  sensitivity evidence.
- `TECHNOLOGY_TAXONOMY_REPORT.md`: annotated corpus audit, leakage-preventing
  grouping, model comparison against the CPV benchmark, per-class metrics with
  support, learning curve, error analysis, temporal robustness, the CamemBERT
  decision, and the episode-level deployment with its confidence diagnostics.
- `CANDIDATE_GENERATION_AUDIT.md`: blocking-loss attribution and CPV-continuity
  evidence for the candidate generator.
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
- Do not treat a predicted technology class as an observed attribute of a
  procurement, or the annotated class counts as market shares.
- Do not read the classifier's confidence as a probability of correctness
  without the reliability table in `TECHNOLOGY_TAXONOMY_REPORT.md`; it is
  conservative by a wide margin, and it is the **raw** score -- Platt scaling was
  evaluated and rejected by the pre-specified rule.
- Do not present the reference's recall figures or the `0.913` candidate-generation
  ceiling as method-independent. The labels are independent; the candidate list the
  reviewer saw came from an unrecorded rule. Precision is not affected.
- Do not quote a 12-quarter trend slope as a finding on its raw p-value alone.
  Five segment series are tested at once and the signal matrix carries Holm and
  Benjamini-Hochberg adjusted p-values beside the raw one.
