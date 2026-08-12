# BOAMP Raw Data Acquisition

Download the historical BOAMP raw corpus with:

```bash
python3 scripts/download_boamp.py
```

This retrieves one JSONL file per complete calendar year for:

```text
2015-01-01 <= dateparution < 2026-01-01
```

Optional current-year partial data is kept separate:

```bash
python3 scripts/download_boamp.py --include-2026
```

Force a redownload of already completed years:

```bash
python3 scripts/download_boamp.py --force
```

Inspect API metadata/counts without downloading raw records:

```bash
python3 scripts/download_boamp.py --inspect-only
```

Raw BOAMP files and logs are ignored by git. Metadata is written to
`data/metadata/`.

## Schema-aware notice and episode layer

The current study pipeline deliberately skips the optional 2026 technology
classification export. Build the 2015-2025 standardized notice layer with:

```bash
python3 scripts/build_standardized_notices.py
```

This preserves every raw BOAMP field and adds schema-aware buyer, buyer
geography, CPV, typed duration, typed amount, text, and provenance fields. The
Grand Ouest subset is based on the contracting buyer's structured postal
geography.

Then reconstruct procurement episodes conservatively:

```bash
python3 scripts/build_procurement_episodes.py
```

Episode edges use, in priority order, `contractfolderid`, explicit linked
notice IDs, and exact procedure reference within the same buyer and a 730-day
adjacency window. Reference fallback also requires an exact normalized buyer
name and excludes the redundant eForms folder reference. Trusted buyer
conflicts reject an edge. This stage groups
notices from one procedure; it does not infer renewals.

Finally, execute the evidence audit:

```bash
python3 -m jupyter nbconvert --execute --to notebook --inplace \
  notebooks/10_standardized_notice_and_episode_evidence_audit.ipynb \
  --ExecutePreprocessor.timeout=0
```

Use `--force` with either build script only when intentionally replacing its
existing outputs.

## Benchmark remapping to v2 episode IDs

The manual benchmark (`data/reference/BOAMP_Internship_Reference_120.csv`,
`BOAMP_Internship_Evaluation_Subset.csv`,
`BOAMP_Internship_Confirmed_Successor_Links.csv`) was annotated against a
superseded Grand Ouest episode reconstruction, so its `anchor_episode_id` and
`successor_episode_id` values do not exist in the current v2 episode layer.
The underlying notice IDs (idweb) are stable, so re-derive the current
episode ID for every benchmark anchor and confirmed successor with:

```bash
python3 scripts/remap_benchmark_to_v2_episodes.py
```

This joins each benchmark notice ID against `episode_membership.parquet`
rather than trusting the stale episode ID. A benchmark anchor whose notices
land in more than one v2 episode is flagged as `split_across_episodes`
instead of being silently resolved to one side. Outputs are written to
`data/processed/boamp_v2/benchmark_remap/`.

## Study cohort, linkage, and survival analysis

These four steps run in order and produce the final study results.

```bash
python3 scripts/build_survival_cohort.py      # 3,800 digital Grand Ouest contracts
python3 scripts/build_linkage_candidates.py   # candidate successors + features
python3 scripts/fit_fellegi_sunter.py         # probabilistic model (comparison arm)
python3 scripts/evaluate_linkage.py           # benchmark evaluation, freezes the policy
python3 scripts/build_survival_dataset.py     # right-censored survival records
```

**Cohort.** A contract is a Grand Ouest episode with a digital CPV (divisions
32, 35, 48, 72) that contains an award notice. The award notice both proves a
contract exists and supplies time zero. The study cutoff is 2025-12-31.

**Linkage.** Candidates are later Grand Ouest episodes from the same buyer
(exact `buyer_key` or exact normalised name), published between 90 and 2,920
days after the award. The 90-day floor keeps concurrent lots of the same
programme from being mistaken for succession; it discards none of the manually
confirmed links, whose shortest gap is 139 days. Four components are
scored — buyer, text, CPV, time — and components that are genuinely
unobservable are excluded from the score rather than counted as disagreement,
so a missing CPV never reads as a category mismatch and a missing duration is
never replaced by an assumed contract length.

**Four methods are compared**, then one is frozen: a deterministic SIREN+CPV
rule, TF-IDF text ranking on the same-buyer pool, the weighted-and-gated score
ported from notebook 08, and Fellegi-Sunter probabilistic linkage whose weights
are estimated by expectation maximisation rather than chosen by hand. The
adoption rule was fixed before the probabilistic model was fitted: it replaces
the incumbent only if it weakly dominates on the locked split. It does not, so
`M_B_text_ranking @ 70` stands. The estimated weights are reported anyway,
because they independently confirm which signals carry the discriminating
power — and because they show *why* unsupervised probabilistic linkage fails
here: true successors are about 0.06% of candidate pairs, so the two-class
mixture converges on text-similarity and documentation-quality structure
instead.

**Survival.** One row per contract. An accepted successor gives an event timed
to the successor's first notice; everything else is right-censored at the
cutoff. A censored contract is not a contract proven never to be renewed.

### Parallel expiry-aware sensitivity

The expiry-aware policy is a parallel sensitivity arm; it does not overwrite
the frozen v1 linkage or survival outputs.

```bash
PYTHONPATH=. python3 scripts/evaluate_expiry_aware_linkage.py
PYTHONPATH=. python3 scripts/build_expiry_aware_survival_dataset.py
```

Candidate generation stays unchanged. `M_E_expiry_aware_text` retains the
normal text threshold of 0.70, but a candidate published more than 365 days
before the best observable expected end must also have text similarity of at
least 0.85 and positive CPV continuity. Expected end uses, in order, one
unambiguous explicit end date, one start date plus reliable duration, or award
date plus reliable duration; missing duration is not imputed.

On the current artifacts, this policy accepts 507 observable successors versus
547 under v1 and sends 42 changed anchors to `expiry_link_review.csv`. Its
reviewed-benchmark metrics are unchanged from v1 (precision 0.80, recall 0.3636,
zero false positives on reviewed negative anchors). The automated promotion
gate therefore passes, but the policy remains parallel until the changed links
are manually reviewed. Outputs are
`accepted_successor_links_expiry_aware.parquet`,
`expiry_aware_linkage_summary.json`, `expiry_link_review.csv`,
`survival_dataset_expiry_aware.parquet`, and
`survival_dataset_expiry_aware_summary.json`.

The operational output is `data/processed/boamp_v2/renewal_watchlist_top20.csv`
— the five nearest-term contracts in each digital segment, ranked by the
probability that a successor becomes visible within twelve months. It is
stratified deliberately: the probability is a function of segment and contract
age, so an unstratified ranking collapses onto whichever segment has the highest
baseline hazard. Treat it as a prioritisation aid, not a per-contract forecast.

Then execute the three evidence notebooks:

```bash
python3 -m jupyter nbconvert --execute --to notebook --inplace \
  notebooks/11_cohort_and_data_quality.ipynb \
  notebooks/12_successor_linkage_and_evaluation.ipynb \
  notebooks/13_survival_analysis.ipynb \
  --ExecutePreprocessor.timeout=0
```

Each notebook ends with an **Evidence / Decision** section recording what was
tested, what the numbers were, what remains uncertain, and what was frozen.

### Two limitations that must travel with any quoted result

The manual benchmark was built by reviewing only the *previous* pipeline's
top-25 candidates per anchor, so a successor that ranker buried was never
available to be labelled. Measured recall is therefore optimistic, and
"no observed successor" means *not found in that review*, not *absent from
BOAMP*. Separately, the locked test is not an untouched holdout: notebook 07
inspected its error counts and feature medians before the gated method was
designed in notebook 08. Locked-test metrics are already-inspected evaluation
evidence.

Throughout the code and notebooks an accepted link is called an **observable
successor procurement**, never a confirmed legal renewal.
