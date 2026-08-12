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

**Survival.** One row per contract. An accepted successor gives an event timed
to the successor's first notice; everything else is right-censored at the
cutoff. A censored contract is not a contract proven never to be renewed.

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
