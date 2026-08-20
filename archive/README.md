# Archive — retained for provenance, read by nothing

Everything under this directory is **historical**. No script, notebook, test,
report generator or configuration in the active pipeline reads any of it, and
`scripts/validate_canonical_state.py` fails the build if an executable source
ever starts to. `pytest.ini` restricts test collection to `tests/` for the same
reason.

| Directory | What it is | Superseded by |
|---|---|---|
| `full_raw_eda/` | Materialised output of the 2026-08-11 raw-data EDA — figures, tables, `eda_run_summary.json`. Generated on a different machine before the canonical pipeline existed. | `DATA_QUALITY_REPORT.md`, `data/processed/boamp/data_quality_profile.json` |
| `retired_national_benchmark_reannotation/` | 88 annotation packets prepared on 2026-08-14 for a second pass over the retired France-level benchmark. The pass was never run; no labels were produced. Excluded from version control by `.gitignore` (18 MB of superseded working material). | The Grand Ouest regional reference at `data/reference/regional_link_benchmark/` |

Nine early exploration notebooks are archived separately under
`notebooks/archive/`, with their own README.

The France-level benchmark itself — its data, construction scripts, annotation
tooling and the library modules that served only it — was **deleted**, not
archived, because its labels were emitted by deterministic rules built from the
same text, CPV and date evidence the linkage methods consume, so scoring well on
it meant only agreeing with a hand-written rule made from a method's own
features. Its history remains in version control. See `README.md` §"The Linkage
Reference, And Why The Other Was Retired".

Do not cite anything in this directory as a current result.
