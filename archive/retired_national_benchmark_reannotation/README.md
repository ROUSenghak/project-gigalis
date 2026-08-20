# Retired: France-level benchmark re-annotation packets

**Status: archive. Nothing in the active pipeline reads this directory.**

88 annotation assignment packets (`assignments_A/A001.json` … `A088.json`) and an
empty `submissions_A/`. They were generated on 2026-08-14 for a second
annotation pass over the **France-level (national) benchmark**, which was retired
on 2026-08-15 because both of its annotation passes were emitted by
deterministic rules built from the same text, CPV and date evidence the linkage
methods consume — so a method could score well on it only by agreeing with a
rule made from its own features.

The pass was never run: `submissions_A/` is empty and no labels were produced.
The generators (`pack_annotation_assignments.py`,
`prepare_annotation_batches.py`) were removed with the rest of the national
benchmark; they remain in git history.

These files lived under `scratchpad/`, which `.gitignore` designates as
disposable, and they are unreferenced by any script, notebook, test, report or
configuration. They are kept here rather than deleted so the provenance of the
retired benchmark's second-pass design is recoverable, and they are excluded
from version control by `.gitignore` because they are 18 MB of superseded
working material.

The **active** linkage reference is the Grand Ouest regional reference under
`data/reference/regional_link_benchmark/`. See `README.md` and
`REGIONAL_BENCHMARK_REFERENCE.md`.
