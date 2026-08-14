# Archived Regional Reference

These three CSV files are an early small regional reference built during this
project, not data supplied pre-filled by the internship program. The project
owner constructed it with LLM assistance, using real BOAMP notices, buyer
identities, CPV codes, dates, and external knowledge sources as evidence for
each labelling decision. It predates the linkage algorithms later developed
and evaluated in this repository.

The CSV schema carries fields for a two-reviewer-plus-adjudication protocol
(`reviewer_1_outcome`, `reviewer_2_outcome`, `adjudicated_outcome`), but those
fields are empty for every row and `annotation_status` is `NOT_STARTED`; only
`final_outcome` is populated, produced by a single evidence-based pass
(`review_method = SINGLE_EVIDENCE_REVIEW_SUPPLIED_BOAMP_CORPUS`). It is
therefore an early, single-pass, LLM-assisted reference grounded in real BOAMP
records, not an independent multi-reviewer human panel and not data received
from the internship program.

These files are retained only for provenance and historical reproducibility.

The canonical pipeline does not read this directory. All active linkage
calibration and evaluation uses `data/processed/boamp/benchmark/`, the national
benchmark with buyer-blocked dev, validation, and sealed-test splits.
