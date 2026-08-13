# Benchmark v3 completion status

Last checked: 2026-08-13 17:55 CEST.

## Completed

- [x] P0-P7 benchmark machinery: national index, sampling frame, declaration
  mining, predecessor resolution, anchor sampling, blinded exposure, structural
  negatives, sealing, and pipeline integration.
- [x] All existing gates pass and the full test suite passes.
- [x] Wave 1 Pass A canonical dossiers: 253 anchors and 7,050 candidate
  judgments in 26 blinded source batches.
- [x] Worker-size restart pack: 87 assignments in
  `scratchpad/w1a/assignments/`, capped at 90 judgments while keeping each
  anchor dossier intact.
- [x] Strict coverage gate: ingest now rejects missing, duplicated, and unknown
  dossier submissions.
- [x] Pass B gate: Pass B cannot be prepared until validated Pass A labels cover
  every currently prepared Pass A anchor.
- [x] Wave 1 Pass A AI-assisted draft annotation: 253 anchors and 7,050 labels
  ingested with zero schema rejections.
- [x] Wave 1 Pass B AI-assisted draft annotation: 253 anchors and 7,050 labels
  ingested with zero schema rejections.
- [x] Adjudication: 7,050 settled labels, no disagreement queue. Kappa is
  same-rule self-consistency and must not be described as human agreement.
- [x] Buyer-blocked v3 splits rebuilt: dev 138 anchors, validation 59 anchors,
  sealed test 56 anchors.
- [x] v3 dev and validation linkage summaries regenerated for the primary
  event set without opening the sealed test.

## Blocked

- [ ] Human or independent expert review is still missing. The current labels
  are AI-assisted draft labels generated from blinded dossier fields and should
  be used for method development with a prominent caveat, not as final human
  ground truth.

## After Pass A

- [ ] Decide whether to accept AI-assisted labels as a development benchmark or
  route high-impact positives/changed decisions to human review.
- [ ] If used for method selection, treat the generated v3 dev/validation
  numbers as AI-assisted evidence and keep the sealed test closed.
- [ ] Recalibrate thresholds on v3 dev only, then select a method on validation.
  Do not open the sealed test during method selection.

## Evidence Already Usable

- 347 buyer-declared strong/medium predecessor pairs were resolved without a
  text fallback. Their median award-to-successor gap is 895 days. Among the 64
  pairs with a usable expected end, 70.31% were published before expected end
  and 29.69% more than 365 days early. These are market-timing observations,
  not annotated renewal labels.
- The separate hard-negative suite contains 2,443 verified pairs. Its 800
  parallel-lot pairs have median text similarity 0.75 and its 520
  same-procurement pairs have median similarity 1.00. This suite is an
  evaluation stress test and is not pooled into the national estimate.
