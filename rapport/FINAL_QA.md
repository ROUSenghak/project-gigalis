# Final scientific QA

## Confirmed or partly confirmed concerns

- **B--C:** Candidate-pool size varies within 473/514 multi-episode buyers (92.0%). Buyer stratification and direct pool-size adjustment now have distinct interpretations. The out-of-time C-index remains 0.479; its buyer-cluster bootstrap interval is [0.360, 0.634], so the report says useful individual discrimination is not established, not ruled out.
- **D:** The 90-day floor appears in the 11 August 2026 project history, but later diagnostics used all 23 reviewed successors, and an earlier evaluation script states that the retained 0.70 operating point drew on the designated locked evidence. The report therefore describes the linkage assessment as internal validation, not a fully untouched held-out test.
- **E--M:** The false EPV argument was removed; ROC/PR, re-procurement terminology, KM censoring, CPV-35, CPV-48, framework robustness, and the absolute-versus-comparative robustness message were corrected. Episode and buyer-cluster KM intervals are both reported. CPV and technology trends now share 2015Q2--2025Q4.
- **N--P, R, V:** Figure 17 was regenerated with both downstream gates stated correctly. Analytical-unit and CamemBERT wording, the annotation limitation, and the evidence hierarchy were tightened in the body, appendices, English synthesis, French synthesis, discussion, and conclusion.
- **W:** Citation coverage is complete (30 cited entries; no missing or unused entries). A malformed procurement-NLP entry was corrected, a weak procurement-data citation was removed, and authoritative Harrell, Lin--Wei, ADF, KPSS, and TF-IDF references were added.

## Concerns not confirmed

- **Q:** No further NLP tuning was warranted or performed. The nested group-aware result and full 11-class OOF macro-F1 of 0.744 remain the headline.
- **S:** The implemented Gate A/Gate B membership was already correct; the defect was the stale Figure 17 subtitle, now fixed.
- **T--U:** The report did not require a new causal technology claim or additional time-series model. Technology survival remains secondary/descriptive, and PELT/HMM remain non-causal monitoring tools.
- **X:** No reliable project source supplied the four cover facts, so they were not invented.

## Recomputed artifacts

- Buyer-stratified Cox sensitivities for all four linkage arms, within-buyer pool-variation diagnostics, episode and buyer-cluster KM intervals, and fixed-prediction episode/buyer-cluster C-index intervals.
- Harmonised 2015Q2--2025Q4 technology series and last-12-quarter slopes with multiplicity adjustment; no technology slope is significant after correction.
- Technology Kaplan--Meier Figure 17.

## Validation and production

- Tests: **139 passed**.
- Canonical-state validation: **passed** (all checks true).
- Bibliography/cross-references: resolved; final LaTeX log has no warnings or overflow messages.
- Visual QA: all **61 A4 pages** inspected after final compilation; no clipping, stale correction notes, or contradictory captions found.

## Unresolved items

- Administrative placeholders: student name, host city, supervisor name, and internship dates. Confirm confidentiality status and obtain/confirm ENSAE authorisation for an English submission; rename the submission PDF to the prescribed student-specific filename.
- Scientific: no verified legal-renewal ground truth; linkage evidence is internal and not independently specialist-validated; technology labels are single-annotator with no agreement statistic and very thin rare classes; buyer dependence and constructed-event/classifier error remain; individual Cox ranking, causal segment effects, technology market shares, and monetary claims are not established.
