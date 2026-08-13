# Linkage Challenge Review Results

Generated: `2026-08-14T00:18:07`

Status: **model-assisted diagnostic complete; independent human validation pending**

## Accepted-Link Diagnostic

Among 20 sampled links accepted by `M_B @ 0.70`:

- `Y`: `14`;
- `N`: `5`;
- `UNCERTAIN`: `1`.

Precision excluding the uncertain row is `0.7368`
(`14/19`), with exact 95% CI
`[0.4880, 0.9085]`.

Treating uncertainty conservatively as not confirmed gives precision
`0.7000` (`14/20`),
with exact 95% CI `[0.4572, 0.8811]`.

The diagnostic point estimate does not meet the predefined `0.80` target.

## Challenge Diagnostics

- Structural-negative challenges: `19` reviewed
  `N`, `1` reviewed `Y`, and
  `0` uncertain.
- Buyer-declared relationships: `6` reviewed `Y`,
  `14` reviewed `N`, and
  `0` uncertain.

The structural contradiction demonstrates that rule-generated structural
negatives are challenge cases, not verified ground truth. Buyer declarations are
also recruitment evidence rather than automatic successor labels. Neither
stratum estimates national recall, prevalence, or false-positive rate.

## Findings Queue

`7` accepted-link non-confirmations or structural contradictions
are listed in `data/review/review_audit_findings.csv`.

## Decision

Keep `M_B @ 0.70` frozen and provisional. These labels are documented as a
model-assisted diagnostic, so they cannot complete the independent validation
gate. An independent human procurement-domain reviewer should assess the
original blinded file before recalibration or final accuracy claims.
