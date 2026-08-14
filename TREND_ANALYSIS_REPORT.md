# BOAMP Descriptive Trend Analysis

Generated: `2026-08-14T11:16:50`  
Analysis window: `2015Q2-2025Q4`  
Unit: awarded Grand Ouest digital procurement episodes

## Technical Summary

This analysis adds the guide's missing time-series component without claiming a forecast. Quarterly episode counts are examined for the overall cohort and CPV divisions 32, 35, 48, and 72. PELT identifies candidate mean shifts, penalty sensitivity distinguishes stable from fragile breaks, and a 12-quarter linear slope describes the current direction.

The results are descriptive signals only. Breaks are not automatically attributed to policy, technology, or COVID; those explanations require documentary evidence and stakeholder validation. Amount trends are omitted because the current episode layer has multiple unvalidated amount candidates rather than one canonical awarded amount.

## Current Signal Matrix

| Segment | Recent direction | Episodes/quarter slope | Exploratory p-value | Last stable PELT break |
|---|---|---:|---:|---|
| Overall | stable_or_uncertain | -0.11 | 0.921 | -- |
| CPV-32 | stable_or_uncertain | -0.01 | 0.989 | 2020Q2 |
| CPV-35 | stable_or_uncertain | 0.03 | 0.923 | -- |
| CPV-48 | decreasing | -0.84 | 0.032 | 2024Q1 |
| CPV-72 | stable_or_uncertain | 0.70 | 0.285 | 2021Q1 |

`stable_or_uncertain` means the 12-quarter slope is not distinguishable from zero at the pre-declared exploratory level α = 0.10. These p-values are descriptive and are not corrected for multiple testing.

![Quarterly episode counts](reports/figures/trend_quarterly_episode_counts.png)

## Stationarity (ADF/KPSS)

| Segment | ADF (H0: unit root) | KPSS (H0: level stationary) | Note |
|---|---|---|---|
| Overall | ADF stat=-5.890, p=0.000 | KPSS stat=0.268, p=0.100 | |
| CPV-32 | ADF stat=-6.086, p=0.000 | KPSS stat=0.808, p=0.010 | |
| CPV-35 | ADF stat=-1.684, p=0.439 | KPSS stat=0.444, p=0.058 | |
| CPV-48 | ADF stat=-1.749, p=0.406 | KPSS stat=0.545, p=0.032 | |
| CPV-72 | ADF stat=-5.439, p=0.000 | KPSS stat=0.611, p=0.022 | |

ADF and KPSS test opposite null hypotheses, so they are read together rather than
individually. A series that rejects the ADF unit-root null while failing to reject
the KPSS stationarity null is consistent with level stationarity around a constant
or slowly varying mean; disagreement between the two tests indicates the series is
not cleanly classified as stationary or non-stationary over this short window. These
diagnostics describe the fitted quarterly series; they are not used to justify or
rule out forecasting, which remains out of scope.

## Regime Detection (HMM)

A 3-state Gaussian hidden Markov model is fit on the quarter-over-quarter change in
episode count (not the level) for the overall cohort and the two highest-volume CPV
segments, so the three states describe typical period-over-period direction —
`decline`, `plateau`, `growth` — rather than the segment's absolute activity level. A
segment can hold a high or low count while still sitting in a `plateau` regime if its
recent changes are small. The reported probability is the model's posterior
probability of the current-quarter regime, not a forecast, and a detected regime is
not a causal explanation of any prior shift.

| Segment | Current regime | Probability | Mean quarterly change by regime |
|---|---|---:|---|
| Overall | growth | 0.750 | decline=-15.5, plateau=-7.8, growth=17.8 |
| CPV-32 | growth | 0.992 | decline=-12.3, plateau=-0.5, growth=11.6 |
| CPV-72 | growth | 0.594 | decline=-6.6, plateau=0.1, growth=7.5 |

Because the model is fit on noisy, low-count quarterly series, the `plateau` state is
a data-driven middle tier rather than a change centered exactly at zero; its mean
change should be read alongside `decline` and `growth` rather than interpreted as
"no change." The HMM's current-regime label and the 12-quarter OLS slope above are
complementary, not identical: the OLS slope summarizes the last 12 quarters, while
the HMM regime reflects the model's belief about the most recent quarter's state and
can differ from the OLS signal without either being wrong.

## Method

For segment $s$ and quarter $q$, the count is

\[
N_{s,q}=\sum_i \mathbf{1}(S_i=s, Q_i=q),
\]

where $S_i$ is the episode's CPV division and $Q_i$ is its award quarter. The partial first quarter of 2015 is excluded; all quarters from 2015Q2 onward are represented, including zeros.

PELT minimizes a penalized segmentation objective,

\[
\sum_{r=0}^m \mathcal{C}(y_{\tau_r+1:\tau_{r+1}})+\beta m,
\]

where \(\mathcal{C}\) is within-segment squared error, \(m\) is the number of breaks, and \(\beta=\lambda\log(n)\) after z-standardization. The central result uses \(\lambda=1\); sensitivity uses 0.5 and 2.0. A break is called stable only when a break lies within one quarter under all three penalties. This follows the PELT framework of [Killick, Fearnhead and Eckley (2012)](https://doi.org/10.1080/01621459.2012.737745).

The recent direction comes from an ordinary least-squares slope over the latest 12 quarters. It is `increasing` or `decreasing` only when its two-sided p-value is below 0.10; otherwise it is `stable_or_uncertain`. This is a signal description, not a forward prediction.

## Duration Completeness Is A Measurement Break

![Duration completeness](reports/figures/trend_duration_completeness.png)

The sharp rise in duration availability in 2025 is a schema/completeness change, not evidence that contract durations suddenly changed. Median-duration trend claims would mix periods with substantially different observation processes, so the report does not run change-point detection on duration values.

## Limits

- The series cover awarded digital episodes, not every procurement notice or every French contract.
- CPV divisions are broad operational segments, not the 8-12 class supervised taxonomy proposed in the internship guide.
- Count changes can reflect publication practice, schema changes, buyer coverage, or procurement activity.
- PELT proposes candidate breaks; it does not identify their causes.
- Monetary trends remain unavailable until one awarded-value definition is validated.

## Reproducible Outputs

- `data/processed/boamp/trend_quarterly.csv`
- `data/processed/boamp/trend_breakpoints.csv`
- `data/processed/boamp/trend_signal_matrix.csv`
- `data/processed/boamp/trend_analysis_summary.json`
- `notebooks/14_data_quality_and_trend_analysis.ipynb`
