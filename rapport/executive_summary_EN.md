# Executive summary

> **Archived drafting source.** The authoritative English synthesis is
> `rapport/BOAMP_Report_EN_Overleaf/sections/00_synthesis_en.tex`. This Markdown
> version is retained for provenance and is not kept numerically synchronized.

**{SURNAME First name} — ENSAE second year — Applied internship 2025-2026**
**Gigalis — Analysis and modelling of digital public procurement using BOAMP data**

*This note is independent of the report and can be read on its own.*

---

## The setting

Gigalis is the public-interest group for digital services in the Pays de la Loire region. Among other roles it acts as a central purchasing body: it negotiates pooled contracts — cloud, cybersecurity, networks, artificial intelligence — that local authorities and public institutions may use, but are never obliged to. The value of those contracts depends on when they are opened. Open one too early and it sits unused; open it too late and buyers have already run their own procurements. Hence the question put to this internship: using public data, can we anticipate when a digital purchasing need will reappear?

## What we know

French public procurement notices are published in the BOAMP, an official bulletin that is freely accessible. This internship processed **1.6 million notices** published between 2015 and 2025 and derived a study population of **3,800 awarded digital contracts** in the Grand Ouest region.

One obstacle appeared immediately and shaped the entire project: **the BOAMP never records that one contract renews another**. There is therefore no list of renewals to learn from. The information had to be constructed: for each awarded contract, search the later notices published by the same buyer for the one that continues the same need, and accept the match only when the resemblance is strong. What the study measures is consequently an **"observable successor"** — a later procurement that visibly takes over — and not a legal renewal. The distinction is maintained throughout, because it changes what may legitimately be concluded.

## What the models indicate

On this basis, roughly **one contract in seven** shows an observable successor before the end of 2025. The probability that a successor becomes visible is estimated at **4.6 % within twelve months** of award and **6.7 % within twenty-four months**. These figures look low because most contracts are still too recent to have been re-procured — something the statistical method used here accounts for explicitly.

The most useful result is not that average level but its shape over time: the probability **rises markedly between the third and fourth year** of a contract, which matches the usual duration of public framework agreements. That is the window where monitoring pays off.

Two differences are solid. Contracts in **security and surveillance equipment** are re-procured appreciably faster than others. And **framework agreements** show a successor earlier — partly because the buyers who use them publish more and are therefore easier to track: some of the gap is visibility rather than purchasing behaviour.

The second contribution is of a different kind. Public contracts are classified under a European administrative vocabulary that is too coarse for business use: a single category mixes telephony, software and IT services. A model trained on **500 hand-annotated notices** learns to recover, from the wording of the contract object alone, an eight-family business segmentation — cloud, cybersecurity, networks, infrastructure, business software, data, artificial intelligence, IT services. It is **substantially better than the official vocabulary** for that purpose, and the gap is measured with its uncertainty. Gigalis therefore gains a technology-level reading that did not exist in the source data.

## What remains uncertain

Three points must be stated plainly.

**The quality of the matches is not independently validated.** It was measured against a reference sample built for the project, and on a very small number of accepted links. The order of magnitude is encouraging; the uncertainty interval is wide. A review by a procurement specialist is the missing control; the protocol and the sample are ready.

**Absolute levels depend on the rule chosen.** Depending on how demanding the matching rule is, the number of successors identified ranges from 296 to 1,332. Percentages must therefore never be quoted alone. By contrast — and this is the study's central finding — **comparisons between segments remain stable** when the rule changes. What is fragile is the level; what holds is the difference.

**Contract-by-contract prediction was not achieved.** A model was trained on the earlier years and tested on the recent ones without any refitting: its ability to rank contracts correctly is **no better than chance**. No individual score is therefore produced, and the "twenty contracts most likely to be renewed" table planned at the outset was not delivered. This is a result, not an implementation failure: it indicates that the information available at award time is not sufficient to predict an individual timetable.

## What this means for Gigalis

The work delivers **an instrument for measurement and monitoring, not a prediction engine**.

In practice it can say which *groups* of contracts are entering the period where a successor becomes likely — by segment and by age — and focus human attention on the three-to-four-year window, starting with the fastest-moving segments. It provides a technology segmentation of the regional portfolio that can be used to map activity. And it leaves behind a reproducible, documented and tested pipeline that can be re-run on updated data.

As it stands, it does not support claiming that a specific contract will be renewed, nor forecasting the evolution of a segment: over the observed period, no volume trend survives the appropriate statistical tests.

## What we recommend next

1. Have a sample of matches reviewed by an independent specialist before any quantified claim about the method's accuracy is communicated.
2. Have part of the technology corpus annotated by a second person, so that label reliability can be measured.
3. Provide Gigalis-internal membership data so that the causal question — does opening a pooled contract actually change members' purchasing behaviour? — can be addressed, which public data alone cannot do.
4. Treat individual prediction as a separate experiment, with its variables, validation protocol and success criterion fixed in advance, rather than as an adjustment of the current model.
