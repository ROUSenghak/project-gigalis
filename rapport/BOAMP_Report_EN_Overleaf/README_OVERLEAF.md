# BOAMP internship report — Overleaf project

**Entry file:** `main.tex`
**Compiler:** pdfLaTeX
**Bibliography:** BibTeX (`references.bib`, `natbib` with `plainnat`)
**Language:** English, with the mandatory French synthesis note at the end of the document

Upload this folder as a ZIP to Overleaf. Overleaf detects `main.tex` automatically; if it does not, set it
as the main document in *Menu → Main document*. The default Overleaf compiler (pdfLaTeX) and its automatic
BibTeX pass are all that is required — no custom `latexmkrc`.

To build locally instead:

```
python3 scripts/sync_final_report_figures.py  # from the repository root
cd rapport/BOAMP_Report_EN_Overleaf
latexmk -pdf main.tex
```

## Before you submit: four placeholders to fill

They are all in one place, near the end of the preamble in `main.tex`:

```latex
\newcommand{\studentName}{\textbf{[SURNAME First name]}}
\newcommand{\hostCity}{[City]}
\newcommand{\supervisorName}{[First name SURNAME]}
\newcommand{\internshipDates}{[start date] to [end date]}
```

Editing those four lines updates the cover page and the French synthesis note together.

## Two school requirements to check yourself

1. **Language authorisation.** The ENSAE instructions state that the report is normally written in French.
   English is permitted if the internship took place abroad, or if the working language was English or the
   host organisation requests it — with an email to `stage@ensae.fr` before **5 October 2026**. This report is
   in English at your request; confirm the authorisation, or plan the French translation, before submitting.
2. **Confidentiality.** The report is currently built as non-confidential, which matches the fact that every
   input is open BOAMP data. If Gigalis asks for confidentiality, add the mention to the cover box in
   `sections/00_cover.tex` and name the file `NOM_Prenom_2A25_CONF.pdf`.

Submission is one PDF containing the report, the annexes and both synthesis notes, named
`NOM_Prenom_2A25.pdf`, uploaded before **2 November 2026, 17:00**. This project produces exactly that single
PDF. The English executive summary sits before the body; the French `Note de synthèse` closes the document.

## Structure

```
main.tex                      entry file, preamble, document order, placeholders
references.bib                30 entries, all cited
sections/
  00_cover.tex                ENSAE cover layout
  00_synthesis_en.tex         English executive summary (non-specialist)
  01_introduction.tex         … 11_conclusion.tex   report body
  fig_pipeline.tex            TikZ diagram of the analytical chain (editable)
  99_synthesis_fr.tex         Note de synthèse in French (mandatory element)
appendices/
  A_notation.tex              notation, cohort rule, blocking rule, decision rule
  B_data_quality.tex          episode reconstruction, integrity checks, missingness
  C_linkage_validation.tex    reference, confusion matrices, threshold sweep, ROC/PR
  D_survival_detail.tex       full Cox output, PH diagnostics, parametric, sensitivity
  E_nlp_detail.tex            corpus, per-class metrics, calibration, downstream gates
  F_trend_detail.tex          signal matrix, PELT, ADF/KPSS, HMM, technology series
  G_reproducibility.tex       pipeline command, environment, validation gate, tests
figures/                      18 PNGs, all referenced; no unused assets
```

## Notes on the figures

Six figures carry the main text; the rest are diagnostics placed in the annexes. One was adjusted for
readability at report size, without touching any underlying result:

- Thirteen figures are copied from the canonical `reports/figures/` outputs by
  `scripts/sync_final_report_figures.py` before a local build.

- `fig05_threshold_left_panel.png` is a crop of the left panel of the three-panel threshold figure. The full
  version appears in Annex C, so nothing is hidden — the crop exists because the three-panel original is
  unreadable at half-page size.
Figure 2 (the analytical chain) is TikZ rather than an image, so you can edit it directly in
`sections/fig_pipeline.tex` — useful if you want to reuse it on a defence slide.
