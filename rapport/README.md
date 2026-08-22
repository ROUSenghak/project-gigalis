# Final internship report

The authoritative report source is `BOAMP_Report_EN_Overleaf/`. The final
reader copy is `BOAMP_Report_EN_Final.pdf`, and
`BOAMP_Report_EN_Final_Overleaf.zip` is the clean upload package. Rebuild from
the source directory with:

```bash
python3 scripts/sync_final_report_figures.py
cd rapport/BOAMP_Report_EN_Overleaf
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The Markdown report and synthesis files in this directory are earlier drafting
sources retained for provenance. They are explicitly marked as archived and are
not authoritative. `FINAL_QA.md`, `audit_coherence_et_revendications.md`,
`BUYER_CLUSTERING_ANALYSIS.md`, `VALIDATION_STUDY_DESIGN.md`, and
`legendes_figures_et_tableaux.md` are supporting audit/design notes, not
competing reports.

The figure synchronisation script records every direct mapping from a canonical
pipeline plot to its reader-facing report filename. Four report-specific
figures remain intentionally local: the selection funnel, linkage-sensitivity
summary, parametric-fit comparison, and threshold-panel crop. They are retained
because they are report layouts derived from the same published tables rather
than competing result artifacts.

Before external submission, replace the four administrative placeholders in
`BOAMP_Report_EN_Overleaf/main.tex`: student name, host city, supervisor name,
and internship dates. Confirm confidentiality status and ENSAE authorisation
for an English-language submission, then rename the PDF to the prescribed
student-specific filename. No scientific result depends on those fields.
