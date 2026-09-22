# IEEE VTC 2027 LaTeX draft

The manuscript uses the standard IEEE conference class with conference mode.

Rewritten on 2026-09-19 after a section-level study of seven accessible VTC
papers. The current five-section manuscript is main.tex. The previous LaTeX
version is preserved in main_before_vtc_rewrite.tex; the older Markdown draft
is historical and is not the current manuscript.

The source study, version caveats, section counts, and editorial rationale
are in ../../research/vtc_section_study/VTC_SECTION_ANALYSIS.md.
External-paper counts use an approximate PDF lexical measure, not TeXcount.
The original PDFs and reproducible counts are kept alongside that report.

Files:

- main.tex: manuscript source
- references.bib: verified bibliography entries
- build/main.pdf: compiled paper
- figures/fig_psnr.pdf: PSNR curves from the raw paired records
- figures/fig_calibration.pdf: paired gain with image-cluster intervals
- figures/gen_fig_calibration.py: reproducible generation of both figures

The author block is intentionally left as a placeholder. Replace it before submission.

The numerical results are the completed A0--A4 exploratory results. Before submission, replace them with the locked independent confirmation and add LPIPS. Do not present A5 as a contribution unless its validation and confirmation results support it.

The added likelihood, least-squares derivation, guidance equations, and
inference procedure describe the existing A4 implementation. No new
experiment, bypass rule, amplitude cap, or covariance method was introduced.

The official VTC2027-Spring call requests a five-page full paper. Up to two additional pages are allowed with overlength charges. Recompile this draft after final figures, author details, and confirmatory results are inserted.
