# VTC 2027 manuscript: blind disturbance-model selection

This is the new sheng study, separate from the earlier A4 manuscript.
The paper asks when a frozen receiver should approximate interference as noise
and when it should explicitly reconstruct the interfering latent. Both power
threshold directions are tested on development data; a fixed receiver is
retained if image-group validation does not support adaptation.

The source now contains the complete development, independent-confirmation and
pressure-test evidence. `scripts/build_vtc2027_evidence.py` generated the
numerical sections only after the policy freeze, paired-input checks and data
audit passed. The final manuscript was reviewed scientifically and visually on
2026-09-22. Author names, affiliations and correspondence remain explicit
placeholders and must be supplied before submission.

Build with a standard TeX Live installation:

```
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The reviewed manuscript is five pages. Figures are vector PDFs, with PNG
previews. The six
prospectively fixed qualitative examples are in `figures/qualitative_prespecified.pdf`
and in Fig. 3 of the paper; they were not selected for favorable results.

Reproduction files live in the project `scripts/` directory: `run_vtc2027.py`,
`prepare_vtc2027.py`, `analyze_vtc2027.py`, `audit_vtc2027_data.py`,
`build_vtc2027_evidence.py`, and `vtc2027_pipeline.py`. The frozen protocol and
its pre-confirmation development amendment are in `research/vtc2027_sheng/`.
All source, checkpoint, image and policy hashes are archived in result metadata.
Raw datasets and the five large checkpoints are not redistributed in this
manuscript source archive. This is a shared-weight receiver study, not a full
retraining reproduction of every cited publication.

The project also archives a separate no-fine-tuning public-checkpoint diagnostic
for DeepJSCC and two official MambaJSCC variants under
`research/pretrained_baselines/`. Its 27,648 records are explicitly labelled as
protocol-mismatched external anchors. They are not mixed with the matched
confirmation ranking; the public DeepJSCC checkpoint also uses substantially
more channel symbols than the locked 1/48-CBR protocol.

Target checked 2026-09-21: IEEE VTC 2027 Spring, Hamburg, 20–23 June 2027.
Official regular-paper deadline at that check: 30 September 2026.
https://events.vtsociety.org/vtc2027-spring/

No conference submission has been made.
