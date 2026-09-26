# VTC 2027 manuscript: blind disturbance-model selection

This is the new sheng study, separate from the earlier A4 manuscript.
The paper asks when a frozen receiver should approximate interference as noise
and when it should explicitly reconstruct the interfering latent. Both power
threshold directions are tested on development data; a fixed receiver is
retained if image-group validation does not support adaptation.

The source now contains the complete development, independent-confirmation and
pressure-test evidence, 18,432 matched-retrained DeepJSCC/MambaJSCC records,
and 27,648 public-weight diagnostic records. `scripts/build_vtc2027_evidence.py` generated the
numerical sections only after the policy freeze, paired-input checks and data
audit passed. The final manuscript was reviewed scientifically and visually on
2026-09-24. Author names, affiliations and correspondence remain explicit
placeholders and must be supplied before submission.

Build with a standard TeX Live installation:

```
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Table I and Figure 1 now include matched-retrained external codec anchors.
Section IV-B reports all three public-weight PSNR results separately. Training
uses one seed and a fixed 40-epoch budget, so these anchors do not establish
convergence or a general architecture ranking. Each matched anchor also trains
its own interference encoder; cross-backbone received tensors differ.

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

Target checked 2026-09-23: IEEE VTC 2027 Spring, Hamburg, 20–23 June 2027.
Official regular-paper deadline at that check: 30 September 2026.
https://events.vtsociety.org/vtc2027-spring/

No conference submission has been made.

Regenerate all numeric content from the project root (CPU only):

```sh
python3 -m scripts.build_vtc2027_evidence \
  --results research/vtc2027_sheng/completed_bundle/results \
  --paper paper/vtc2027_selective_sheng \
  --backbones research/backbone_baselines/confirmation \
  --pretrained research/pretrained_baselines/confirmation
```

`research/vtc2027_sheng/BASELINE_DELIVERY_AUDIT.json` records locked-input
checks; the optional server-file re-audit additionally checks training/validation
content hashes. Its completion status is recorded in the final review receipt.
`LOCAL_CHECKPOINT_RECOVERY.json` verifies all eight recovered best/latest weights. Raw confirmation records and
40-epoch logs are archived under `research/backbone_baselines/`. Large training
checkpoints remain on sheng under `research/backbone_baselines/formal/`; their
best/latest SHA256 identities are in each receipt. No original CDDM/ICDM
full-system replication or superiority is claimed.

The latest structural revision integrates prior work into the introduction,
which fills the remainder of page one after the abstract. Section II starts
on page two and defines the encoders, normalization, interference channel,
decoder and receiver decision; Section III presents the selection method.
All five figure/table captions are single sentences. Details and reading
sources are documented in `research/vtc2027_sheng/VTC_WRITING_REVIEW_ZH.md`.

The section-opening revision adds explicit “In this section” roadmaps to
Sections II–IV and starts the concise conclusion with “In this paper”.
The five-page layout was recompiled and visually checked on 2026-09-24.

The introduction now follows the user-provided problem–gap–insight–method–evidence argument, retaining the first-page boundary and five-page manuscript.

Section II now follows representation/transmission, diffusion-assisted recovery, and quality/cost/selection. It states the actual fixed-codec PSNR fitting objective; the candidate sampler implementations remain in Section III.

The 2026-09-26 prose revision follows sentence-organization patterns extracted
from the official CVPR 2025 VGGT paper: action-first clauses, stable paragraph
subjects, old-to-new information flow, short landing sentences after dense
relations, and goal--protocol--result--interpretation ordering in experiments.
These patterns were applied throughout the abstract, introduction, system
model, method, experiments and conclusion without copying VGGT wording or
changing scientific claims. The analysis and sentence-level mapping are in
`research/vtc2027_sheng/VGGT_SENTENCE_STYLE_ANALYSIS_ZH.md`.
