"""Server-side DeepJSCC/MambaJSCC retraining, confirmation, and paper assembly."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "research/backbone_baselines"
STATUS = BASE / "pipeline_status.json"
PYTHON = Path(sys.executable)


def record(stage: str, **values) -> None:
    BASE.mkdir(parents=True, exist_ok=True)
    value = {"stage": stage, "updated_at": time.time(), **values}
    tmp = STATUS.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(STATUS)
    print(json.dumps(value, allow_nan=False), flush=True)


def run_logged(stage: str, command: list[str], log_name: str) -> None:
    logs = BASE / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    record(stage, command=command, log=str(logs / log_name))
    with (logs / log_name).open("a") as stream:
        stream.write("\nCOMMAND " + json.dumps(command) + "\n")
        stream.flush()
        subprocess.run(command, cwd=ROOT, check=True, stdout=stream, stderr=subprocess.STDOUT)


def train(architecture: str, role: str, batch_size: int) -> None:
    output = BASE / "formal" / f"{architecture}_{role}"
    receipt = output / "receipt.json"
    if receipt.exists() and json.loads(receipt.read_text()).get("status") == "complete":
        record(f"skip_complete_{architecture}_{role}", receipt=str(receipt))
        return
    run_logged(
        f"training_{architecture}_{role}",
        [str(PYTHON), "scripts/train_backbone_baseline.py",
         "--architecture", architecture, "--role", role,
         "--external-root", f"external_baselines/{architecture}",
         "--data-root", "/mnt/d/ICDM_SC_data_full", "--output", str(output),
         "--epochs", "40", "--batch-size", str(batch_size), "--workers", "8",
         "--validation-images", "1024", "--snr", "20", "--seed", "20260922"],
        f"train_{architecture}_{role}.log",
    )


def main() -> None:
    try:
        record("starting", gpu="NVIDIA GeForce RTX 4090", note="No selector thresholds are changed")
        for architecture, batch in [("deepjscc", 256), ("mambajscc", 32)]:
            for role in ["source", "interference"]:
                train(architecture, role, batch)
        confirmation = BASE / "confirmation"
        meta = confirmation / "metadata.json"
        if not (meta.exists() and json.loads(meta.read_text()).get("status") == "complete"):
            run_logged(
                "confirming_backbones",
                [str(PYTHON), "scripts/evaluate_backbone_baselines.py",
                 "--manifest", "configs/vtc_confirmation.json", "--output", str(confirmation),
                 "--deepjscc-root", "external_baselines/deepjscc",
                 "--mambajscc-root", "external_baselines/mambajscc",
                 "--deepjscc-source", str(BASE / "formal/deepjscc_source/best.pt"),
                 "--deepjscc-interference", str(BASE / "formal/deepjscc_interference/best.pt"),
                 "--mambajscc-source", str(BASE / "formal/mambajscc_source/best.pt"),
                 "--mambajscc-interference", str(BASE / "formal/mambajscc_interference/best.pt"),
                 "--resume"],
                "confirmation.log",
            )
        run_logged(
            "analyzing_backbones",
            [str(PYTHON), "scripts/analyze_backbone_baselines.py", "--run", str(confirmation),
             "--reference", "results/confirmation"],
            "analysis.log",
        )
        paper = ROOT / "paper/vtc2027_selective_sheng"
        run_logged(
            "rebuilding_evidence",
            [str(PYTHON), "-m", "scripts.build_vtc2027_evidence", "--results", "results",
             "--paper", str(paper), "--backbones", str(confirmation)],
            "paper_evidence.log",
        )
        build = paper / "build_backbones"
        build.mkdir(exist_ok=True)
        env = dict(os.environ, TECTONIC_CACHE_DIR=str(ROOT / "tex_cache"))
        record("compiling_paper", build=str(build))
        with (build / "compiler_output.txt").open("w") as log:
            subprocess.run([str(ROOT / "bin/tectonic"), "--keep-logs", "--keep-intermediates",
                            "--outdir", str(build), str(paper / "main.tex")], cwd=paper,
                           env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
        audit = json.loads(subprocess.check_output(
            [str(PYTHON), "scripts/audit_vtc2027_pdf.py", str(build)], cwd=ROOT, text=True))
        output = ROOT / "output/vtc2027_sheng"
        output.mkdir(parents=True, exist_ok=True)
        pdf = output / "VTC2027_Sheng_Manuscript_BackboneAnchors.pdf"
        archive = output / "VTC2027_Sheng_LaTeX_BackboneAnchors.zip"
        shutil.copy2(build / "main.pdf", pdf)
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
            for name in ["main.tex", "references.bib", "README.md"]:
                bundle.write(paper / name, arcname=name)
            for folder in ["generated", "figures"]:
                for path in (paper / folder).rglob("*"):
                    if path.is_file():
                        bundle.write(path, arcname=str(path.relative_to(paper)))
        record("complete_pending_human_review", pdf=str(pdf), archive=str(archive), pdf_audit=audit,
               expected_confirmation_records=18_432,
               note="Scientific and page-by-page visual review required before delivery")
    except Exception as error:
        record("failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
