"""One-shot server-side completion, independent of the local Mac staying awake."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1]
STATUS=ROOT/'research/vtc2027_sheng/server_delivery_status.json'


def record(stage,**kw):
    d=dict(stage=stage,updated_at=time.time(),**kw)
    tmp=STATUS.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2));tmp.replace(STATUS)
    print(json.dumps(d),flush=True)


def main():
    try:
        record('awaiting_experiments')
        deadline=time.time()+36*3600
        while time.time()<deadline:
            p=json.loads((ROOT/'research/vtc2027_sheng/pipeline_status.json').read_text())
            if p['stage']=='failed': raise RuntimeError(p.get('error','Experiment failed'))
            if p['stage']=='experiment_complete': break
            time.sleep(30)
        else: raise TimeoutError('Experiment completion timeout')
        paper=ROOT/'paper/vtc2027_selective_sheng'
        record('building_evidence')
        subprocess.run([sys.executable,'-m','scripts.build_vtc2027_evidence','--results',str(ROOT/'results'),
                        '--paper',str(paper)],cwd=ROOT,check=True)
        record('compiling')
        build=paper/'build_sheng';build.mkdir(exist_ok=True)
        env=dict(os.environ,TECTONIC_CACHE_DIR=str(ROOT/'tex_cache'))
        with (build/'compiler_output.txt').open('w') as log:
            subprocess.run([str(ROOT/'bin/tectonic'),'--keep-logs','--keep-intermediates','--outdir',str(build),
                            str(paper/'main.tex')],cwd=paper,env=env,check=True,stdout=log,stderr=subprocess.STDOUT)
        audit=json.loads(subprocess.check_output([sys.executable,str(ROOT/'scripts/audit_vtc2027_pdf.py'),str(build)],text=True))
        output=ROOT/'output';output.mkdir(exist_ok=True)
        shutil.copy2(build/'main.pdf',output/'VTC2027_Sheng_Manuscript.pdf')
        shutil.copy2(paper/'generated/evidence_receipt.json',output/'evidence_receipt.json')
        with zipfile.ZipFile(output/'VTC2027_Sheng_LaTeX.zip','w',zipfile.ZIP_DEFLATED) as z:
            for name in ['main.tex','references.bib','README.md']: z.write(paper/name,arcname=name)
            for folder in ['generated','figures']:
                for path in (paper/folder).rglob('*'):
                    if path.is_file(): z.write(path,arcname=str(path.relative_to(paper)))
        record('manuscript_assembled',pdf_audit=audit,
               note='All evidence complete; full scientific/visual final review and author information still required')
    except Exception as exc:
        record('failed',error=repr(exc));raise


if __name__=='__main__': main()
