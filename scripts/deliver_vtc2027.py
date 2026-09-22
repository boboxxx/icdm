"""Keep this one authorized job moving through data retrieval and manuscript build."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'research/vtc2027_sheng/delivery_status.json'
HOST='sheng@100.94.183.27'
REMOTE='/home/sheng/ICDM_VTC2027_20260921'
LATEX=Path('/Users/chen/.codex/plugins/cache/openai-bundled/latex/0.2.7')
PDFPY='/Users/chen/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3'
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',HOST]


def state(stage,**kwargs):
    value=dict(stage=stage,updated_at=time.time(),**kwargs)
    tmp=STATE.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(STATE)
    print(json.dumps(value),flush=True)


def command(args,**kwargs):
    return subprocess.check_output(args,text=True,**kwargs)


def retrieve():
    remote_archive=REMOTE+'/results/vtc2027_completed_bundle.tar.gz'
    members=['results/development','results/confirmation','results/pressure',
             'research/vtc2027_sheng/data_audit.json','research/vtc2027_sheng/pipeline_status.json']
    command(SSH+['cd '+shlex.quote(REMOTE)+' && tar -czf '+shlex.quote(remote_archive)+' '+' '.join(shlex.quote(x) for x in members)])
    dest=ROOT/'research/vtc2027_sheng/completed_bundle';dest.mkdir(parents=True,exist_ok=True)
    archive=dest/'server_bundle.tar.gz'
    subprocess.run(['scp','-o','BatchMode=yes',HOST+':'+remote_archive,str(archive)],check=True)
    with tarfile.open(archive) as tar:
        for item in tar.getmembers():
            target=(dest/item.name).resolve()
            if not (target==dest.resolve() or dest.resolve() in target.parents) or item.issym() or item.islnk():
                raise ValueError('Unexpected archive path or link')
        tar.extractall(dest)
    return dest/'results'


def finalize(results):
    paper=ROOT/'paper/vtc2027_selective_sheng'
    env=dict(os.environ,MPLCONFIGDIR='/tmp/vtc2027_matplotlib')
    state('building_evidence')
    subprocess.run([sys.executable,'-m','scripts.build_vtc2027_evidence',
                    '--results',str(results),'--paper',str(paper)],cwd=ROOT,env=env,check=True)
    state('compiling_manuscript')
    build=paper/'build'
    process=subprocess.run([sys.executable,str(LATEX/'scripts/compile_latex.py'),str(paper/'main.tex'),
                            '--compiler','texlive','--output-directory',str(build),'--json'],
                           cwd=LATEX,text=True,capture_output=True)
    (ROOT/'research/vtc2027_sheng/compile_report.json').write_text(process.stdout)
    if process.returncode: raise RuntimeError('LaTeX compilation failed; see compile_report.json')
    check=command([PDFPY,str(ROOT/'scripts/audit_vtc2027_pdf.py'),str(build)])
    audit=json.loads(check)
    state('packaging',pdf_audit=audit)
    output=ROOT/'output/vtc2027_sheng';output.mkdir(parents=True,exist_ok=True)
    pdf=output/'VTC2027_Sheng_Manuscript.pdf'
    shutil.copy2(build/'main.pdf',pdf)
    archive=output/'VTC2027_Sheng_LaTeX.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name in ['main.tex','references.bib','README.md']:
            z.write(paper/name,arcname=name)
        for folder in ['generated','figures']:
            for path in (paper/folder).rglob('*'):
                if path.is_file(): z.write(path,arcname=str(path.relative_to(paper)))
    receipt=paper/'generated/evidence_receipt.json'
    shutil.copy2(receipt,output/'evidence_receipt.json')
    command(SSH+['mkdir -p '+shlex.quote(REMOTE+'/output')])
    subprocess.run(['scp','-o','BatchMode=yes',str(pdf),str(archive),str(output/'evidence_receipt.json'),
                    HOST+':'+REMOTE+'/output/'],check=True)
    state('manuscript_assembled',pdf=str(pdf),latex=str(archive),
          note='Complete experimental evidence inserted and PDF mechanically checked. Scientific/visual final review and author information remain required.')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--results',type=Path,help='Finalize an already retrieved complete bundle')
    a=p.parse_args()
    try:
        if a.results:
            finalize(a.results);return
        state('awaiting_server_experiments')
        deadline=time.time()+36*3600
        failures=0
        while time.time()<deadline:
            try:
                remote=json.loads(command(SSH+['cat '+shlex.quote(REMOTE+'/research/vtc2027_sheng/pipeline_status.json')],timeout=45))
                failures=0
            except (subprocess.SubprocessError,json.JSONDecodeError) as exc:
                failures+=1
                if failures>=30: raise RuntimeError('Server unreachable for 30 attempts; server jobs are not terminated') from exc
                time.sleep(60);continue
            if remote['stage']=='failed': raise RuntimeError('Server pipeline failed: '+remote.get('error','unknown'))
            if remote['stage']=='experiment_complete':
                state('retrieving_complete_results')
                results=retrieve();finalize(results);return
            # This is the same running task, not an installed recurring automation.
            time.sleep(60)
        raise TimeoutError('36-hour completion deadline elapsed; server jobs not terminated')
    except Exception as exc:
        state('failed',error=repr(exc));raise


if __name__=='__main__': main()
