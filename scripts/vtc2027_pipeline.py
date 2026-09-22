"""Complete the prespecified study sequentially; a failed gate stops all later stages."""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
STATUS=ROOT/'research/vtc2027_sheng/pipeline_status.json'


def status(stage,**extra):
    value=dict(stage=stage,updated_at=time.time(),**extra)
    temp=STATUS.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(STATUS)
    print(json.dumps(value),flush=True)


def run(*args):
    subprocess.run([sys.executable,'-m',*args],cwd=ROOT,check=True)


def main():
    try:
        status('data_audit')
        run('scripts.audit_vtc2027_data')
        status('awaiting_development')
        deadline=time.time()+6*3600
        meta_path=ROOT/'results/development/metadata.json'
        while True:
            if meta_path.exists():
                m=json.loads(meta_path.read_text())
                if m['status']=='complete': break
                if m['status']=='failed': raise RuntimeError(m.get('error','development failed'))
            if time.time()>deadline: raise TimeoutError('Development has not finished within six hours')
            time.sleep(30)
        status('freezing_policy')
        if not (ROOT/'results/development/policy.json').exists():
            run('scripts.analyze_vtc2027','--run','results/development','--fit')
        for stage in ['confirmation','pressure']:
            status(stage)
            manifest=ROOT/f'configs/vtc_{stage}.json'
            if not manifest.exists():
                run('scripts.prepare_vtc2027','--stage',stage,'--policy','results/development/policy.json','--output',str(manifest))
            output=ROOT/'results'/stage
            prior=json.loads((output/'metadata.json').read_text()) if (output/'metadata.json').exists() else None
            if prior is None or prior['status']!='complete':
                command=['scripts.run_vtc2027','--manifest',str(manifest),'--output',str(output)]
                if output.exists(): command.append('--resume')
                run(*command)
            run('scripts.analyze_vtc2027','--run',str(output))
        status('experiment_complete',note='All records complete; manuscript synthesis and PDF review follow separately')
    except Exception as exc:
        status('failed',error=repr(exc));raise


if __name__=='__main__': main()
