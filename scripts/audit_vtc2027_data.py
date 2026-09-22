"""Check reserved image identities before fitting or opening confirmation outputs."""
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root=Path('/mnt/d/ICDM_SC_data_full')
    ranges={'previous_test':('test',0,200),'development':('val',0,64),
            'confirmation':('test',512,768),'pressure':('test',1024,1088)}
    report={'scope':'Exact file hashes and declared image splits; not person-identity independence',
            'datasets':{},'status':'complete'}
    for dataset in ['celeba','cifar10']:
        groups={}
        for name,(split,start,end) in ranges.items():
            paths=sorted((root/dataset/split/'class0').glob('*.png'))[start:end]
            if len(paths)!=end-start: raise ValueError('Missing reserved images')
            rows=[{'path':str(p),'sha256':digest(p)} for p in paths]
            if len({r['sha256'] for r in rows})!=len(rows):
                raise ValueError('Duplicate images within '+name)
            groups[name]=rows
        names=list(groups)
        for i,a in enumerate(names):
            for b in names[i+1:]:
                if {x['sha256'] for x in groups[a]} & {x['sha256'] for x in groups[b]}:
                    raise ValueError('Image leakage '+dataset+' '+a+' '+b)
        report['datasets'][dataset]=groups
    previous=Path('/mnt/d/ICDM_SC_runs/phase1_full_eval/metadata.json')
    if not previous.exists(): raise FileNotFoundError('Original evaluation metadata required')
    old=json.loads(previous.read_text())
    report['previous_metadata_sha256']=digest(previous)
    report['previous_metadata_keys']=list(old)
    # The old manifest declares the first 200 test pairs; archive it verbatim.
    manifest=old.get('manifest',old.get('config'))
    if manifest is None or manifest.get('max_images')!=200:
        raise ValueError('Audit previous evaluation reservation manually')
    report['previous_manifest']=manifest
    if '/test/' not in manifest['images'] or '/test/' not in manifest['interference_images']:
        raise ValueError('Unexpected previous test split')
    out=Path('research/vtc2027_sheng/data_audit.json')
    out.write_text(json.dumps(report,indent=2))
    print(json.dumps({'status':report['status'],'output':str(out),'hash_disjoint':True}))


if __name__=='__main__': main()
