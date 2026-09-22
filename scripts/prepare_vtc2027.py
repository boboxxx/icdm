"""Resolve predeclared sheng manifests, never inspect reconstruction outcomes."""
import argparse
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=['smoke','development','confirmation','pressure'],required=True)
    p.add_argument('--policy')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    m=json.loads(Path('configs/rg_validation.json').read_text())
    for k in m['weights']:
        m['weights'][k]['path']='/mnt/d/ICDM_SC_runs/phase1_full/weights/'+k+'.pth'
    m['images']='/mnt/d/ICDM_SC_data_full/celeba/val/class0'
    m['interference_images']='/mnt/d/ICDM_SC_data_full/cifar10/val/class0'
    m.update(stage='development',offset=0,seeds=[1700],modes=['DIRECT','CDDM_N0','CDDM_BLIND','A3','A4','B_COUPLED','CDDM_ORACLE'])
    m['conditions']=[dict(snr=20,sinr=v,profile=q) for v in [-7,-4,0,4,7,10] for q in ['stationary','alternating','burst']]
    for k in ['snr_sinr_pairs','profiles','protocol','metrics']: m.pop(k,None)
    if a.stage=='smoke':
        m.update(max_images=2,seeds=[91700],conditions=[dict(snr=20,sinr=v,profile='burst') for v in [-7,4,10]])
    if a.stage in ['confirmation','pressure']:
        if not a.policy: raise ValueError('Frozen policy required')
        m.update(stage=a.stage,split='test',offset=512,max_images=256,seeds=[2700,2701],policy=a.policy,
                 modes=['DIRECT','CDDM_N0','CDDM_BLIND','A3','A4','SELECT'])
        m['images']=m['images'].replace('/val/','/test/')
        m['interference_images']=m['interference_images'].replace('/val/','/test/')
    if a.stage=='pressure':
        m.update(offset=1024,max_images=64,seeds=[3700],modes=['DIRECT','CDDM_BLIND','A4','SELECT'])
        m['conditions']=[dict(snr=20,sinr=v,profile=q) for v in [-4,4,10] for q in ['shifted','random_length']]+[dict(snr=20,sinr=20,profile='stationary')]
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f: json.dump(m,f,indent=2)

if __name__=='__main__': main()
