"""Strict paired descriptive analysis; oracle routing is analysis only."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np

KEY=['index','seed','snr','true_global_sinr','profile']


def read_groups(path, modes=None):
    groups=defaultdict(dict)
    for line in path.open():
        r=json.loads(line)
        if modes is not None and r['mode'] not in modes: continue
        k=tuple(r[x] for x in KEY)
        if k in groups[r['mode']]: raise ValueError('Duplicate paired key')
        if not np.isfinite(r['psnr']): raise ValueError('Nonfinite PSNR')
        groups[r['mode']][k]=r
    if not groups: raise ValueError('No records')
    keys=set(next(iter(groups.values())))
    if any(set(g)!=keys for g in groups.values()): raise ValueError('Missing paired records')
    for k in keys:
        if len({(g[k]['image'],g[k]['interference_image']) for g in groups.values()})!=1:
            raise ValueError('Image pairing mismatch')
    return groups,sorted(keys)


def endpoints(delta):
    d=np.asarray(delta)
    return dict(mean_delta_psnr_db=float(d.mean()),harm_rate_gt_half_db=float(np.mean(d<-.5)),
                fifth_percentile_delta_db=float(np.quantile(d,.05)),win_fraction=float(np.mean(d>0)))


def oracle(groups, keys, modes):
    p=np.array([[groups[m][k]['psnr'] for k in keys] for m in modes])
    mean=p.mean(1); best=int(mean.argmax())
    return dict(modes=modes,best_fixed=modes[best],best_fixed_psnr_db=float(mean[best]),
                oracle_psnr_db=float(p.max(0).mean()),headroom_db=float(p.max(0).mean()-mean[best]),
                selection_fraction={m:float(np.mean(p.argmax(0)==i)) for i,m in enumerate(modes)},
                warning='Oracle uses ground-truth quality for analysis, never for receiver routing.')


def analyze(run):
    meta=json.loads((run/'metadata.json').read_text())
    if meta['status']!='complete': raise ValueError('Incomplete experiment')
    m=meta['manifest']; groups,keys=read_groups(run/'records.jsonl')
    expected={(i,s,snr,sinr,p) for i in range(m['max_images']) for s in m['seeds']
              for snr,sinr in m['snr_sinr_pairs'] for p in m['profiles']}
    if set(keys)!=expected or set(groups)!=set(m['modes']): raise ValueError('Protocol coverage mismatch')
    means={}
    for mode,rows in groups.items():
        rr=list(rows.values())
        means[mode]={field:float(np.mean([r[field] for r in rr])) for field in
                     ['psnr','mse','receiver_seconds','lpips_vgg','ms_ssim'] if field in rr[0]}
        nfe=[sum(np.asarray(r['receiver_estimates'][f]).sum() for f in ['nfe_signal','nfe_interference']) for r in rr]
        means[mode]['mean_prior_nfe']=float(np.mean(nfe))
        means[mode]['bypass_fraction']=float(np.mean([np.asarray(r['receiver_estimates'].get('bypass',[[0.]])).mean() for r in rr]))
        means[mode]['accepted_update_fraction']=float(sum(np.asarray(r['receiver_estimates'].get('accepted_count',[[0.]])).sum() for r in rr)/max(1,sum(np.asarray(r['receiver_estimates'].get('attempt_count',[[0.]])).sum() for r in rr)))
    comparisons=[]
    for target,ref in [('B1','A3'),('B2','B1'),('B3','B2'),('B_GUIDE','B2'),('B_COUPLED','B2'),('B3','A3'),('B3','NO_ICDM')]:
        for profile in [None]+m['profiles']:
            kk=[k for k in keys if profile is None or k[-1]==profile]
            comparisons.append(dict(target=target,reference=ref,profile=profile,
                                    **endpoints([groups[target][k]['psnr']-groups[ref][k]['psnr'] for k in kk])))
    cells=[]
    for profile in m['profiles']:
        for _,sinr in m['snr_sinr_pairs']:
            kk=[k for k in keys if k[-1]==profile and k[-2]==sinr]
            cells.append(dict(profile=profile,sinr=sinr,mean_psnr={mode:float(np.mean([rows[k]['psnr'] for k in kk])) for mode,rows in groups.items()}))
    return dict(phase='engineering_smoke' if m['max_images']<=4 else 'development_validation',
                records=sum(len(g) for g in groups.values()),summaries=means,comparisons=comparisons,
                by_condition=cells,oracle_routing=oracle(groups,keys,['NO_ICDM','A2','A3']),
                caveats=['Validation has previously been inspected; no independent confirmatory claim.',
                         'Metrics describe paired conditions, not independent repeated samples.',
                         'No downstream semantic metric yet; LPIPS is perceptual, MS-SSIM structural.'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path); p.add_argument('--historical',type=Path)
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.historical:
        groups,keys=read_groups(a.historical,['NO_ICDM','A2','A3'])
        result=dict(phase='historical_exploratory',paired_trials=len(keys),oracle_routing=oracle(groups,keys,['NO_ICDM','A2','A3']))
    else: result=analyze(a.run)
    output=a.output or a.run/'analysis.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2,allow_nan=False))

if __name__=='__main__': main()
