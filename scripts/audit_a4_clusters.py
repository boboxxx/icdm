"""Reproducible historical pairing audit and image-cluster bootstrap."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def load(path):
    with path.open() as f: return [json.loads(line) for line in f]

def key(r):
    return tuple(r[k] for k in ['index','seed','snr','true_global_sinr','profile'])

def main():
    base=load(ROOT/'research/sheng_phase1_full/records.jsonl')
    a4=load(ROOT/'research/sheng_a4_full/records.jsonl')
    groups=defaultdict(dict)
    for r in base+a4:
        assert key(r) not in groups[r['mode']], 'duplicate'
        groups[r['mode']][key(r)]=r
    assert all(set(rows)==set(groups['A4']) for rows in groups.values())
    metadata=[json.loads((ROOT/'research'/x/'metadata.json').read_text())
              for x in ['sheng_phase1_full','sheng_a4_full']]
    assert metadata[0]['weights_sha256']==metadata[1]['weights_sha256']
    for k,r in groups['A4'].items():
        old=groups['A3'][k]
        assert r['image']==old['image'] and r['interference_image']==old['interference_image']
        np.testing.assert_array_equal(r['receiver_estimates']['initial_power'],old['receiver_estimates']['power'])
    report={'pairs':len(a4),'image_clusters':len({r['index'] for r in a4}),
            'pairing_and_initial_power_exact':True,'ci_method':'10000 image-pair-cluster percentile bootstrap; seed 20260919',
            'status':'historical exploratory, not independent confirmation','contrasts':[]}
    for ref in ['A3','A0','NO_ICDM']:
        for profile in ['all','stationary','alternating','burst']:
            for sinr in [None,-7,-4,0,4,7,10]:
                per_image=defaultdict(list)
                for k,r in groups['A4'].items():
                    if profile!='all' and r['profile']!=profile: continue
                    if sinr is not None and r['true_global_sinr']!=sinr: continue
                    per_image[r['index']].append(r['psnr']-groups[ref][k]['psnr'])
                values=np.array([np.mean(per_image[i]) for i in sorted(per_image)])
                rng=np.random.default_rng(20260919)
                samples=values[rng.integers(0,len(values),size=(10000,len(values)))].mean(1)
                lo,hi=np.quantile(samples,[.025,.975])
                report['contrasts'].append({'reference':ref,'profile':profile,'sinr':sinr,
                    'mean':float(values.mean()),'ci95':[float(lo),float(hi)]})
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
