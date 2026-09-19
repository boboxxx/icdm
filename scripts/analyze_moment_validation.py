"""Paired image-cluster validation analysis; never launches test experiments."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np


def contrast(groups, target, reference, profile=None, sinrs=None):
    per_image = defaultdict(list)
    for key, row in groups[target].items():
        if profile is not None and row['profile'] != profile:
            continue
        if sinrs is not None and row['true_global_sinr'] not in sinrs:
            continue
        per_image[row['index']].append(row['psnr']-groups[reference][key]['psnr'])
    values = np.array([np.mean(per_image[i]) for i in sorted(per_image)])
    if not len(values):
        raise ValueError('Empty contrast')
    rng = np.random.default_rng(20260919)
    samples = values[rng.integers(0, len(values), (10000, len(values)))].mean(1)
    return dict(target=target, reference=reference, profile=profile, sinrs=sinrs,
                image_clusters=len(values), mean_db=float(values.mean()),
                ci95_db=np.quantile(samples, [.025,.975]).tolist())


def analyze(run):
    meta = json.loads((run/'metadata.json').read_text())
    if meta['status'] != 'complete':
        raise ValueError('Cannot report an incomplete experiment')
    m = meta['manifest']
    expected = {(i,s,snr,sinr,p) for i in range(m['max_images']) for s in m['seeds']
                for snr,sinr in m['snr_sinr_pairs'] for p in m['profiles']}
    groups = defaultdict(dict)
    with (run/'records.jsonl').open() as f:
        for line in f:
            r = json.loads(line)
            key = tuple(r[k] for k in ['index','seed','snr','true_global_sinr','profile'])
            if key in groups[r['mode']]:
                raise ValueError('Duplicate record')
            if not np.isfinite([r['mse'], r['psnr'], r['receiver_seconds']]).all():
                raise ValueError('Nonfinite endpoint')
            groups[r['mode']][key] = r
    if set(groups) != set(m['modes']) or any(set(g) != expected for g in groups.values()):
        raise ValueError('Incomplete or mismatched paired keys')
    for key in expected:
        identities = {(g[key]['image'],g[key]['interference_image']) for g in groups.values()}
        if len(identities) != 1:
            raise ValueError('Image pairing mismatch')
    primary = contrast(groups, 'A5_FULL','A4_R')
    severe = contrast(groups, 'A5_FULL','A4_R','burst',[-7,-4])
    exploratory = []
    for target,reference in [('A5_FULL','A4_R'),('A5_DIAG','A4_R'),('A5_FULL','A5_DIAG'),
                             ('A4_R','A4'),('A4','A3'),('A5_FULL','NO_ICDM')]:
        for profile in [None,'stationary','alternating','burst']:
            exploratory.append(contrast(groups,target,reference,profile))
    summaries = {}
    for mode,group in groups.items():
        rows = list(group.values())
        summary = dict(mean_psnr=float(np.mean([r['psnr'] for r in rows])),
                       mean_receiver_seconds_per_image=float(np.mean([r['receiver_seconds'] for r in rows])))
        power_errors = [np.asarray(r['receiver_estimates']['power'])-np.asarray(r['true_block_power'])
                        for r in rows if r.get('receiver_estimates') is not None]
        if power_errors:
            summary['nominal_block_power_rmse'] = float(np.sqrt(np.mean(np.concatenate([e.ravel()**2 for e in power_errors]))))
        for field in ['max_power_seen','max_objective_increase','candidate_at_cap','candidate_at_floor',
                      'denominator','mean_residual_energy']:
            arrays = [np.asarray(r['receiver_estimates'][field]).ravel() for r in rows
                      if r.get('receiver_estimates') is not None and field in r['receiver_estimates']]
            if arrays:
                a = np.concatenate(arrays)
                if not np.isfinite(a).all(): raise ValueError('Nonfinite diagnostic')
                summary[field] = dict(min=float(a.min()),mean=float(a.mean()),max=float(a.max()))
        summaries[mode] = summary
    smoke = m['max_images'] == 4
    return dict(status='complete', phase='smoke_engineering_only' if smoke else 'validation',
                input_signature=meta['input_signature'], records=sum(map(len,groups.values())),
                primary=primary, low_sinr_burst=severe, exploratory_contrasts=exploratory,
                summaries=summaries, validation_gate_passed=None if smoke else
                bool(primary['ci95_db'][0] > 0 and severe['mean_db'] >= 0),
                caveats=['Gaussian-surrogate moments are not exact neural posterior moments.',
                         'Image clusters, not records, are bootstrap units.',
                         'Stratum intervals exploratory; no multiplicity adjustment.',
                         'Validation only; no independent held-out test evidence.'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    a = p.parse_args()
    report = analyze(a.run)
    out = a.run/'analysis.json'
    out.write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__ == '__main__':
    main()
