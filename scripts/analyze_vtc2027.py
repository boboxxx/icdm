"""Strict paired summaries and image-group validation of a predeclared selector."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

METRICS=['psnr','mse','lpips_vgg','ms_ssim','receiver_seconds','nfe_signal','nfe_interference']
POWER=[0,.1,.2,.35,.5,.75,1,1.5,2,3,4,8,1e6]
HETERO=[.05,.1,.2,.4,.8,1.6]


def read_run(folder):
    meta=json.loads((folder/'metadata.json').read_text())
    if meta['status']!='complete': raise ValueError('Only completed runs may select a method')
    rows=[json.loads(s) for s in (folder/'records.jsonl').read_text().splitlines()]
    if len(rows)!=meta['expected_records']: raise ValueError('Incomplete count')
    paired=defaultdict(dict)
    for r in rows:
        k=(r['index'],r['seed'],r['condition'])
        if r['mode'] in paired[k]: raise ValueError('Duplicate key')
        paired[k][r['mode']]=r
    expected={(item['index'],seed,ci) for item in meta['images']
              for seed in meta['manifest']['seeds']
              for ci in range(len(meta['manifest']['conditions']))}
    if set(paired)!=expected: raise ValueError('Unexpected or missing image/seed/condition keys')
    for r in rows:
        if not all(np.isfinite(r[k]) for k in METRICS): raise ValueError('Nonfinite metric')
    for group in paired.values():
        if set(group)!=set(meta['manifest']['modes']): raise ValueError('Unmatched methods')
        if len({r['y_sha256'] for r in group.values()})!=1: raise ValueError('Different received signals')
        if len({r['sampler_seed'] for r in group.values()})!=1: raise ValueError('Different sampler seeds')
    return meta,rows,[paired[k] for k in sorted(paired)]


def rule(low, high, hetero, family,side='below'):
    return dict(power_threshold_low=low,power_threshold_high=high,
                heterogeneity_threshold=hetero,family=family,gaussian_side=side)


def mask(policy,power,hetero):
    threshold=np.where(hetero>policy['heterogeneity_threshold'],policy['power_threshold_high'],policy['power_threshold_low'])
    decision=power<=threshold
    return ~decision if policy.get('gaussian_side','below')=='above' else decision


def fit(policies,power,hetero,g,s,gt,st,indices):
    best=None
    for p in policies:
        choose_g=mask(p,power,hetero)
        quality=float(np.where(choose_g,g,s)[indices].mean())
        latency=float(np.where(choose_g,gt,st)[indices].mean())
        ranking=(quality,-latency)
        if best is None or ranking>best[0]: best=(ranking,p)
    return best[1]


def fit_policy(groups,folder,meta):
    if meta['manifest']['stage']!='development' or len(meta['images'])!=64:
        raise ValueError('Policy fitting is restricted to declared development images')
    ids=np.array([g['A4']['index'] for g in groups])
    power=np.array([g['A4']['features']['power'] for g in groups])
    hetero=np.array([g['A4']['features']['heterogeneity'] for g in groups])
    g,s=(np.array([x[m]['psnr'] for x in groups]) for m in ['CDDM_BLIND','A4'])
    gt,st=(np.array([x[m]['receiver_seconds'] for x in groups]) for m in ['CDDM_BLIND','A4'])
    families={'energy':[rule(t,t,1e6,'energy',side) for side in ['below','above'] for t in POWER],
              'energy_heterogeneity':[rule(a,b,h,'energy_heterogeneity',side) for side in ['below','above'] for h in HETERO for a in POWER for b in POWER]}
    report={}; fitted={}; predictions={}
    for name,candidates in families.items():
        cv=np.zeros(len(g),dtype=bool); folds=[]
        for fold in range(8):
            train=ids%8!=fold; test=~train
            p=fit(candidates,power,hetero,g,s,gt,st,train)
            cv[test]=mask(p,power[test],hetero[test]); folds.append(p)
        fitted[name]=fit(candidates,power,hetero,g,s,gt,st,np.ones(len(g),dtype=bool))
        predictions[name]=cv
        report[name]={'cv_psnr':float(np.where(cv,g,s).mean()),'cv_component_seconds':float(np.where(cv,gt,st).mean()),
                      'fold_rules':folds,'full_development_rule':fitted[name],'candidates':len(candidates)}
    name=max(report,key=lambda n:(report[n]['cv_psnr'],-report[n]['cv_component_seconds'],n=='energy'))
    best=fitted[name]
    fixed=max([('CDDM_BLIND',g.mean()),('A4',s.mean())],key=lambda v:v[1])
    status='selected_policy'
    if report[name]['cv_psnr']<=fixed[1]:
        # Degenerate rules return the fixed receiver when selection does not help.
        t=1e6 if fixed[0]=='CDDM_BLIND' else -1
        best=rule(t,t,1e6,'fixed_'+fixed[0]); status='no_cv_gain_over_fixed'
    best.update(scientific_status=status,development_signature=meta['signature'],
                objective='mean image PSNR, 8 image-group folds; no confirmation tuning')
    report.update(best_fixed={'mode':fixed[0],'psnr':float(fixed[1])},selected=best,
                  offline_oracle_psnr=float(np.maximum(g,s).mean()),
                  warning='CV is development evidence; component times are estimates, not online selector timings')
    (folder/'selection_cv.json').write_text(json.dumps(report,indent=2))
    with (folder/'policy.json').open('x') as out: json.dump(best,out,indent=2)
    with (folder/'energy_policy.json').open('x') as out: json.dump(fitted['energy'],out,indent=2)
    cvrows=[]
    for name,pred in predictions.items():
        for i,group in enumerate(groups):
            r=dict(group['CDDM_BLIND' if pred[i] else 'A4']);r['mode']='CV_'+name;cvrows.append(r)
    return report,cvrows


def summaries(rows):
    by=defaultdict(list)
    for r in rows: by[r['mode']].append(r)
    return {mode:{k:float(np.mean([r[k] for r in rs])) for k in METRICS} for mode,rs in by.items()}


def contrast(groups,target,reference):
    ds=np.array([g[target]['psnr']-g[reference]['psnr'] for g in groups])
    ids=np.array([g[target]['index'] for g in groups]); uid=np.unique(ids)
    cluster=np.array([ds[ids==i].mean() for i in uid])
    rng=np.random.default_rng(20260921)
    bs=cluster[rng.integers(0,len(cluster),size=(5000,len(cluster)))].mean(1)
    secondary={}
    for metric in ['mse','lpips_vgg','ms_ssim']:
        delta=np.array([g[target][metric]-g[reference][metric] for g in groups])
        secondary[metric]=float(delta.mean())
    return {'target':target,'reference':reference,'mean_delta_psnr':float(ds.mean()),
            'ci95_image_cluster':np.quantile(bs,[.025,.975]).tolist(),
            'harm_gt_half_db':float((ds<-.5).mean()),'fifth_percentile':float(np.quantile(ds,.05)),
            'win_fraction':float((ds>0).mean()),'image_clusters':len(uid),
            'secondary_mean_deltas':secondary}


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--fit',action='store_true')
    a=p.parse_args();meta,rows,groups=read_run(a.run)
    derived=[]
    for group in groups:
        if {'DIRECT','A4'}.issubset(group):
            r=dict(group['DIRECT' if group['A4']['features']['power']<=.15 else 'A4'])
            r['mode']='A4_GATE';r['timing_kind']='component estimate, gate overhead excluded'
            group['A4_GATE']=r;derived.append(r)
        if 'SELECT' in group:
            online=group['SELECT'];branch=group[online['selected']]
            for metric in ['psnr','mse','lpips_vgg','ms_ssim']:
                if not np.isclose(online[metric],branch[metric],rtol=1e-5,atol=1e-6):
                    raise ValueError('Online selector and chosen branch do not match')
    selected_report=None
    if a.fit:
        selected_report,cvrows=fit_policy(groups,a.run,meta);derived.extend(cvrows)
    allrows=rows+derived
    bycondition={}
    for ci,condition in enumerate(meta['manifest']['conditions']):
        bycondition[str(ci)]=dict(condition,means=summaries([r for r in allrows if r['condition']==ci]))
    comparisons=[]
    target='SELECT' if 'SELECT' in groups[0] else 'B_COUPLED'
    if target in groups[0]:
        for ref in ['DIRECT','CDDM_N0','CDDM_BLIND','A3','A4','A4_GATE']:
            if ref in groups[0]:comparisons.append(contrast(groups,target,ref))
    report={'status':'complete','stage':meta['manifest']['stage'],'records':len(rows),
            'images':len(meta['images']),'signature':meta['signature'],'means':summaries(allrows),
            'by_condition':bycondition,'comparisons':comparisons,'selection':selected_report,
            'timing_caveat':'A4_GATE and CV selectors use replayed component costs; SELECT is executed online'}
    if {'SELECT','CDDM_BLIND','A4'}.issubset(groups[0]):
        selected=np.array([g['SELECT']['psnr'] for g in groups])
        oracle=np.array([max(g['CDDM_BLIND']['psnr'],g['A4']['psnr']) for g in groups])
        report['selector']={'gaussian_fraction':float(np.mean([g['SELECT']['selected']=='CDDM_BLIND' for g in groups])),
                            'offline_oracle_psnr':float(oracle.mean()),'mean_oracle_regret_db':float((oracle-selected).mean())}
    (a.run/'analysis.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    with (a.run/'means.csv').open('w') as out:
        writer=csv.DictWriter(out,fieldnames=['mode']+METRICS);writer.writeheader()
        for mode,s in report['means'].items(): writer.writerow(dict(mode=mode,**s))
    with (a.run/'derived_records.jsonl').open('w') as out:
        for row in derived: out.write(json.dumps(row,allow_nan=False)+'\n')
    print(json.dumps({'status':'complete','means':report['means'],'selection':None if selected_report is None else selected_report['selected']},indent=2))

if __name__=='__main__': main()
