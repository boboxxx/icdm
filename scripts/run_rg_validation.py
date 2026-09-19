"""Prospectively specified RG-ICDM validation; no test tuning or training."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import tarfile
import time

import torch
from Diffusion import ICDMSampler
from Diffusion.calibration import equalize, to_complex
from Autoencoder.channel import Channel
from scripts.run_phase1 import (files, load_image, make_config, codec, load_weights,
                                digest, block_powers, seed, sync, recover)


MODES=['NO_ICDM','A2','A3','B1','B2','B3','B_GUIDE','B_COUPLED']


@torch.no_grad()
def receive(mode,sampler,decoder,y,h,snr,block_size,device):
    from Diffusion.selective import selective_receive
    calls=[0,0]
    def hook_s(*_): calls[0]+=1
    def hook_z(*_): calls[1]+=1
    handles=[sampler.model_s.register_forward_hook(hook_s),sampler.model_z.register_forward_hook(hook_z)]
    try:
        if mode in {'NO_ICDM','A2','A3'}:
            rec,est,timing=recover(mode,sampler,decoder,y,h,snr,None,'awgn',block_size,device)
        else:
            sync(device); t0=time.perf_counter()
            latent,_,est=selective_receive(sampler,snr,y,h,block_size,mode=mode)
            sync(device); t1=time.perf_counter()
            latent=latent/(2*latent.square().flatten(1).mean(1).clamp_min(1e-12)).sqrt().view(-1,1,1,1)
            rec=decoder(latent)
            sync(device); t2=time.perf_counter()
            if not torch.isfinite(rec).all(): raise FloatingPointError('Nonfinite output')
            timing={'sampling_and_calibration_seconds':t1-t0,
                    'normalization_and_decode_seconds':t2-t1,'receiver_seconds':t2-t0}
        est={} if est is None else est
        est.update(nfe_signal=y.real.new_full((y.shape[0],1),calls[0]),
                   nfe_interference=y.real.new_full((y.shape[0],1),calls[1]))
        return rec,est,timing
    finally:
        for handle in handles: handle.remove()


def atomic_json(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False))
    temporary.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    m=json.loads(args.manifest.read_text())
    if m['channel_type']!='awgn' or m['split']!='val':
        raise ValueError('This locked runner only runs validation AWGN')
    if m['modes']!=MODES: raise ValueError('All locked controls must be included')
    if m['evaluation_batch_size']!=1: raise ValueError('Matched bypass protocol requires batch one')
    if not torch.cuda.is_available(): raise RuntimeError('Use a Slurm GPU allocation')
    device=torch.device('cuda:0')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    count=m['max_images']
    paths=files(m['images'])[:count]
    inter=files(m['interference_images'])[:count]
    if len(paths)!=count or len(inter)!=count: raise ValueError('Missing locked data')
    root=Path(__file__).resolve().parents[1]
    weights_hash={k:digest(v['path']) for k,v in m['weights'].items()}
    if weights_hash!=m['expected_weights_sha256']: raise ValueError('Checkpoint hash mismatch')
    provenance={
        'manifest':m,'weights_sha256':weights_hash,
        'protocol_sha256':digest(root/m['protocol']),
        'model_config_sha256':digest(root/m['model_config']),
        'images':[(str(x),digest(x)) for x in paths],
        'interference_images':[(str(x),digest(x)) for x in inter],
        'metrics':m.get('metrics',['psnr']),
        'source_sha256':{str(x.relative_to(root)):digest(x)
            for folder in ['Diffusion','Autoencoder','DiT','scripts']
            for x in sorted((root/folder).rglob('*.py')) if not x.name.startswith('._')},
        'torch':torch.__version__,'gpu':torch.cuda.get_device_name(device),
        'hostname':platform.node(),'slurm_job_id':os.environ.get('SLURM_JOB_ID'),
    }
    # SLURM_JOB_ID is allowed to differ on a resumed allocation; scientific inputs aren't.
    scientific={k:v for k,v in provenance.items() if k not in {'hostname','slurm_job_id'}}
    signature=hashlib.sha256(json.dumps(scientific,sort_keys=True).encode()).hexdigest()
    expected=count*len(m['seeds'])*len(m['snr_sinr_pairs'])*len(m['profiles'])*len(MODES)
    args.output.mkdir(parents=True,exist_ok=args.resume)
    meta_path=args.output/'metadata.json'
    records_path=args.output/'records.jsonl'
    done=set()
    if records_path.exists() and not meta_path.exists():
        raise ValueError('Records without provenance metadata: unsafe resume')
    if args.resume and meta_path.exists():
        previous=json.loads(meta_path.read_text())
        if previous['input_signature']!=signature: raise ValueError('Unsafe resume: inputs changed')
        if records_path.exists():
            for line in records_path.read_text().splitlines():
                r=json.loads(line)
                k=tuple(r[x] for x in ['index','seed','snr','true_global_sinr','profile','mode'])
                if k in done: raise ValueError('Duplicate records')
                done.add(k)
    metadata=dict(provenance,input_signature=signature,status='loading',expected_records=expected)
    source_archive=args.output/'source_snapshot.tar.gz'
    if not source_archive.exists():
        with tarfile.open(source_archive,'x:gz') as archive:
            for name in sorted(set(provenance['source_sha256']) | {m['protocol'],m['model_config']}):
                archive.add(root/name,arcname=name)
            archive.add(args.manifest,arcname='resolved_run_manifest.json')
    metadata['source_archive_sha256']=digest(source_archive)
    atomic_json(meta_path,metadata)
    try:
        cfg=make_config(m)
        w=m['weights']
        enc=codec(cfg.MODEL.MODEL_NAME,'encoder',cfg,w['encoder']).to(device)
        dec=codec(cfg.MODEL.MODEL_NAME,'decoder',cfg,w['decoder']).to(device)
        infenc=codec(m['interference_model'],'encoder',cfg,w['infencoder']).to(device)
        from DiT.models import DiT_models
        kw=dict(in_channels=cfg.MODEL.OUT_CHANS,input_size=m['image_size']//2**len(cfg.MODEL.DEPTHS))
        ms=load_weights(DiT_models(**kw),w['icdm_s']).to(device)
        mz=load_weights(DiT_models(**kw),w['icdm_z']).to(device)
        sampler=ICDMSampler(ms,mz,cfg).to(device)
        metric=None
        if 'lpips_vgg' in m.get('metrics',[]):
            from scripts.baseline_metrics import ImageMetrics
            metric=ImageMetrics(device)
            lpips_path=Path(__import__('lpips').__file__).parent/'weights/v0.1/vgg.pth'
            vgg_path=Path(torch.hub.get_dir())/'checkpoints/vgg16-397923af.pth'
            metadata['metric_weights_sha256']={str(p):digest(p) for p in [lpips_path,vgg_path]}
        channel=Channel(cfg)
        metadata['status']='running'
        atomic_json(meta_path,metadata)
        warmed=set()
        setting=0
        bs=m['evaluation_batch_size']
        with torch.no_grad(),records_path.open('a' if args.resume else 'x') as stream:
            for start in range(0,count,bs):
                indices=list(range(start,min(count,start+bs)))
                im=torch.cat([load_image(paths[i],m['image_size'],device) for i in indices])
                zi=torch.cat([load_image(inter[i],m['interference_image_size'],device) for i in indices])
                zi=torch.nn.functional.interpolate(zi,size=im.shape[-2:],mode='nearest')
                x,z=enc(im),infenc(zi)
                for sd in m['seeds']:
                    for snr,sinr in m['snr_sinr_pairs']:
                        for profile in m['profiles']:
                            block=m['block_size']
                            power=block_powers(to_complex(x)[0].numel(),block,snr,sinr,profile,device)
                            ys,hs,zs=[],[],[]
                            for local,index in enumerate(indices):
                                seed(sd+10000*index)
                                if profile=='stationary':
                                    yi,_,_,hi=channel.inf_forward(x[local:local+1],z[local:local+1],snr,sinr)
                                else:
                                    yi,_,_,hi=channel.inf_forward_blockwise(x[local:local+1],z[local:local+1],snr,power,block)
                                ys.append(yi); hs.append(hi)
                                zz,_=channel.complex_normalize(z[local:local+1],power=1)
                                zs.append(to_complex(zz))
                            y,h,ztrue=torch.cat(ys),torch.cat(hs),torch.cat(zs)
                            block_energy=torch.stack([v.abs().square().mean(1) for v in ztrue.flatten(1).split(block,1)],1)*power
                            # Deterministic cyclic ordering distributes thermal/order effects.
                            offset=setting%len(MODES)
                            order=MODES[offset:]+MODES[:offset]
                            setting+=1
                            for mode in order:
                                keys=[(i,sd,snr,sinr,profile,mode) for i in indices]
                                if all(k in done for k in keys): continue
                                if mode not in warmed:
                                    seed(sd+10000*start+1000000)
                                    receive(mode,sampler,dec,y,h,snr,block,device)
                                    warmed.add(mode)
                                seed(sd+10000*start+1000000)
                                rec,est,timing=receive(mode,sampler,dec,y,h,snr,block,device)
                                mses=(rec.clamp(0,1)-im).square().flatten(1).mean(1)
                                scores={} if metric is None else metric(rec,im)
                                for local,index in enumerate(indices):
                                    if keys[local] in done: continue
                                    mse=mses[local].item()
                                    row=dict(index=index,image=str(paths[index]),interference_image=str(inter[index]),
                                        split='val',seed=sd,snr=snr,true_global_sinr=sinr,profile=profile,mode=mode,
                                        mse=mse,psnr=-10*math.log10(max(mse,1e-15)),
                                        block_size=block,true_block_power=power.cpu().tolist(),
                                        realized_block_interference_energy=block_energy[local:local+1].cpu().tolist(),
                                        evaluation_batch_count=len(indices),receiver_batch_seconds=timing['receiver_seconds'],
                                        **{k:v/len(indices) for k,v in timing.items()})
                                    row.update({k:float(v[local]) for k,v in scores.items()})
                                    if est is not None:
                                        row['receiver_estimates']={k:v[local:local+1].cpu().tolist() for k,v in est.items()}
                                    stream.write(json.dumps(row,allow_nan=False)+'\n')
                                    stream.flush()
                                    done.add(keys[local])
                                print(json.dumps(dict(records=len(done),expected=expected,mode=mode,profile=profile,sinr=sinr)),flush=True)
        if len(done)!=expected: raise ValueError('Incomplete result count')
        metadata['status']='complete'
        metadata['records']=len(done)
    except Exception as error:
        metadata.update(status='failed',error=repr(error))
        raise
    finally:
        atomic_json(meta_path,metadata)


if __name__=='__main__': main()
