"""Matched, receiver-only experiments; all scientific inputs are archived."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import time

import numpy as np
import torch
from PIL import Image

from scripts.run_phase1 import files, load_image, make_config, codec, load_weights, digest, sync, block_powers
from scripts.baseline_metrics import ImageMetrics, METRIC_PROTOCOL
from scripts.cddm_receiver import cddm_receive
from scripts.run_rg_validation import receive as receive_rg
from scripts.run_published_baselines import normalize
from Diffusion import ICDMSampler
from Diffusion.calibration import to_complex, equalize, estimate_energy_power
from Autoencoder.channel import Channel
from DiT.models import DiT_models
from DiT import create_diffusion


MODES = ['DIRECT', 'CDDM_N0', 'CDDM_BLIND', 'A3', 'A4', 'B_COUPLED', 'CDDM_ORACLE']


def seeded(value):
    random.seed(value)
    np.random.seed(value % (2**32))
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def keyseed(*parts):
    return int(hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:8], 16)


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def atomic(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def features(y, snr, block):
    energy = y.abs().square().flatten(1)
    local = torch.stack([p.mean(1) for p in energy.split(block, 1)], 1)
    mean = energy.mean(1)
    return {'power': float((mean-1-10**(-snr/10)).clamp_min(0)),
            'heterogeneity': float(local.var(1, unbiased=False)/mean.square().clamp_min(1e-12)),
            'mean_energy': float(mean)}


def choose(policy, f):
    threshold = policy['power_threshold_low']
    if f['heterogeneity'] > policy['heterogeneity_threshold']:
        threshold = policy['power_threshold_high']
    take_gaussian = f['power'] <= threshold
    if policy.get('gaussian_side', 'below') == 'above':
        take_gaussian = not take_gaussian
    return 'CDDM_BLIND' if take_gaussian else 'A4'


@torch.no_grad()
def receive(mode, sampler, decoder, y, h, snr, block, abar, oracle=None, policy=None):
    counts = [0, 0]
    def hs(*_): counts[0] += 1
    def hz(*_): counts[1] += 1
    handles = [sampler.model_s.register_forward_hook(hs), sampler.model_z.register_forward_hook(hz)]
    sync(y.device)
    begin = time.perf_counter()
    selected = mode
    try:
        if mode == 'SELECT':
            selected = choose(policy, features(y, snr, block))
        info = {'selected': selected}
        if selected == 'B_COUPLED':
            rec, _, _ = receive_rg(selected, sampler, decoder, y, h, snr, block, y.device)
        else:
            if selected == 'DIRECT':
                latent = equalize(y, h, snr, 'awgn')
            elif selected.startswith('CDDM_'):
                if selected == 'CDDM_N0':
                    total = 10**(-snr/10)
                elif selected == 'CDDM_BLIND':
                    total = float(estimate_energy_power(y, h, snr)) + 10**(-snr/10)
                elif selected == 'CDDM_ORACLE':
                    if oracle is None: raise ValueError('Explicit oracle diagnostic only')
                    total = oracle
                else:
                    raise ValueError(selected)
                latent, extra = cddm_receive(sampler.model_s, equalize(y,h,snr,'awgn'), total, abar, 40)
                info.update(extra)
            elif selected == 'A3':
                latent, _, _ = sampler.SIC_sampling_estimated(snr, y, h, 'awgn', block)
            elif selected == 'A4':
                latent, _, _ = sampler.SIC_sampling_alternating(snr,y,h,'awgn',block,min_alpha=.5)
            else:
                raise ValueError(selected)
            rec = decoder(normalize(latent))
        sync(y.device)
        seconds = time.perf_counter()-begin
        if not torch.isfinite(rec).all(): raise FloatingPointError(selected)
        info.update(nfe_signal=counts[0], nfe_interference=counts[1], receiver_seconds=seconds)
        return rec, info
    finally:
        for handle in handles: handle.remove()


def gpu_snapshot():
    return subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.used,utilization.gpu', '--format=csv'], text=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    m=json.loads(args.manifest.read_text())
    policy=json.loads(Path(m['policy']).read_text()) if m.get('policy') else None
    if m['stage']!='development' and policy is None:
        raise ValueError('Freeze policy before opening confirmation')
    if m['channel_type']!='awgn': raise ValueError('AWGN-only audited receiver')
    if m['evaluation_batch_size']!=1: raise ValueError('Use batch one')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    device=torch.device('cuda:0')
    if not torch.cuda.is_available(): raise RuntimeError('GPU required')
    root=Path(__file__).resolve().parents[1]
    offset,count=m['offset'],m['max_images']
    paths=files(m['images'])[offset:offset+count]
    inter=files(m['interference_images'])[offset:offset+count]
    if len(paths)!=count or len(inter)!=count: raise ValueError('Incomplete data')
    hashes={k:digest(v['path']) for k,v in m['weights'].items()}
    if hashes!=m['expected_weights_sha256']: raise ValueError('Weights changed')
    source=[q for folder in ['Autoencoder','Diffusion','DiT'] for q in (root/folder).rglob('*.py') if not q.name.startswith('._')]
    source += [root/'scripts'/n for n in ['run_vtc2027.py','run_phase1.py','run_rg_validation.py','run_published_baselines.py','baseline_metrics.py','cddm_receiver.py']]
    images=[{'index':offset+i,'image':str(a),'image_sha256':digest(a),'interference':str(b),'interference_sha256':digest(b)} for i,(a,b) in enumerate(zip(paths,inter))]
    scientific={'manifest':m,'weights_sha256':hashes,'images':images,'policy':policy,
                'source_sha256':{str(q.relative_to(root)):digest(q) for q in source},
                'protocol_sha256':digest(root/'research/vtc2027_sheng/PROTOCOL.md'),
                'model_config_sha256':digest(m['model_config']), 'torch':torch.__version__}
    signature=hashlib.sha256(json.dumps(scientific,sort_keys=True).encode()).hexdigest()
    expected=count*len(m['seeds'])*len(m['conditions'])*len(m['modes'])
    args.output.mkdir(parents=True,exist_ok=args.resume)
    records=args.output/'records.jsonl'; metadata=args.output/'metadata.json'
    done=set()
    if args.resume and metadata.exists():
        old=json.loads(metadata.read_text())
        if old['signature']!=signature: raise ValueError('Inputs changed on resume')
        if records.exists():
            for line in records.read_text().splitlines():
                row=json.loads(line); key=(row['index'],row['seed'],row['condition'],row['mode'])
                if key in done: raise ValueError('Duplicate row')
                done.add(key)
    meta=dict(scientific,signature=signature,status='loading',expected_records=expected,
              host=platform.node(),gpu=torch.cuda.get_device_name(0),gpu_start=gpu_snapshot(),
              python=platform.python_version(),metrics=METRIC_PROTOCOL,started_at=time.time())
    atomic(metadata,meta)
    try:
        cfg=make_config(m); w=m['weights']
        enc=codec(cfg.MODEL.MODEL_NAME,'encoder',cfg,w['encoder']).to(device)
        dec=codec(cfg.MODEL.MODEL_NAME,'decoder',cfg,w['decoder']).to(device)
        infenc=codec(m['interference_model'],'encoder',cfg,w['infencoder']).to(device)
        kw=dict(in_channels=cfg.MODEL.OUT_CHANS,input_size=m['image_size']//2**len(cfg.MODEL.DEPTHS))
        ms=load_weights(DiT_models(**kw),w['icdm_s']).to(device)
        mz=load_weights(DiT_models(**kw),w['icdm_z']).to(device)
        sampler=ICDMSampler(ms,mz,cfg).to(device)
        metric=ImageMetrics(device); channel=Channel(cfg); abar=create_diffusion('').alphas_cumprod
        lp=Path(__import__('lpips').__file__).parent/'weights/v0.1/vgg.pth'
        vg=Path(torch.hub.get_dir())/'checkpoints/vgg16-397923af.pth'
        meta['metric_weights_sha256']={str(q):digest(q) for q in [lp,vg]}
        meta['status']='running'; atomic(metadata,meta)
        warmed=set(); counter=0
        with torch.no_grad(),records.open('a' if args.resume else 'x') as out:
            for local,(pa,pz) in enumerate(zip(paths,inter)):
                index=offset+local
                im=load_image(pa,m['image_size'],device)
                zi=load_image(pz,m['interference_image_size'],device)
                zi=torch.nn.functional.interpolate(zi,size=im.shape[-2:],mode='nearest')
                x,z=enc(im),infenc(zi)
                if local==0:
                    meta['complex_channel_uses']=x[0].numel()//2
                    meta['cbr']=meta['complex_channel_uses']/im[0].numel()
                    atomic(metadata,meta)
                for sd in m['seeds']:
                    for ci,condition in enumerate(m['conditions']):
                        snr,sinr,profile=condition['snr'],condition['sinr'],condition['profile']
                        keys=[(index,sd,ci,mode) for mode in m['modes']]
                        if all(k in done for k in keys): continue
                        block=m['block_size']; seeded(keyseed('channel',sd,index,ci))
                        power=block_powers(to_complex(x)[0].numel(),block,snr,sinr,
                                           profile if profile in ['stationary','alternating','burst'] else 'burst',device)
                        if profile in ['shifted','random_length']:
                            xc=to_complex(channel.complex_normalize(x,power=1)[0])
                            zc=to_complex(channel.complex_normalize(z,power=1)[0])
                            pattern=power.sqrt().repeat_interleave(block,1)[:,:xc[0].numel()]
                            if profile=='shifted': pattern=pattern.roll(block//2,1)
                            else:
                                rng=np.random.default_rng(keyseed('length',sd,index,ci)); seq=[]
                                while len(seq)<xc[0].numel(): seq.extend([float(rng.integers(0,2))]*int(rng.integers(16,129)))
                                pattern=torch.tensor(seq[:xc[0].numel()],device=device)[None,:]
                                pattern=pattern/ pattern.square().mean().clamp_min(1e-12).sqrt()
                                pattern=pattern*(10**(-sinr/10)-10**(-snr/10))**.5
                            y,h=channel.complex_inf_forward(xc,zc*pattern.reshape_as(zc),snr)
                            h=torch.ones_like(y.real)
                        else:
                            y,_,_,h=channel.inf_forward_blockwise(x,z,snr,power,block)
                        f=features(y,snr,block); yh=tensor_hash(y)
                        sampler_seed=keyseed('sampler',sd,index,ci)
                        order=m['modes'][counter%len(m['modes']):]+m['modes'][:counter%len(m['modes'])]
                        counter+=1
                        for mode in order:
                            key=(index,sd,ci,mode)
                            if key in done: continue
                            oracle=10**(-sinr/10) if mode=='CDDM_ORACLE' else None
                            just_warmed=mode not in warmed
                            if mode not in warmed:
                                seeded(sampler_seed)
                                warm,_=receive(mode,sampler,dec,y,h,snr,block,abar,oracle,policy)
                                warmed.add(mode)
                            seeded(sampler_seed)
                            rec,info=receive(mode,sampler,dec,y,h,snr,block,abar,oracle,policy)
                            if just_warmed and not torch.allclose(warm,rec,rtol=1e-5,atol=1e-6):
                                raise AssertionError('Seed-reset repeat differs: '+mode)
                            scores=metric(rec,im)
                            row=dict(index=index,seed=sd,condition=ci,mode=mode,stage=m['stage'],
                                     snr=snr,sinr=sinr,profile=profile,features=f,y_sha256=yh,
                                     sampler_seed=sampler_seed,**info,**{k:float(v) for k,v in scores.items()})
                            out.write(json.dumps(row,allow_nan=False)+'\n'); out.flush(); done.add(key)
                            if local in [0,15] and sd==m['seeds'][0] and ci in [0,8,15]:
                                folder=args.output/'qualitative'/f'{index}_{ci}'; folder.mkdir(parents=True,exist_ok=True)
                                for name,t in [('source',im),(mode,rec)]:
                                    Image.fromarray((255*t[0].clamp(0,1).permute(1,2,0).cpu().numpy()).round().astype('uint8')).save(folder/(name+'.png'))
                        print(json.dumps({'records':len(done),'expected':expected,'index':index,'condition':ci,'elapsed_seconds':time.time()-meta['started_at']}),flush=True)
        if len(done)!=expected: raise ValueError('Incomplete run')
        meta.update(status='complete',records=len(done),gpu_end=gpu_snapshot(),completed_at=time.time())
    except Exception as exc:
        meta.update(status='failed',error=repr(exc)); raise
    finally:
        atomic(metadata,meta)


if __name__=='__main__': main()
