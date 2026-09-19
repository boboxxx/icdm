"""Receiver-only hierarchical power estimates and conservative feedback (AWGN)."""
import math
import torch
from Diffusion.calibration import (estimate_energy_power, expand_blocks, power_to_sinr,
                                  guidance_parameters, equalize, to_complex)


def hierarchical_power(y, h, snr, block_size):
    # Reuse strict input validation, but estimate heterogeneity before clipping.
    estimate_energy_power(y, h, snr, block_size)
    excess = (y.abs().square()-h.abs().square()-10**(-snr/10)).flatten(1)
    parts = excess.split(block_size, 1)
    sizes = y.real.new_tensor([p.shape[1] for p in parts])
    q = torch.stack([p.mean(1) for p in parts], 1)
    # A singleton cannot estimate variance; conservatively use frame variance.
    fallback = excess.var(1, unbiased=False).clamp_min(1e-8)
    s2 = torch.stack([p.var(1, unbiased=True)/p.shape[1] if p.shape[1]>1
                      else fallback for p in parts], 1).clamp_min(1e-8)
    weights = sizes/sizes.sum()
    global_q = (q*weights).sum(1, keepdim=True)
    if len(parts) == 1:
        tau = torch.zeros_like(global_q)
    else:
        variance = ((q-global_q).square()*weights).sum(1, keepdim=True)
        sampling = (s2*weights*(1-weights)).sum(1, keepdim=True)
        tau = ((variance-sampling)/(1-weights.square().sum())).clamp_min(0)
    pooling = tau/(tau+s2)
    pg = global_q.clamp_min(0)
    power = (1-pooling)*pg + pooling*q.clamp_min(0)
    return dict(power=power, global_power=pg, raw_power=q, variance=s2,
                heterogeneity=tau, local_weight=pooling)


def trusted_amplitude(y, x, z, old, anchor, block_size, snr, alpha, sigma,
                      damping=.25, radius=.25, margin=.05):
    """Even-coordinate proposal, odd-coordinate residual check; not a formal test."""
    yc, xc, zc = map(to_complex, (y, x, z))
    if yc.shape != xc.shape or yc.shape != zc.shape:
        raise ValueError('Mismatched point estimates')
    if not all(torch.isfinite(v).all() for v in (yc, xc, zc, old, anchor)):
        raise ValueError('Nonfinite receiver input')
    if not 0 < damping <= 1 or radius <= 0 or margin < 0:
        raise ValueError('Invalid trust controls')
    result, evidence, accepted, denominators = [], [], [], []
    nu = 10**(-snr/10)+(float(sigma)/max(float(alpha), 1e-8))**2
    for b, (r, zz) in enumerate(zip((yc-xc).flatten(1).split(block_size,1),
                                   zc.flatten(1).split(block_size,1))):
        current = old[:,b]
        fit, check = zz[:,::2], zz[:,1::2]
        rf, rc = r[:,::2], r[:,1::2]
        df = fit.abs().square().sum(1)
        dc = check.abs().square().sum(1)
        ridge = .01*max(fit.shape[1],1)
        proposal = (((fit.conj()*rf).real.sum(1)+ridge*anchor[:,b])/(df+ridge)).clamp(0,8)
        candidate = (current+damping*(proposal-current).clamp(-radius,radius)).clamp(0,8)
        gain = ((rc-current[:,None]*check).abs().square().sum(1)
                -(rc-candidate[:,None]*check).abs().square().sum(1)) / (max(check.shape[1],1)*nu)
        use = (gain>margin)&(df>1e-4*max(fit.shape[1],1))&(dc>1e-4*max(check.shape[1],1))
        result.append(torch.where(use,candidate,current))
        evidence.append(gain); accepted.append(use.to(current.dtype)); denominators.append(df)
    return torch.stack(result,1), dict(evidence=torch.stack(evidence,1),
            accepted=torch.stack(accepted,1), denominator=torch.stack(denominators,1))


def feedback_step(state, y, x, z, snr, alpha, sigma):
    controller, diag = trusted_amplitude(y,x,z,state['controller'],state['anchor'],
                                        state['block_size'],snr,alpha,sigma)
    state['controller'] = controller
    amplitude = state['amplitude'] if state['mode']=='B_GUIDE' else controller
    control_power = controller.square() if state['mode'] in {'B_GUIDE','B_COUPLED'} else state['initial_power']
    lam,beta = guidance_parameters(power_to_sinr(control_power,snr),'awgn')
    state['accepted_count'] += diag['accepted']
    state['attempt_count'] += 1
    state['diagnostics'] = diag
    state.update(amplitude=amplitude,power=amplitude.square(),**{'lambda':lam,'beta':beta})
    return amplitude,lam,beta


@torch.no_grad()
def selective_receive(sampler, snr, y, h, block_size, mode='B3', threshold=.15):
    if mode not in {'B1','B2','B3','B_GUIDE','B_COUPLED'}:
        raise ValueError(mode)
    if not math.isfinite(threshold) or threshold<0:
        raise ValueError('Invalid bypass threshold')
    if y.shape[0] != 1:
        raise ValueError('Use batch one for exact matched random initializations with bypass')
    if not torch.allclose(h,torch.ones_like(h)):
        raise ValueError('RG currently supports AWGN only')
    estimates=hierarchical_power(y,h,snr,block_size)
    power=estimates['power']
    estimates['initial_power']=power.clone()
    bypass=mode!='B1' and estimates['global_power'].item()<=threshold
    estimates['bypass']=power.new_full((1,1),float(bypass))
    lam,beta=guidance_parameters(power_to_sinr(power,snr),'awgn')
    estimates.update(**{'lambda':lam,'beta':beta})
    estimates['accepted_count']=torch.zeros_like(power)
    estimates['attempt_count']=torch.zeros_like(power)
    if bypass:
        return equalize(y,h,snr,'awgn'),torch.zeros_like(equalize(y,h,snr,'awgn')),estimates
    def mapped(value):
        v=expand_blocks(value,y,block_size)
        return torch.cat((v,v),2)
    state=None
    if mode in {'B3','B_GUIDE','B_COUPLED'}:
        state=dict(variant='RG',mode=mode,block_size=block_size,min_alpha=.8,
                   amplitude=power.sqrt(),anchor=power.sqrt().clone(),controller=power.sqrt().clone(),
                   initial_power=power.clone(),power=power.clone(),updates=0,
                   accepted_count=torch.zeros_like(power),attempt_count=torch.zeros_like(power))
    x,z=sampler.SIC_sampling(snr,None,equalize(y,h,snr,'awgn'),h,
            Lambda=mapped(lam),Beta=mapped(beta),amplitude_mode='power_consistent',
            interference_amp=mapped(power.sqrt()),point_calibration=state)
    if state is not None:
        estimates.update({k:state[k] for k in ['power','lambda','beta','accepted_count','attempt_count']})
        estimates['controller_power']=state['controller'].square()
        estimates.update(state.get('diagnostics',{}))
    return x,z,estimates
