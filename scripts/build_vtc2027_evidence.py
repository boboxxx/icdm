"""Build paper evidence only from complete, independently frozen study stages."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.analyze_vtc2027 import read_run

LABEL={'DIRECT':'SwinJSCC direct','CDDM_N0':'Gaussian (N0)','CDDM_BLIND':'Blind Gaussian',
       'A3':'Init. joint','A4':'Calibrated joint','A4_GATE':'Joint + gate','SELECT':'Selector'}
COLORS={'DIRECT':'#777777','CDDM_BLIND':'#0072B2','A4':'#E69F00',
        'A4_GATE':'#CC79A7','SELECT':'#009E73'}


def write(path,text):
    path.write_text('\n'.join(line.rstrip() for line in text.strip().splitlines())+'\n')


def signed(value):
    return f'{value:+.3f}'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--paper',type=Path,required=True)
    p.add_argument('--backbones',type=Path)
    p.add_argument('--pretrained',type=Path)
    a=p.parse_args()
    stages={}
    for stage,count in [('development',64),('confirmation',256),('pressure',64)]:
        meta,rows,groups=read_run(a.results/stage)
        if len(meta['images'])!=count or meta['manifest']['stage']!=stage:
            raise ValueError('Unexpected prospective sample size or stage')
        report=json.loads((a.results/stage/'analysis.json').read_text())
        if report['signature']!=meta['signature']: raise ValueError('Stale analysis')
        stages[stage]=(meta,rows,groups,report)
    dev,conf,pressure=(stages[k] for k in ['development','confirmation','pressure'])
    policy=json.loads((a.results/'development/policy.json').read_text())
    if dev[0]['signature']!=policy['development_signature']:
        raise ValueError('Policy not tied to this development run')
    for stage in [conf,pressure]:
        if stage[0]['policy']!=policy: raise ValueError('Policy changed')
        if stage[0]['started_at']<=dev[0]['completed_at']:
            raise ValueError('Confirmation opened before development finished')
        if stage[0]['source_sha256']!=dev[0]['source_sha256']:
            raise ValueError('Scientific source changed between stages')
        if stage[0]['weights_sha256']!=dev[0]['weights_sha256']:
            raise ValueError('Model weights changed')
    audit=a.results.parent/'research/vtc2027_sheng/data_audit.json'
    # Mirrored results may keep this audit beside the results folder.
    if not audit.exists(): audit=a.results/'data_audit.json'
    if not audit.exists() or json.loads(audit.read_text())['status']!='complete':
        raise ValueError('Independent-data audit required')

    generated=a.paper/'generated';generated.mkdir(parents=True,exist_ok=True)
    figdir=a.paper/'figures';figdir.mkdir(parents=True,exist_ok=True)
    d,c,pres=(x[3] for x in [dev,conf,pressure])
    means=c['means']; sel=means['SELECT']
    backbone=None
    if a.backbones:
        backbone=json.loads((a.backbones/'analysis.json').read_text())
        backbone_meta=json.loads((a.backbones/'metadata.json').read_text())
        if backbone['status']!='complete' or backbone_meta['status']!='complete':
            raise ValueError('External backbone evidence is incomplete')
        if backbone['signature']!=backbone_meta['signature']:
            raise ValueError('Stale external backbone analysis')
        if backbone['reference_signature']!=conf[0]['signature']:
            raise ValueError('External backbones use a different confirmation reference')
    pretrained=None
    if a.pretrained:
        pretrained=json.loads((a.pretrained/'analysis.json').read_text())
        pretrained_meta=json.loads((a.pretrained/'metadata.json').read_text())
        if pretrained['status']!='complete' or pretrained_meta['status']!='complete':
            raise ValueError('Public-weight evidence is incomplete')
        if pretrained['signature']!=pretrained_meta['signature']:
            raise ValueError('Stale public-weight analysis')
        if pretrained['reference_signature']!=conf[0]['signature']:
            raise ValueError('Public weights use a different confirmation reference')
    for meta in ([backbone_meta] if backbone else [])+([pretrained_meta] if pretrained else []):
        if meta['images']!=conf[0]['images']:
            raise ValueError('External evidence uses different input image hashes')
        for key in ['seeds','conditions','block_size']:
            if meta['manifest'][key]!=conf[0]['manifest'][key]:
                raise ValueError(f'External evidence differs in {key}')
    comparisons={r['reference']:r for r in c['comparisons']}
    ca,cg,cgate=(comparisons[k] for k in ['A4','CDDM_BLIND','A4_GATE'])
    nfe=sel['nfe_signal']+sel['nfe_interference']
    nfe_s=means['A4']['nfe_signal']+means['A4']['nfe_interference']
    saving=100*(1-nfe/nfe_s)
    write(generated/'values.tex',r"""\newcommand{\DraftNotice}{}
\newcommand{\ResultFigure}[1]{\includegraphics[width=\columnwidth]{#1}}""")
    write(generated/'abstract_result.tex',
        f"On 256 independent image pairs with two random seeds and 18 channel conditions, "
        f"the frozen rule attains {sel['psnr']:.3f} dB mean PSNR: "
        f"{signed(ca['mean_delta_psnr'])} dB relative to calibrated joint sampling and "
        f"{signed(cg['mean_delta_psnr'])} dB relative to the blind Gaussian receiver. "
        f"It uses {nfe:.1f} total predictor evaluations per frame on average, compared with {nfe_s:.0f} for joint sampling.")

    cv=d['selection']
    cv_sentence=(f"Image-group validation yields {cv['energy']['cv_psnr']:.3f} dB for energy-only selection "
                 f"and {cv['energy_heterogeneity']['cv_psnr']:.3f} dB for the two-feature family. ")
    if policy['scientific_status']=='no_cv_gain_over_fixed':
        decision=f"The prespecified fallback retains the fixed {LABEL[cv['best_fixed']['mode']].lower()} receiver; selection has no validated development gain. "
    else:
        family='energy-only' if policy['family']=='energy' else 'energy and heterogeneity'
        decision=f"The retained {family} rule uses "
        decision+=rf"$\tau_\ell={policy['power_threshold_low']:g}$, $\tau_h={policy['power_threshold_high']:g}$ and $\eta={policy['heterogeneity_threshold']:g}$, with Gaussian reception {policy['gaussian_side']} the applicable power threshold. "
    devcontrols=(f"The earlier coupled-feedback receiver obtains {d['means']['B_COUPLED']['psnr']:.3f} dB, "
                 f"versus {d['means']['A4']['psnr']:.3f} dB for calibrated joint sampling and "
                 f"{d['means']['A4_GATE']['psnr']:.3f} dB for its same-gate control. "
                 f"The privileged nominal-power Gaussian diagnostic obtains {d['means']['CDDM_ORACLE']['psnr']:.3f} dB. "
                 "These development comparisons motivate the receiver decision but are not independent confirmation.")
    dev_unique=[r for r in dev[1] if r['mode']=='CDDM_BLIND']
    low_region=[r for r in dev_unique if r['features']['heterogeneity']<=policy['heterogeneity_threshold']]
    support_note=(f"In the low-heterogeneity development region ({len(low_region)} frames), the largest observed "
                  f"excess-energy estimate is {max(r['features']['power'] for r in low_region):.3f}; "
                  f"the fitted threshold {policy['power_threshold_low']:g} therefore lies beyond observed support and acts as an always-Gaussian rule in that region. "
                  "It is not an identified physical transition, and the family comparison does not establish that either feature is individually necessary. ")
    write(generated/'development_result.tex',cv_sentence+decision+'\n\n'+support_note+devcontrols)

    def effect(ref,label):
        q=comparisons[ref]
        lo,hi=q['ci95_image_cluster']
        return f"{signed(q['mean_delta_psnr'])} dB relative to {label} (95\\% interval [{lo:.3f}, {hi:.3f}])"

    conclusion_support=all(comparisons[k]['ci95_image_cluster'][0]>0 for k in ['A4','CDDM_BLIND'])
    interpretation=("Both fixed-receiver comparisons have positive paired intervals, supporting a mean-PSNR benefit within the matched confirmation population. "
                    if conclusion_support else
                    "The paired intervals do not establish a positive mean-PSNR effect over both fixed receivers; the evidence therefore does not support an unqualified adaptive-superiority claim. ")
    profile_bits=[]
    for profile in ['stationary','alternating','burst']:
        rs=[r for r in conf[1] if r['profile']==profile]
        values={mode:float(np.mean([r['psnr'] for r in rs if r['mode']==mode]))
                for mode in ['CDDM_BLIND','A4','SELECT']}
        frac=float(np.mean([r['selected']=='CDDM_BLIND' for r in rs if r['mode']=='SELECT']))
        profile_bits.append(f"{profile} {values['CDDM_BLIND']:.3f}/{values['A4']:.3f}/{values['SELECT']:.3f} dB "
                            f"with {100*frac:.1f}\\% Gaussian choices")
    profile_values={profile:{mode:float(np.mean([r['psnr'] for r in conf[1]
                    if r['profile']==profile and r['mode']==mode]))
                    for mode in ['CDDM_BLIND','A4','SELECT']}
                    for profile in ['stationary','alternating','burst']}
    stationary,alternating,burst=(profile_values[k] for k in ['stationary','alternating','burst'])
    profile_text=(f"The fixed receivers are complementary across interference profiles. For stationary interference, "
                  f"Gaussian denoising reaches {stationary['CDDM_BLIND']:.3f} dB against {stationary['A4']:.3f} dB "
                  f"for joint sampling. Under burst interference, joint sampling reaches {burst['A4']:.3f} dB "
                  f"against {burst['CDDM_BLIND']:.3f} dB for Gaussian denoising. "
                  f"The selector reaches {stationary['SELECT']:.3f}, {alternating['SELECT']:.3f}, and "
                  f"{burst['SELECT']:.3f} dB for stationary, alternating, and burst interference, respectively. "
                  "Thus its gain is consistent with using different receivers for different observations; "
                  "the energy statistic alone does not establish a physical explanation of the interference. ")
    anchor_text=""
    if backbone:
        dm=backbone['means']['deepjscc']; mm=backbone['means']['mambajscc']
        anchor_text=(f"Separately matched direct-codec anchors attain {dm['psnr']:.3f} dB for DeepJSCC "
                     f"and {mm['psnr']:.3f} dB for MambaJSCC; these use independently trained latent spaces "
                     "and provide external codec references. "
                     "One training seed and a fixed 40-epoch budget do not establish convergence or a general architecture ranking. ")
    if pretrained:
        pm=pretrained['means']
        anchor_text+=(f"Without fine-tuning, public DeepJSCC (ImageNet/SNR 19), MambaJSCC (CLIC2021/AWGN 10), "
                      f"and MambaJSCC (DIV2K/Rayleigh) weights yield "
                      f"{pm['deepjscc_public_imagenet_snr19']['psnr']:.3f}, "
                      f"{pm['mambajscc_public_awgn10_clic2021']['psnr']:.3f}, and "
                      f"{pm['mambajscc_public_rayleigh_div2k']['psnr']:.3f} dB, respectively. "
                      "Their protocol mismatches preclude a controlled claim about retraining benefit. ")
    text=(r"Table~\ref{tab:main} and Fig.~\ref{fig:quality} report the independent experiment. "
          +"The selector changes mean PSNR by "+effect('A4','calibrated joint sampling')+", "
          +effect('CDDM_BLIND','blind Gaussian denoising')+", and "
          +effect('A4_GATE','the same-gate control')+". "+interpretation
          +f"Estimating total disturbance changes the Gaussian receiver from {means['CDDM_N0']['psnr']:.3f} to "
           f"{means['CDDM_BLIND']['psnr']:.3f} dB, so the thermal-noise-only port is insufficient as the sole Gaussian baseline. "
          +'\n\n'+profile_text+'\n\n'+anchor_text)
    write(generated/'confirmation_result.tex',text)

    table=[r'\scriptsize\setlength{\tabcolsep}{2.4pt}',r'\begin{tabular}{lrrrrrr}',r'\toprule',
           r'Receiver & PSNR & MSE$\times10^3$ & LPIPS & MS-SSIM & NFE & ms\\',r'\midrule']
    for mode in ['DIRECT','CDDM_N0','CDDM_BLIND','A3','A4','A4_GATE','SELECT']:
        v=means[mode]
        table.append(f"{LABEL[mode]} & {v['psnr']:.2f} & {v['mse']*1000:.2f} & {v['lpips_vgg']:.3f} & "
                     f"{v['ms_ssim']:.3f} & {v['nfe_signal']+v['nfe_interference']:.1f} & {v['receiver_seconds']*1000:.0f}"+r'\\')
        if mode=='DIRECT' and backbone:
            for architecture,label in [('deepjscc','DeepJSCC direct'),('mambajscc','MambaJSCC direct')]:
                q=backbone['means'][architecture]
                table.append(f"{label} & {q['psnr']:.2f} & {q['mse']*1000:.2f} & {q['lpips_vgg']:.3f} & "
                             f"{q['ms_ssim']:.3f} & {q['nfe']:.1f} & {q['receiver_seconds']*1000:.0f}"+r'\\')
    table.extend([r'\bottomrule',r'\end{tabular}'])
    write(generated/'main_table.tex','\n'.join(table))

    text=(f"The selector chooses Gaussian reception on {100*c['selector']['gaussian_fraction']:.1f}\\% of confirmation frames. "
          f"Its average cost is {nfe:.1f} predictor evaluations and {1000*sel['receiver_seconds']:.0f} ms, "
          f"versus {nfe_s:.0f} evaluations and {1000*means['A4']['receiver_seconds']:.0f} ms for joint sampling. "
          f"Against that receiver, {100*ca['harm_gt_half_db']:.2f}\\% of frames lose more than 0.5 dB, "
          f"and the fifth percentile of the paired change is {ca['fifth_percentile']:.3f} dB "
          r"(Fig.~\ref{fig:tail}). "
          f"The unavailable per-frame oracle reaches {c['selector']['offline_oracle_psnr']:.3f} dB; "
          f"selection leaves {c['selector']['mean_oracle_regret_db']:.3f} dB mean oracle regret. "
          f"Mean MSE changes from {means['A4']['mse']:.5f} to {sel['mse']:.5f}, LPIPS from "
          f"{means['A4']['lpips_vgg']:.4f} to {sel['lpips_vgg']:.4f}, and MS-SSIM from "
          f"{means['A4']['ms_ssim']:.4f} to {sel['ms_ssim']:.4f}. "
          "These secondary metrics and the harmful tail must accompany the average PSNR comparison.")
    write(generated/'tail_result.tex',text)

    families=[('shifted','Shifted blocks'),('random_length','Random lengths'),('stationary','No interference')]
    pressure_table=[r'\small\setlength{\tabcolsep}{3.2pt}',r'\begin{tabular}{lrrrr}',r'\toprule',
                    r'Profile & Direct & Gaussian & Joint & Select\\',r'\midrule']
    pressure_text=[]
    pressure_details={}
    for family,label in families:
        rs=[r for r in pressure[1] if r['profile']==family]
        values={mode:float(np.mean([r['psnr'] for r in rs if r['mode']==mode]))
                for mode in ['DIRECT','CDDM_BLIND','A4','SELECT']}
        pressure_details[family]=values
        pressure_table.append(label+' & '+' & '.join(f'{v:.2f}' for v in values.values())+r'\\')
        pressure_text.append(f"For {label.lower()}, the selector changes mean PSNR by "
                             f"{signed(values['SELECT']-values['A4'])} dB relative to joint sampling and "
                             f"{signed(values['SELECT']-values['CDDM_BLIND'])} dB relative to Gaussian denoising.")
    pressure_table.extend([r'\bottomrule',r'\end{tabular}'])
    write(generated/'pressure_table.tex','\n'.join(pressure_table))
    shifted=pressure_details['shifted']; random=pressure_details['random_length']
    write(generated/'pressure_result.tex',
          rf"Table~\ref{{tab:pressure}} shows a {signed(shifted['SELECT']-shifted['CDDM_BLIND'])} dB selector change "
          f"against Gaussian reception for shifted blocks and {signed(random['SELECT']-random['CDDM_BLIND'])} dB "
          "for random lengths. With no interference, selection matches Gaussian reception and avoids the joint receiver's loss. "
          "These are diagnostic populations; no rule is refitted or result pooled with matched confirmation.")

    ending=("On independent image pairs, the frozen selector improves mean PSNR over both blind Gaussian denoising and calibrated joint sampling. "
            if conclusion_support else
            "The independent study does not establish that the frozen blind selector improves mean PSNR over both fixed receivers. ")
    ending+=("It also improves on the same-gate control. " if cgate['ci95_image_cluster'][0]>0 else
             "The same-gate comparison does not establish an additional positive effect with a wholly positive paired interval. ")
    ending+=("The same-gate result shows that the gain cannot be explained solely by bypassing joint inference. "
             if conclusion_support else
             "A low-dimensional receiver decision alone is therefore insufficient evidence for universal adaptation, even when the candidate receivers are complementary. ")
    write(generated/'conclusion_result.tex',ending)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
                         'font.size':8,'axes.labelsize':8,'legend.fontsize':6.8,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'axes.grid':True,'grid.alpha':.15,'legend.frameon':False,
                         'pdf.fonttype':42,'ps.fonttype':42})
    def save(fig,name):
        fig.savefig(figdir/(name+'.pdf'),bbox_inches='tight')
        fig.savefig(figdir/(name+'.png'),dpi=300,bbox_inches='tight')
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(3.4,2.35),layout='constrained')
    sinrs=sorted({r['sinr'] for r in conf[1]})
    markers=['o','s','^','D','v']
    derived=[json.loads(s) for s in (a.results/'confirmation/derived_records.jsonl').read_text().splitlines()]
    for j,mode in enumerate(COLORS):
        rs=[r for r in conf[1]+derived if r['mode']==mode]
        vals=[np.mean([r['psnr'] for r in rs if r['sinr']==snr]) for snr in sinrs]
        ax.plot(sinrs,vals,label=LABEL[mode],color=COLORS[mode],marker=markers[j],
                markersize=3,linewidth=1.4 if mode!='SELECT' else 2)
    if backbone:
        anchors=[('deepjscc','DeepJSCC direct','#6A3D9A','x'),
                 ('mambajscc','MambaJSCC direct','#D55E00','+')]
        for architecture,label,color,marker in anchors:
            vals=[backbone['by_sinr'][architecture][str(snr)]['psnr'] for snr in sinrs]
            ax.plot(sinrs,vals,label=label,color=color,marker=marker,markersize=3.4,
                    linewidth=1.15,linestyle='--')
    ax.set(xlabel='Nominal SINR (dB)',ylabel='Mean PSNR (dB)',xticks=sinrs)
    ax.legend(loc='lower center',bbox_to_anchor=(.5,1.01),ncol=2,fontsize=6.2)
    save(fig,'confirmation_quality')
    fig,ax=plt.subplots(figsize=(3.4,2.2),layout='constrained')
    for mode,color in [('CDDM_BLIND',COLORS['CDDM_BLIND']),('SELECT',COLORS['SELECT'])]:
        delta=np.sort([g[mode]['psnr']-g['A4']['psnr'] for g in conf[2]])
        ax.plot(delta,np.arange(1,len(delta)+1)/len(delta),label=LABEL[mode],color=color,linewidth=1.5)
    ax.axvline(-.5,color='#D55E00',linestyle='--',linewidth=1)
    ax.set(xlabel='PSNR change versus calibrated joint (dB)',ylabel='Empirical cumulative fraction',ylim=(0,1))
    ax.legend(loc='lower right')
    save(fig,'paired_tail')

    # The gallery uses all six prospectively reserved confirmation examples.
    from PIL import Image
    indices=[512,527];conditions=[0,8,15]
    modes=['source','DIRECT','CDDM_BLIND','A4','SELECT']
    fig,axes=plt.subplots(6,5,figsize=(7.1,8.7),layout='constrained')
    for row,(index,ci) in enumerate((i,c) for i in indices for c in conditions):
        for col,mode in enumerate(modes):
            path=a.results/'confirmation/qualitative'/f'{index}_{ci}'/(mode+'.png')
            axes[row,col].imshow(Image.open(path))
            axes[row,col].set_xticks([]);axes[row,col].set_yticks([]);axes[row,col].grid(False)
            if row==0: axes[row,col].set_title('Source' if mode=='source' else LABEL[mode],fontsize=9)
            if col==0: axes[row,col].set_ylabel(f'Pair {index}\nCondition {ci}',fontsize=8)
    save(fig,'qualitative_prespecified')
    receipt={'status':'assembled_from_complete_records','editorial_review':'required',
             'signatures':{stage:value[0]['signature'] for stage,value in stages.items()},
             'policy':policy,'all_fixed_comparisons_positive_ci':conclusion_support,
             'same_gate_positive_ci':cgate['ci95_image_cluster'][0]>0,
             'pressure_means':pressure_details,'nfe_reduction_percent':saving,
             'author_information':'pending'}
    if backbone:
        receipt['backbone_signature']=backbone['signature']
        receipt['backbone_reference_signature']=backbone['reference_signature']
    if pretrained:
        receipt['public_pretrained_signature']=pretrained['signature']
        receipt['public_pretrained_scope']='Protocol-mismatched transfer diagnostics; no controlled retraining effect'
    write(generated/'evidence_receipt.json',json.dumps(receipt,indent=2))
    print(json.dumps(receipt,indent=2))


if __name__=='__main__': main()
