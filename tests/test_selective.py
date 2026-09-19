import os
import unittest
from types import SimpleNamespace as NS
import torch
from Diffusion.selective import hierarchical_power, trusted_amplitude, selective_receive
from Diffusion.calibration import to_stacked, guidance_parameters, power_to_sinr

class Tests(unittest.TestCase):
    def setUp(self):
        self.device=os.environ.get('ICDM_TEST_DEVICE','cpu')
    def test_pooling_and_short_block(self):
        y=torch.ones(1,1,1,17,device=self.device,dtype=torch.complex64)*2
        e=hierarchical_power(y,torch.ones_like(y),20,8)
        torch.testing.assert_close(e['power'],torch.full((1,3),2.99,device=self.device))
        self.assertTrue((e['local_weight']==0).all())
        y[...,8:16]=4
        e=hierarchical_power(y,torch.ones_like(y),20,8)
        self.assertTrue((e['heterogeneity']>0).all())
        self.assertGreater(e['power'][0,1].item(),e['power'][0,0].item())
        self.assertTrue(all(torch.isfinite(v).all() for v in e.values()))
    def test_feedback_rejection_and_bound(self):
        z=torch.ones(1,1,1,8,device=self.device,dtype=torch.complex64)
        x=torch.zeros_like(z); old=torch.ones(1,1,device=self.device)
        y=2*z
        value,d=trusted_amplitude(*map(to_stacked,(y,x,z)),old,old,8,20,1.,0.)
        self.assertTrue((value>old).all()); self.assertTrue((value-old<=.062501).all())
        # Even coordinates support increase, odd coordinates contradict it.
        y[...,1::2]=0
        value,d=trusted_amplitude(*map(to_stacked,(y,x,z)),old,old,8,20,1.,0.)
        torch.testing.assert_close(value,old)
        value,d=trusted_amplitude(*map(to_stacked,(y,x,torch.zeros_like(z))),old,old,8,20,1.,0.)
        torch.testing.assert_close(value,old)
    def test_sampler_bypass_and_decoupling(self):
        from Diffusion import ICDMSampler
        class Zero(torch.nn.Module):
            def __init__(self): super().__init__(); self.calls=0
            def forward(self,x,t): self.calls+=1; return torch.zeros_like(x)
        s,z=Zero(),Zero()
        cfg=NS(DATA=NS(TEST_BATCH=1,IMG_SIZE=8),MODEL=NS(OUT_CHANS=2,DEPTHS=[1]))
        sampler=ICDMSampler(s,z,cfg).to(self.device)
        y=torch.ones(1,2,2,4,device=self.device,dtype=torch.complex64)
        result,_,e=selective_receive(sampler,20,y,torch.ones_like(y),5,'B2')
        torch.testing.assert_close(result,to_stacked(y)); self.assertEqual(s.calls+z.calls,0)
        torch.manual_seed(12)
        y=3*torch.complex(torch.randn(1,2,2,4,device=self.device),torch.randn(1,2,2,4,device=self.device))
        runs={}
        for mode in ['B1','B2','B3','B_GUIDE','B_COUPLED']:
            torch.manual_seed(77)
            result,_,e=selective_receive(sampler,20,y,torch.ones_like(y),5,mode)
            self.assertTrue(torch.isfinite(result).all())
            self.assertTrue(all(torch.isfinite(v).all() for v in e.values()))
            runs[mode]=(result,e)
            if mode=='B3':
                lam,beta=guidance_parameters(power_to_sinr(e['initial_power'],20),'awgn')
                torch.testing.assert_close(e['lambda'],lam); torch.testing.assert_close(e['beta'],beta)
            if mode=='B_GUIDE': torch.testing.assert_close(e['power'],e['initial_power'])
        torch.testing.assert_close(runs['B1'][0],runs['B2'][0])
        self.assertTrue((runs['B3'][1]['attempt_count']>0).all())

if __name__=='__main__': unittest.main()
