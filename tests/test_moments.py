"""Independent analytic and finite-search checks for the declared surrogate."""
import os
import unittest
from types import SimpleNamespace as NS
import torch
from Diffusion.moment_calibration import (
    gaussian_joint_moments, quadratic_statistics, bounded_moment_update,
)
from Diffusion.calibration import normalize_unit_complex_power


class MomentTests(unittest.TestCase):
    def setUp(self):
        self.device = os.environ.get("ICDM_TEST_DEVICE", "cpu")
        self.dtype = torch.float64

    def tensor(self, value):
        return torch.tensor(value, dtype=self.dtype, device=self.device)

    def test_against_dense_gaussian_conditioning(self):
        # Independent matrix calculation, not a restatement of scalar formula.
        for a in [0., .3, 2., 7.]:
            mean = self.tensor([.7, -.4])
            cov = torch.diag(self.tensor([.4, .2]))
            H = self.tensor([[1., a]])
            y, nu = self.tensor([1.3]), .015
            gain = cov @ H.T @ torch.linalg.inv(H @ cov @ H.T + self.tensor([[nu]]))
            posterior_mean = mean + gain @ (y-H @ mean)
            posterior_cov = cov-gain @ H @ cov
            mx,mz,cxx,czz,cxz=gaussian_joint_moments(y,mean[0],mean[1],self.tensor(a),.4,.2,nu)
            torch.testing.assert_close(torch.cat((mx,mz)),posterior_mean)
            actual = torch.stack((torch.stack((cxx,cxz)),torch.stack((cxz,czz))))
            torch.testing.assert_close(actual,posterior_cov)
            self.assertGreaterEqual(torch.linalg.eigvalsh(actual).min().item(),-1e-12)

    def test_quadratic_minimum_matches_grid_and_cross_term(self):
        torch.manual_seed(74)
        y,dx,dz=[torch.randn(2,1,4,3,device=self.device,dtype=self.dtype) for _ in range(3)]
        mx,mz,_,cz,cxz=gaussian_joint_moments(y,dx,dz,torch.full_like(y,.8),.3,.2,.01)
        n,d=quadratic_statistics(y,mx,mz,cz,cxz,5)
        grid=torch.linspace(0,8,100001,device=self.device,dtype=self.dtype)
        # Direct expected residual for the first block, pairing I/Q coordinates.
        def block(x):
            re,im=x.chunk(2,2)
            return torch.cat((re[0].flatten()[:5],im[0].flatten()[:5]))
        yy,xx,zz,cc,cross=map(block,(y,mx,mz,cz,cxz))
        residual=yy[None,:]-xx[None,:]-grid[:,None]*zz[None,:]
        objective=(residual.square()+grid[:,None].square()*cc[None,:]+2*grid[:,None]*cross[None,:]).sum(1)
        optimum=(n[0,0]/d[0,0]).clamp(0,8)
        self.assertLessEqual(abs(grid[objective.argmin()].item()-optimum.item()),8.1e-5)
        self.assertTrue((cxz < 0).all())

    def test_stabilized_update_bounds_tail_and_descent(self):
        torch.manual_seed(42)
        y,dx,dz=[torch.randn(3,2,4,3,device=self.device,dtype=self.dtype) for _ in range(3)]
        dx,dz=map(normalize_unit_complex_power,(dx,dz))
        old=torch.rand(3,3,device=self.device,dtype=self.dtype)*4
        for variant in ['A4_R','A5_DIAG','A5_FULL']:
            value,diag=bounded_moment_update(y,dx,dz,old,old,5,self.tensor(.7),self.tensor(.5),20,variant)
            self.assertEqual(value.shape,(3,3))
            self.assertTrue(((value>=0)&(value<=8)).all())
            self.assertLessEqual(diag['objective_change'].max().item(),1e-10)
            self.assertTrue(torch.isfinite(value).all())

    def test_zero_uncertainty_reduces_to_regularized_point(self):
        torch.manual_seed(19)
        y,dx,dz=[torch.randn(2,1,2,4,device=self.device,dtype=self.dtype) for _ in range(3)]
        old=torch.ones(2,2,device=self.device,dtype=self.dtype)
        outputs=[bounded_moment_update(y,dx,dz,old,old,3,self.tensor(1.),self.tensor(0.),20,v)[0]
                 for v in ['A4_R','A5_DIAG','A5_FULL']]
        for x in outputs[1:]: torch.testing.assert_close(x,outputs[0])

    def test_all_new_variants_integrate_with_sampler(self):
        from Diffusion import ICDMSampler
        class ZeroNoise(torch.nn.Module):
            def forward(self, x, t):
                return torch.zeros_like(x)
        cfg = NS(DATA=NS(TEST_BATCH=1, IMG_SIZE=8), MODEL=NS(OUT_CHANS=2, DEPTHS=[1]))
        sampler = ICDMSampler(ZeroNoise(), ZeroNoise(), cfg).to(self.device)
        torch.manual_seed(9)
        y = torch.complex(torch.randn(2,2,2,4,device=self.device),
                          torch.randn(2,2,2,4,device=self.device))
        for variant in ['A4_R','A5_DIAG','A5_FULL']:
            torch.manual_seed(42)
            x,z,e = sampler.SIC_sampling_alternating(20,y,torch.ones_like(y),'awgn',5,variant=variant)
            self.assertTrue(torch.isfinite(x).all() and torch.isfinite(z).all())
            self.assertEqual(e['power'].shape,(2,4))
            self.assertTrue((e['update_count'] == 21).all())
            self.assertTrue(((e['power'] >= 0) & (e['max_power_seen'] <= 64)).all())
            self.assertLessEqual(e['max_objective_increase'].max().item(),1e-3)
            self.assertTrue(all(torch.isfinite(v).all() for v in e.values()))


if __name__ == '__main__':
    unittest.main()
