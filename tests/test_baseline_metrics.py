import unittest
import torch
from scripts.baseline_metrics import ms_ssim_rgb
from scripts.cddm_receiver import cddm_receive


class BaselineTests(unittest.TestCase):
    def test_last_scale_luminance_is_used_once(self):
        # Constant images have contrast-structure=1 at every scale. Only the
        # last-scale luminance factor remains; it must not be cubed.
        x = torch.full((1, 3, 128, 128), .5, dtype=torch.float64)
        y = torch.full_like(x, .6)
        luminance = (2 * .5 * .6 + .01 ** 2) / (.5 ** 2 + .6 ** 2 + .01 ** 2)
        exponent = .2363 / sum([.0448, .2856, .3001, .2363])
        expected = x.new_tensor([luminance ** exponent])
        torch.testing.assert_close(ms_ssim_rgb(x, y), expected, atol=1e-10, rtol=1e-9)

    def test_identity_and_batch_separation(self):
        torch.manual_seed(9)
        x = torch.rand(2, 3, 128, 128)
        y = (x * .8 + .05).clamp(0, 1)
        torch.testing.assert_close(ms_ssim_rgb(x, x), torch.ones(2), atol=1e-5, rtol=0)
        torch.testing.assert_close(ms_ssim_rgb(x, y), torch.cat([ms_ssim_rgb(x[i:i+1], y[i:i+1]) for i in range(2)]))
        self.assertTrue(bool((ms_ssim_rgb(x, y) < 1).all()))

    def test_reference_five_scale(self):
        from pytorch_msssim import ms_ssim
        torch.manual_seed(10)
        x = torch.rand(2, 3, 256, 256)
        y = (x + .1 * torch.randn_like(x)).clamp(0, 1)
        torch.testing.assert_close(ms_ssim_rgb(x, y, levels=5), ms_ssim(x, y, data_range=1, size_average=False), atol=2e-6, rtol=1e-5)

    def test_cddm_perfect_predictor(self):
        torch.manual_seed(11)
        abar = torch.cumprod(1 - torch.linspace(.0001, .02, 1000), 0)
        clean = torch.randn(2, 8, 16, 16)
        start = 400
        nc = float(2 * (1 - abar[start]) / abar[start])
        y = clean + (nc / 2) ** .5 * torch.randn_like(clean)
        def oracle(state, t):
            a = abar[t].view(-1, 1, 1, 1)
            return (state - a.sqrt() * clean) / (1 - a).sqrt()
        rec, info = cddm_receive(oracle, y, nc, abar, steps=40)
        torch.testing.assert_close(rec, clean, atol=2e-6, rtol=1e-5)
        self.assertEqual(info["nfe"], 40)
        self.assertEqual(info["start_index"], start)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
