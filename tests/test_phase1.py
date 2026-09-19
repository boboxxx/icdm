"""Numerical and integration checks; these are NOT trained-model experiments."""
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace as NS
import unittest

import torch
from Autoencoder.channel import Channel
from Diffusion import ICDMSampler
from Diffusion.calibration import (
    interference_amplitude, estimate_energy_power, expand_blocks,
    to_stacked, to_complex, guidance_parameters, estimate_point_amplitude,
    normalize_unit_complex_power,
)
from Diffusion.uni_pc import Joint_UniPC, NoiseScheduleVP


class ZeroNoise(torch.nn.Module):
    def forward(self, x, t):
        return torch.zeros_like(x)


class Phase1Tests(unittest.TestCase):
    def test_power_and_no_interference(self):
        for snr, sinr in [(20, 0), (5, 0), (20, 20)]:
            a = interference_amplitude(snr, sinr, "power_consistent")
            self.assertAlmostEqual(a * a, 10 ** (-sinr / 10) - 10 ** (-snr / 10))
        self.assertEqual(interference_amplitude(20, 20, "power_consistent"), 0)
        self.assertAlmostEqual(interference_amplitude(20, 20), .09)
        with self.assertRaises(ValueError):
            interference_amplitude(5, 20, "power_consistent")

    def test_pair_layout_and_tail(self):
        shape = torch.zeros(2, 1, 2, 5, dtype=torch.complex64)
        values = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
        mapped = expand_blocks(values, shape, 4)
        self.assertEqual(mapped.flatten(1).tolist()[0], [1]*4 + [2]*4 + [3]*2)
        torch.testing.assert_close(to_complex(to_stacked(shape)), shape)

    def test_point_estimate_unit_power_constraint(self):
        torch.manual_seed(2)
        value = torch.randn(3, 2, 4, 5) * torch.tensor([1., 2., 4.]).view(3, 1, 1, 1)
        normalized = to_complex(normalize_unit_complex_power(value))
        power = normalized.abs().square().flatten(1).mean(1)
        torch.testing.assert_close(power, torch.ones_like(power))

    def test_energy_clips_after_averaging_and_separates_frames(self):
        h = torch.ones(2, 1, 1, 2, dtype=torch.complex64)
        y = torch.tensor([[[[0., 2.]]], [[[2., 2.]]]], dtype=torch.complex64)
        result = estimate_energy_power(y, h, 0)
        torch.testing.assert_close(result, torch.tensor([[0.], [2.]]))

    def test_long_block_energy_consistency(self):
        torch.manual_seed(1)
        h = torch.ones(2, 1, 1, 100000, dtype=torch.complex64)
        def noise():
            return torch.complex(torch.randn(h.shape), torch.randn(h.shape)) / 2**.5
        powers = torch.tensor([.2, 2.]).reshape(2, 1, 1, 1)
        y = noise() + powers.sqrt() * noise() + .1 * noise()
        torch.testing.assert_close(estimate_energy_power(y, h, 20), powers.reshape(2, 1), atol=.04, rtol=0)

    def test_table_interpolation_and_clipping(self):
        lam, beta = guidance_parameters(torch.tensor([-100., -7., -2., 100.]), "awgn")
        torch.testing.assert_close(lam, torch.tensor([.3, .3, 1.2, 3.]))
        torch.testing.assert_close(beta, torch.tensor([.1, .1, .85, 2.]))

    def test_point_amplitude_recovers_blocks_batch_and_tail(self):
        torch.manual_seed(3)
        signal = torch.complex(torch.randn(2, 1, 1, 7), torch.randn(2, 1, 1, 7))
        interference = torch.complex(torch.randn(2, 1, 1, 7), torch.randn(2, 1, 1, 7))
        amplitudes = torch.tensor([[.2, 1.1, .7], [1.4, .0, .3]])
        mapped = expand_blocks(amplitudes, signal, 3)
        received = signal + mapped * interference
        actual = estimate_point_amplitude(
            to_stacked(received), to_stacked(signal), to_stacked(interference), 3
        )
        torch.testing.assert_close(actual, amplitudes, atol=1e-6, rtol=1e-6)

    def test_simulator_zero_blocks(self):
        channel = Channel(NS(CHANNEL=NS(TYPE="awgn")))
        x = torch.ones(1, 1, 4, 4)
        y, _, _, h = channel.inf_forward_blockwise(x, x, 200, torch.zeros(1, 3), 3)
        torch.testing.assert_close(y, to_complex(x / 2**.5), atol=1e-7, rtol=0)
        torch.testing.assert_close(h, torch.ones_like(h))

    def test_constant_block_simulator_matches_original_channel(self):
        device = os.environ.get("ICDM_TEST_DEVICE", "cpu")
        x, z = [torch.randn(1, 2, 4, 4, device=device) for _ in range(2)]
        for channel_type in ["awgn", "rayleigh"]:
            channel = Channel(NS(CHANNEL=NS(TYPE=channel_type)))
            torch.manual_seed(11)
            original = channel.inf_forward(x, z, 20, 0)
            torch.manual_seed(11)
            blockwise = channel.inf_forward_blockwise(x, z, 20, torch.full((1, 4), .99, device=device), 4)
            for old, new in zip(original, blockwise):
                torch.testing.assert_close(old, new, atol=1e-6, rtol=1e-6)

    def test_amplitude_map_matches_scalar_guidance(self):
        device = os.environ.get("ICDM_TEST_DEVICE", "cpu")
        torch.manual_seed(4)
        x, z, y = [torch.randn(1, 2, 4, 4, device=device) for _ in range(3)]
        for channel_type in ["awgn", "rayleigh"]:
            h = torch.ones(1, 2, 2, 4, dtype=torch.complex64, device=device)
            if channel_type == "rayleigh":
                h = h + .3j
            results = []
            for amplitude in [None, torch.full_like(x, interference_amplitude(20, 0, "power_consistent"))]:
                solver = Joint_UniPC(lambda x, t: .1*x, lambda x, t: .1*x, NoiseScheduleVP("linear"), algorithm_type="noise_prediction", amplitude_mode="power_consistent", interference_amp=amplitude)
                results.append(solver.sample(x.clone(), z.clone(), steps=40, order=3, skip_type="logSNR", SNR=20, SINR=0 if amplitude is None else None, deg_feature=y, h=h))
            for scalar, mapped in zip(*results):
                torch.testing.assert_close(scalar, mapped)

    def test_original_output_matches_unmodified_commit(self):
        root = Path(__file__).resolve().parents[1]
        try:
            source = subprocess.check_output(
                ["git", "show", "e6b7f8d:Diffusion/uni_pc.py"],
                cwd=root,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError:
            self.skipTest("upstream reference commit e6b7f8d is not included in a source-only clone")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "original.py"
            path.write_bytes(source)
            spec = importlib.util.spec_from_file_location("original_unipc", path)
            original = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(original)
            for channel_type in ["awgn", "rayleigh"]:
                torch.manual_seed(7)
                device = os.environ.get("ICDM_TEST_DEVICE", "cpu")
                x, z, y = [torch.randn(1, 2, 4, 4, device=device) for _ in range(3)]
                h = torch.ones(1, 2, 2, 4, dtype=torch.complex64, device=device)
                if channel_type == "rayleigh":
                    h = h + .3j
                outputs = []
                for cls in [original.Joint_UniPC, Joint_UniPC]:
                    solver = cls(lambda x, t: .1*x, lambda x, t: .1*x, NoiseScheduleVP("linear"), algorithm_type="noise_prediction")
                    outputs.append(solver.sample(x.clone(), z.clone(), steps=40, order=3, skip_type="logSNR", SNR=20, SINR=0, deg_feature=y, h=h))
                for old, new in zip(*outputs):
                    torch.testing.assert_close(old, new, atol=0, rtol=0)

    def test_blind_sampler_40_steps_both_channels_batch_and_tail(self):
        config = NS(DATA=NS(TEST_BATCH=1, IMG_SIZE=8), MODEL=NS(OUT_CHANS=2, DEPTHS=[1]))
        device = os.environ.get("ICDM_TEST_DEVICE", "cpu")
        sampler = ICDMSampler(ZeroNoise(), ZeroNoise(), config).to(device)
        for channel_type in ["awgn", "rayleigh"]:
            for block_size in [None, 5]:
                torch.manual_seed(5)
                received = torch.complex(torch.randn(2, 2, 2, 4), torch.randn(2, 2, 2, 4)).to(device)
                h = torch.ones_like(received)
                if channel_type == "rayleigh":
                    h = h + .3j
                x, z, estimates = sampler.SIC_sampling_estimated(20, received, h, channel_type, block_size)
                self.assertEqual(x.shape, (2, 2, 4, 4))
                self.assertTrue(torch.isfinite(x).all() and torch.isfinite(z).all())
                self.assertEqual(estimates["power"].shape, (2, 1 if block_size is None else 4))

    def test_a4_sampler_updates_without_oracle_inputs(self):
        config = NS(DATA=NS(TEST_BATCH=1, IMG_SIZE=8), MODEL=NS(OUT_CHANS=2, DEPTHS=[1]))
        device = os.environ.get("ICDM_TEST_DEVICE", "cpu")
        sampler = ICDMSampler(ZeroNoise(), ZeroNoise(), config).to(device)
        torch.manual_seed(9)
        received = torch.complex(
            torch.randn(2, 2, 2, 4, device=device),
            torch.randn(2, 2, 2, 4, device=device),
        )
        h = torch.ones_like(received)
        x, z, estimates = sampler.SIC_sampling_alternating(
            20, received, h, "awgn", block_size=5, min_alpha=.01
        )
        self.assertEqual(x.shape, (2, 2, 4, 4))
        self.assertTrue(torch.isfinite(x).all() and torch.isfinite(z).all())
        self.assertGreater(estimates["update_count"][0, 0].item(), 0)
        self.assertEqual(estimates["power"].shape, (2, 4))


if __name__ == "__main__":
    unittest.main()
