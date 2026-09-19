import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from Diffusion import sampling
from Diffusion import sde_lib

# from Diffusion.Autoencoder import noise_encoder
def sigmoid_schedule(T, start=-3, end=3, tau=1.0, clip_min=1e-9):
    t = torch.linspace(0, T, T).double()
    start = torch.tensor(start)
    end = torch.tensor(end)
    v_start = torch.sigmoid(start / tau)
    v_end = torch.sigmoid(end / tau)
    output = torch.sigmoid((t / T * (end - start) + start) / tau)
    output = (v_end - output) / (v_end - v_start)
    return torch.clip(output, clip_min, 1.0)


def cosine_schedule(T, start=-3, end=3, tau=1.0, clip_min=1e-9):
    t = torch.linspace(0, T, T).double()
    start = torch.tensor(start)
    end = torch.tensor(end)
    v_start = torch.cos(start * math.pi / 2) ** (2 * tau)
    v_end = torch.cos(end * math.pi / 2) ** (2 * tau)
    output = torch.cos((t / T * (end - start) + start) * math.pi / 2) ** (2 * tau)
    output = (v_end - output) / (v_end - v_start)
    return torch.clip(output, clip_min, 1.0)


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    v = v.to(device)
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class ICDMTrainer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, ICDM, diffusion, channel_type, h, train_for):
        """
        Algorithm 1.
        """
        device = x.device
        t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)

        # print(torch.mean(x**2))
        loss_dict = diffusion.training_losses(ICDM, x, t, channel_type, h, train_for=train_for)
        loss = loss_dict["loss"].mean()

        # print(t,torch.mean(out))
        # out = out.reshape(B, H, W)

        return loss


class ICDMSampler(nn.Module):
    def __init__(self, model_s, model_z, config):
        super().__init__()
        self.model_s = model_s
        self.model_z = model_z
        sde = sde_lib.VPSDE(beta_min=0.1, beta_max=20, N=10)
        sampling_eps = 1e-3
        sampling_shape = (
            config.DATA.TEST_BATCH,
            config.MODEL.OUT_CHANS,
            config.DATA.IMG_SIZE // (2 ** len(config.MODEL.DEPTHS)),
            config.DATA.IMG_SIZE // (2 ** len(config.MODEL.DEPTHS)),
        )
        inverse_scaler = lambda x: x

        self.joint_sampling_fn = sampling.get_sampling_fn(
            config,
            sde,
            sampling_shape,
            inverse_scaler,
            sampling_eps,
            sampler_name="joint_unipc",
        )

        self.random_sampling_fn = sampling.get_sampling_fn(
            config,
            sde,
            sampling_shape,
            inverse_scaler,
            sampling_eps,
            sampler_name="uni_pc",
        )

    def rand_generate(self):

        samples, n = self.random_sampling_fn(self.model_s)

        return samples  # torch.clip(x_0, 0, 1)

    def mask_generate(self, A, Ap, deg_feature):
        samples, n = self.random_sampling_fn(self.model_s, A=A, Ap=Ap, deg_feature=deg_feature)

        return samples

    def SIC_sampling(self, SNR, SINR, deg_feature, h, Lambda=1, Beta=1, return_intermediate=False, amplitude_mode="original", interference_amp=None, point_calibration=None):

        x_0, z_0 = self.joint_sampling_fn(
            self.model_s,
            self.model_z,
            SNR,
            SINR,
            deg_feature,
            h=h,
            Lambda=Lambda,
            Beta=Beta,
            return_intermediate=return_intermediate,
            amplitude_mode=amplitude_mode,
            interference_amp=interference_amp,
            point_calibration=point_calibration,
        )
        return x_0, z_0



    def SIC_sampling_estimated(self, SNR, received, h, channel_type, block_size=None):
        """A2 (global) / A3 (fixed blockwise energy); never takes true SINR.

        Returns recovered signal, recovered interference, and receiver estimates.
        A3 is initialization-only, not iterative self-calibration (A4/A5).
        """
        from Diffusion.calibration import (
            estimate_energy_power, expand_blocks, power_to_sinr,
            guidance_parameters, equalize,
        )
        power = estimate_energy_power(received, h, SNR, block_size)
        block_size = received[0].numel() if block_size is None else block_size
        sinr = power_to_sinr(power, SNR)
        lam, beta = guidance_parameters(sinr, channel_type)
        def real_map(values):
            mapped = expand_blocks(values, received, block_size)
            return torch.cat((mapped, mapped), dim=2)
        x, z = self.SIC_sampling(
            SNR, None, equalize(received, h, SNR, channel_type), h,
            Lambda=real_map(lam), Beta=real_map(beta),
            amplitude_mode="power_consistent", interference_amp=real_map(power.sqrt()),
        )
        return x, z, {"power": power, "sinr": sinr, "lambda": lam, "beta": beta}

    def SIC_sampling_alternating(self, SNR, received, h, channel_type, block_size, min_alpha=0.5, variant="A4"):
        """Blind alternating calibration with point or Gaussian-surrogate moments.

        A3 energy estimates initialize the sampler.  Once the reverse process
        reaches ``min_alpha``, each corrector uses its current denoised x/z
        point estimates for a nonnegative least-squares amplitude update.  No
        clean latent, true SINR or true block power is available. A4 uses only
        point estimates; A5 uses declared surrogate (not oracle) covariances.
        """
        if channel_type != "awgn":
            raise ValueError("A4 point calibration currently supports AWGN only")
        if not isinstance(block_size, int) or block_size < 1:
            raise ValueError("block_size must be a positive integer")
        if not 0 < min_alpha <= 1:
            raise ValueError("min_alpha must be in (0, 1]")
        from Diffusion.calibration import (
            estimate_energy_power, expand_blocks, power_to_sinr,
            guidance_parameters, equalize,
        )
        initial_power = estimate_energy_power(received, h, SNR, block_size)
        if variant not in {"A4", "A4_R", "A5_DIAG", "A5_FULL"}:
            raise ValueError("Unknown alternating variant")
        start_amplitude = initial_power.sqrt()
        if variant != "A4":
            start_amplitude = start_amplitude.clamp(max=8.)
        starting_power = initial_power if variant == "A4" else start_amplitude.square()
        initial_sinr = power_to_sinr(starting_power, SNR)
        initial_lam, initial_beta = guidance_parameters(initial_sinr, channel_type)
        complex_received = received
        def real_map(values):
            mapped = expand_blocks(values, complex_received, block_size)
            return torch.cat((mapped, mapped), dim=2)
        state = {
            "block_size": block_size,
            "channel_type": channel_type,
            "min_alpha": min_alpha,
            "amplitude": start_amplitude,
            "anchor": start_amplitude.clone(),
            "variant": variant,
            "power": starting_power,
            "sinr": initial_sinr,
            "lambda": initial_lam,
            "beta": initial_beta,
            "updates": 0,
        }
        x, z = self.SIC_sampling(
            SNR, None, equalize(received, h, SNR, channel_type), h,
            Lambda=real_map(initial_lam), Beta=real_map(initial_beta),
            amplitude_mode="power_consistent",
            interference_amp=real_map(start_amplitude),
            point_calibration=state,
        )
        batch = received.shape[0]
        estimates = {
            "initial_power": initial_power,
            "power": state["power"],
            "sinr": state["sinr"],
            "lambda": state["lambda"],
            "beta": state["beta"],
            "update_count": initial_power.new_full((batch, 1), state["updates"]),
        }
        if "diagnostics" in state:
            estimates.update(state["diagnostics"])
        return x, z, estimates
