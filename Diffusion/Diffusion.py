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

    def SIC_sampling(self, SNR, SINR, deg_feature, h, Lambda=1, Beta=1, return_intermediate=False):

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
        )
        return x_0, z_0


