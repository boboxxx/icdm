"""Explicit, per-image RGB metrics. No training-loss wrappers or silent resize."""
import torch
import torch.nn.functional as F


def ms_ssim_rgb(x, y, levels=4):
    if x.shape != y.shape or x.ndim != 4 or x.shape[1] != 3:
        raise ValueError("Expected equal NCHW RGB tensors")
    if levels not in (4, 5) or min(x.shape[-2:]) < 11 * 2 ** (levels - 1):
        raise ValueError("Images too small for requested scales and 11-pixel window")
    weights = x.new_tensor([.0448, .2856, .3001, .2363, .1333])[:levels]
    if levels == 4:
        weights = weights / weights.sum()
    pos = torch.arange(11, device=x.device, dtype=x.dtype) - 5
    kernel = torch.exp(-pos.square() / (2 * 1.5 ** 2))
    kernel = (kernel / kernel.sum()).view(1, 1, 1, 11).repeat(3, 1, 1, 1)

    def smooth(v):
        v = F.conv2d(v, kernel, groups=3)
        return F.conv2d(v, kernel.transpose(2, 3), groups=3)

    values = []
    for scale in range(levels):
        mx, my = smooth(x), smooth(y)
        vx, vy = smooth(x * x) - mx * mx, smooth(y * y) - my * my
        cov = smooth(x * y) - mx * my
        cs = (2 * cov + .03 ** 2) / (vx + vy + .03 ** 2)
        ss = (2 * mx * my + .01 ** 2) / (mx * mx + my * my + .01 ** 2) * cs
        # Reduce spatially, multiply scales ONCE, then average RGB channels.
        values.append((ss if scale == levels - 1 else cs).mean((-2, -1)).relu())
        if scale < levels - 1:
            padding = (x.shape[-2] % 2, x.shape[-1] % 2)
            x = F.avg_pool2d(x, 2, padding=padding)
            y = F.avg_pool2d(y, 2, padding=padding)
    return torch.prod(torch.stack(values).clamp_min(1e-12) ** weights[:, None, None], dim=0).mean(1)


class ImageMetrics:
    def __init__(self, device):
        import lpips
        self.lpips = lpips.LPIPS(net="vgg", version="0.1", verbose=False).to(device).eval().requires_grad_(False)

    @torch.no_grad()
    def __call__(self, reconstruction, reference):
        if not torch.isfinite(reconstruction).all() or not torch.isfinite(reference).all():
            raise FloatingPointError("Nonfinite metric input")
        x, y = reconstruction.float().clamp(0, 1), reference.float().clamp(0, 1)
        mse = (x - y).square().flatten(1).mean(1)
        structural = ms_ssim_rgb(x, y)
        perceptual = self.lpips(2 * x - 1, 2 * y - 1).flatten(1).mean(1)
        return dict(mse=mse, psnr=-10 * mse.clamp_min(1e-12).log10(),
                    lpips_vgg=perceptual, ms_ssim=structural,
                    ms_ssim_db=-10 * (1 - structural).clamp_min(1e-12).log10())


METRIC_PROTOCOL = {
    "input": "float32 RGB [0,1]; identical clipping; no resizing/quantization",
    "mse": "per-image mean over C,H,W, then arithmetic mean over images",
    "psnr": "per-image -10 log10(MSE), then arithmetic mean; floor=1e-12",
    "lpips": "official lpips 0.1 VGG; inputs explicitly transformed to [-1,1]",
    "ms_ssim": "4 scales for 128x128, Gaussian 11 sigma1.5, valid convolution, K=.01,.03; normalized first four Wang weights; product once then RGB mean",
    "ms_ssim_db": "per-image -10 log10(1-MS-SSIM); floor=1e-12",
    "lpips_db_optional": "10 log10(mean LPIPS), lower is better (ICDM convention)",
    "warning": "MSE and PSNR share pixel-error information; not independent wins. Four-scale MS-SSIM is not directly comparable with five-scale scores.",
}
