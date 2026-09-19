"""Receiver-only power estimation; powers refer to one complex symbol.

The latent layout is [batch, channel, stacked real/imag height, width].
Blocks follow flattened [channel, complex height, width] order per frame.
No clean test latent, interference power or true SINR enters the estimator.
"""
import math
import torch


def interference_amplitude(snr, sinr, mode="original"):
    if not math.isfinite(snr) or not math.isfinite(sinr) or sinr > snr:
        raise ValueError("Require finite SINR <= SNR")
    noise = 10 ** (-snr / 10)
    total = 10 ** (-sinr / 10)
    if mode == "original":
        return math.sqrt(total) - noise
    if mode == "power_consistent":
        return math.sqrt(max(total - noise, 0.0))
    raise ValueError("Unknown amplitude mode: " + mode)


def to_complex(x):
    if x.ndim != 4 or x.shape[2] % 2 or x.is_complex():
        raise ValueError("Expected real BCHW with even stacked real/imag height")
    real, imag = x.chunk(2, dim=2)
    return torch.complex(real, imag)


def to_stacked(x):
    if not x.is_complex() or x.ndim != 4:
        raise ValueError("Expected complex BCHW")
    return torch.cat((x.real, x.imag), dim=2)


def normalize_unit_complex_power(x):
    """Remove the x/z scale ambiguity using the transmitter constraint."""
    complex_x = to_complex(x)
    dims = tuple(range(1, complex_x.ndim))
    power = complex_x.abs().square().mean(dim=dims, keepdim=True)
    floor = torch.finfo(power.dtype).eps
    return x / power.clamp_min(floor).sqrt()


def expand_blocks(values, like, block_size):
    """Expand [batch, blocks] to the complex-symbol shape of like."""
    count = like[0].numel()
    if not isinstance(block_size, int) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    expected = (like.shape[0], (count + block_size - 1) // block_size)
    if tuple(values.shape) != expected:
        raise ValueError("Block power shape must be %s" % (expected,))
    return values.repeat_interleave(block_size, dim=1)[:, :count].reshape(like.shape)


def estimate_energy_power(received, h, snr, block_size=None):
    """Estimate E|az|² from RAW y, assuming E|x|²=E|z|²=1.

    Subtract E|hx|²=|h|² and E|w|²=10**(-SNR/10), then clip each
    block mean (not individual symbols). Short blocks can be very noisy.
    Returns a nonnegative [batch, blocks] tensor, independently per frame.
    """
    if received.ndim != 4 or not received.is_complex():
        raise ValueError("Use raw complex received signal, before equalization")
    if h.shape != received.shape or not math.isfinite(snr):
        raise ValueError("CSI must match received signal; SNR must be finite")
    if not torch.isfinite(received).all() or not torch.isfinite(h).all():
        raise ValueError("Received signal and CSI must be finite")
    count = received[0].numel()
    block_size = count if block_size is None else block_size
    if not isinstance(block_size, int) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    excess = (received.abs().square() - h.abs().square() - 10 ** (-snr / 10)).flatten(1)
    return torch.stack([part.mean(1).clamp_min(0) for part in excess.split(block_size, 1)], 1)


def init_block_power(received, h, snr, block_size):
    return estimate_energy_power(received, h, snr, block_size)


def estimate_point_amplitude(received, signal, interference, block_size, fallback=None):
    """Nonnegative blockwise least-squares amplitude for A4.

    Solves ``min_{a_b >= 0} ||y_b - x_b - a_b z_b||^2`` from receiver-side
    point estimates.  All three tensors use the stacked real/imag layout.  A
    fallback is used only for a numerically empty interference estimate; it is
    normally the amplitude from the preceding alternating update.
    """
    y, x, z = map(to_complex, (received, signal, interference))
    if y.shape != x.shape or y.shape != z.shape:
        raise ValueError("Received, signal and interference estimates must agree")
    if not all(torch.isfinite(value).all() for value in (y, x, z)):
        raise ValueError("A4 point estimates must be finite")
    count = y[0].numel()
    if not isinstance(block_size, int) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    number = (count + block_size - 1) // block_size
    if fallback is not None and tuple(fallback.shape) != (y.shape[0], number):
        raise ValueError("Fallback amplitude has the wrong block shape")
    residual = (y - x).flatten(1)
    z = z.flatten(1)
    amplitudes = []
    for block, (residual_part, z_part) in enumerate(
        zip(residual.split(block_size, 1), z.split(block_size, 1))
    ):
        numerator = (z_part.conj() * residual_part).real.sum(1)
        denominator = z_part.abs().square().sum(1)
        threshold = torch.finfo(denominator.dtype).eps * z_part.shape[1]
        estimate = (numerator / denominator.clamp_min(threshold)).clamp_min(0)
        if fallback is not None:
            estimate = torch.where(denominator > threshold, estimate, fallback[:, block])
        amplitudes.append(estimate)
    return torch.stack(amplitudes, 1)


def power_to_sinr(power, snr):
    return -10 * torch.log10(power + 10 ** (-snr / 10))


def guidance_parameters(sinr, channel_type):
    """Linear interpolation, clipped to published table endpoints.

    Used only with estimated SINR for A2/A3. No test-set tuning.
    These published values are labeled C8 in Train.py, not universal optima.
    """
    if channel_type == "awgn":
        knots = [-7, -4, 0, 4, 7, 10, 20]
        lam, beta = [.3, 1, 1.4, 3, 4.5, 5.5, 3], [.1, .7, 1, 1, 1, .5, 2]
    elif channel_type == "rayleigh":
        knots = [-7, -4, 0, 4, 7, 10]
        lam, beta = [.3, .5, 1, 1.8, 2.2, 2.5], [.1, .1, .3, 1, 1, 1]
    else:
        raise ValueError("Expected awgn or rayleigh")
    grid = sinr.new_tensor(knots)
    clipped = sinr.clamp(grid[0], grid[-1])
    right = torch.searchsorted(grid, clipped.contiguous()).clamp(1, len(knots) - 1)
    left = right - 1
    fraction = (clipped - grid[left]) / (grid[right] - grid[left])
    def interp(values):
        values = sinr.new_tensor(values)
        return values[left] + fraction * (values[right] - values[left])
    return interp(lam), interp(beta)


def equalize(received, h, snr, channel_type):
    """Keep upstream equalizer (including 2*N0) for isolated A0–A3 ablation."""
    if channel_type == "awgn":
        return to_stacked(received)
    if channel_type == "rayleigh":
        return to_stacked(received * h.conj() / (h.abs().square() + 2 * 10 ** (-snr / 10)))
    raise ValueError("Expected awgn or rayleigh")
