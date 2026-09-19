"""AWGN joint Gaussian surrogate moments; not the exact neural posterior."""
import torch
from Diffusion.calibration import to_complex, expand_blocks


def gaussian_joint_moments(y, dx, dz, amplitude, vx, vz, noise_variance):
    """Real-coordinate diagonal-prior Gaussian conditioning on y=x+a*z+n.

    Covariances are per real coordinate. Inputs broadcast to y's shape.
    A shared observation induces nonzero cross covariance.
    """
    if noise_variance <= 0:
        raise ValueError("Noise variance must be positive")
    if torch.as_tensor(vx).min() < 0 or torch.as_tensor(vz).min() < 0:
        raise ValueError("Surrogate variances must be nonnegative")
    innovation_variance = vx + amplitude.square() * vz + noise_variance
    residual = y - dx - amplitude * dz
    mx = dx + vx / innovation_variance * residual
    mz = dz + amplitude * vz / innovation_variance * residual
    cxx = vx * (amplitude.square() * vz + noise_variance) / innovation_variance
    czz = vz * (vx + noise_variance) / innovation_variance
    cxz = -amplitude * vx * vz / innovation_variance
    return mx, mz, cxx, czz, cxz


def real_block_sums(values, block_size):
    """Sum paired real/imag entries within the complex-symbol block layout."""
    real, imag = values.chunk(2, dim=2)
    combined = (real + imag).flatten(1)
    return torch.stack([part.sum(1) for part in combined.split(block_size, 1)], 1)


def quadratic_statistics(y, mx, mz, czz, cxz, block_size):
    numerator = real_block_sums(mz * (y - mx) - cxz, block_size)
    denominator = real_block_sums(mz.square() + czz, block_size)
    return numerator, denominator


def bounded_moment_update(y, dx, dz, previous, anchor, block_size, alpha,
                          sigma, snr, variant, ridge=.05, damping=.25, cap=8.):
    """Equal stabilization for A4_R/A5_DIAG/A5_FULL; returns diagnostics."""
    if variant not in {"A4_R", "A5_DIAG", "A5_FULL"}:
        raise ValueError("Unknown moment calibration variant")
    if not (0 < damping <= 1 and ridge > 0 and cap > 0):
        raise ValueError("Require positive ridge/cap and damping in (0,1]")
    complex_y = to_complex(y)
    amap = expand_blocks(previous, complex_y, block_size)
    amap = torch.cat((amap, amap), dim=2)
    prior_variance = .5 * sigma.square() / (.5 * alpha.square() + sigma.square())
    if variant == "A4_R":
        mx, mz = dx, dz
        czz, cxz = torch.zeros_like(y), torch.zeros_like(y)
    else:
        mx, mz, _, czz, cxz = gaussian_joint_moments(
            y, dx, dz, amap, prior_variance, prior_variance, .5*10**(-snr/10)
        )
        if variant == "A5_DIAG":
            cxz = torch.zeros_like(cxz)
    numerator, denominator = quadratic_statistics(y, mx, mz, czz, cxz, block_size)
    real_counts = real_block_sums(torch.ones_like(y), block_size)
    penalty = ridge * real_counts
    candidate = ((numerator + penalty*anchor)/(denominator + penalty)).clamp(0, cap)
    updated = (1-damping)*previous + damping*candidate
    # Within the declared fixed-q quadratic this must be nonpositive, up to roundoff.
    n = numerator + penalty*anchor
    d = denominator + penalty
    objective_change = d*(updated.square()-previous.square()) - 2*n*(updated-previous)
    diagnostics = {
        "denominator": denominator,
        "numerator": numerator,
        "cross_trace": real_block_sums(cxz, block_size),
        "variance_trace": real_block_sums(czz, block_size),
        "candidate_at_cap": (candidate >= cap).to(y.dtype),
        "candidate_at_floor": (candidate <= 0).to(y.dtype),
        "objective_change": objective_change,
    }
    updated_map = expand_blocks(updated, complex_y, block_size)
    updated_map = torch.cat((updated_map, updated_map), dim=2)
    diagnostics["mean_residual_energy"] = real_block_sums(
        (y-mx-updated_map*mz).square(), block_size
    ) / real_counts
    if not torch.isfinite(updated).all():
        raise FloatingPointError("Nonfinite moment amplitude update")
    return updated, diagnostics
