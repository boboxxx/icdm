"""CDDM Algorithm 2 receiver-only port; NOT the complete three-stage system.

Uses the existing DiT prior and its exact discrete training schedule, a frozen
shared decoder, and 40 respaced deterministic updates. Original CDDM uses a
separately trained U-Net and subsequently adapts its decoder. These differences
must remain visible in every experiment label and publication claim.
"""
import torch


@torch.no_grad()
def cddm_receive(model, received_real, complex_noise_power, alphas_cumprod, steps=40):
    if complex_noise_power <= 0 or steps < 1:
        raise ValueError("Positive noise variance and step budget required")
    # Stored complex symbols have unit power; each real component has var=1/2.
    # Forward diffusion uses unit-variance real epsilon, so sigma_r^2=Nc/2.
    abar = torch.as_tensor(alphas_cumprod, device=received_real.device, dtype=received_real.dtype)
    real_variance = complex_noise_power / 2
    start = int(((1 - abar) / abar - real_variance).abs().argmin())
    indices = torch.linspace(start, 0, min(steps, start + 1), device=received_real.device).round().long().unique(sorted=True).flip(0)
    # Match y=x+n to sqrt(abar)*x+sqrt(1-abar)*epsilon.
    state = abar[start].sqrt() * received_real
    for count, index in enumerate(indices):
        t = torch.full((state.shape[0],), int(index), device=state.device, dtype=torch.long)
        epsilon = model(state, t)
        clean = (state - (1 - abar[index]).sqrt() * epsilon) / abar[index].sqrt()
        if count + 1 < len(indices):
            nxt = indices[count + 1]
            state = abar[nxt].sqrt() * clean + (1 - abar[nxt]).sqrt() * epsilon
    return clean, {"nfe": len(indices), "start_index": start,
                   "assumed_complex_noise_power": float(complex_noise_power),
                   "matched_real_noise_variance": float((1 - abar[start]) / abar[start])}
