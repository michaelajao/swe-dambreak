"""Shared pieces for interface flux kernels.

Kernels are written in interface-local coordinates: ``un`` is the velocity
normal to the interface, ``ut`` tangential. Returned fluxes are stacked on a
new axis -3 as (mass, normal momentum, tangential momentum); the solver maps
these back to (h, hu, hv) per direction. All kernels are pure, vectorized
over arbitrary interface tensor shapes, and autograd-safe (guarded
denominators so no NaN flows through untaken ``torch.where`` branches).
"""

from __future__ import annotations

import torch

from ..constants import G, H_EPS

_TINY = 1.0e-14


def physical_flux(
    h: torch.Tensor, un: torch.Tensor, ut: torch.Tensor, g: float = G
) -> torch.Tensor:
    """Exact SWE flux normal to the interface: (h un, h un^2 + g h^2/2, h un ut)."""
    hun = h * un
    return torch.stack([hun, hun * un + 0.5 * g * h * h, hun * ut], dim=-3)


def celerity(h: torch.Tensor, g: float = G) -> torch.Tensor:
    """sqrt(g h) with a derivative-safe floor.

    d(sqrt)/dh is infinite at h = 0, which turns into NaN gradients at dry
    interfaces via 0 * inf products in the backward pass even when the wet/dry
    ``torch.where`` masks zero the forward value. Clamping h at a subnormal
    floor makes the gradient exactly 0 at h = 0 and leaves every physical
    depth untouched.
    """
    return torch.sqrt(g * torch.clamp(h, min=1e-300))


def safe_div(num: torch.Tensor, den: torch.Tensor, tiny: float = _TINY) -> torch.Tensor:
    """num / den with |den| floored at ``tiny`` (sign preserved; sign(0) -> +)."""
    sgn = torch.where(den >= 0, torch.ones_like(den), -torch.ones_like(den))
    den_safe = sgn * torch.clamp(den.abs(), min=tiny)
    return num / den_safe


def wave_speeds(
    hL: torch.Tensor,
    unL: torch.Tensor,
    hR: torch.Tensor,
    unR: torch.Tensor,
    g: float = G,
    h_eps: float = H_EPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Left/right wave-speed estimates SL, SR with dry-bed handling.

    Davies-type direct bounds
        SL = min(uL - aL, uR - aR),  SR = max(uL + aL, uR + aR),
    plus the exact dry-front speeds when one side is dry (h <= h_eps):
        right state dry: SL = uL - aL,  SR = uL + 2 aL
        left  state dry: SL = uR - 2 aR, SR = uR + aR.

    Toro's two-rarefaction estimate with the q_K shock correction was tried
    first and abandoned: in thin films at wet/dry fronts q_K amplifies the
    interface speeds far beyond every *cell* speed (|S| ~ 100x |u|+a), so the
    CFL time step chosen from cell speeds is locally violated and the front
    blows up. The Davies bounds are bounded by the cell speeds by
    construction, hence consistent with the CFL control — and match the
    estimate used by the reference paper's HLL, which helps reconciliation.
    """
    aL = celerity(hL, g)
    aR = celerity(hR, g)

    SL = torch.minimum(unL - aL, unR - aR)
    SR = torch.maximum(unL + aL, unR + aR)

    dryL = hL <= h_eps
    dryR = hR <= h_eps
    SL = torch.where(dryR, unL - aL, SL)
    SR = torch.where(dryR, unL + 2.0 * aL, SR)
    SL = torch.where(dryL, unR - 2.0 * aR, SL)
    SR = torch.where(dryL, unR + aR, SR)
    return SL, SR


def zero_dry_dry(
    F: torch.Tensor, hL: torch.Tensor, hR: torch.Tensor, h_eps: float = H_EPS
) -> torch.Tensor:
    """Force zero flux across interfaces where both sides are dry."""
    wet_any = (hL > h_eps) | (hR > h_eps)
    return F * wet_any.unsqueeze(-3)
