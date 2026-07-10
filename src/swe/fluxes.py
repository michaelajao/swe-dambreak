"""Riemann flux kernels: Rusanov, HLL, and HLLC, keyed by name in ``FLUXES``.

Kernels are written in interface-local coordinates: ``un`` is the velocity
normal to the interface, ``ut`` tangential. Returned fluxes are stacked on a
new axis -3 as (mass, normal momentum, tangential momentum); the solver maps
these back to (h, hu, hv) per direction. All kernels are pure, vectorized
over arbitrary interface tensor shapes, and autograd-safe (guarded
denominators so no NaN flows through untaken ``torch.where`` branches).
"""

from __future__ import annotations

import torch

from .state import G, H_EPS, THIN_FACTOR

_TINY = 1.0e-14


# --------------------------------------------------------------------------
# Shared pieces
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# Rusanov (local Lax-Friedrichs)
# --------------------------------------------------------------------------

def rusanov_flux(
    hL: torch.Tensor,
    unL: torch.Tensor,
    utL: torch.Tensor,
    hR: torch.Tensor,
    unR: torch.Tensor,
    utR: torch.Tensor,
    g: float = G,
    h_eps: float = H_EPS,
) -> torch.Tensor:
    """Rusanov flux: F = (F_L + F_R)/2 - a (U_R - U_L)/2 with
    a = max(|un_L| + a_L, |un_R| + a_R). The most dissipative of the three
    kernels; serves as the robustness baseline."""
    FL = physical_flux(hL, unL, utL, g)
    FR = physical_flux(hR, unR, utR, g)
    a = torch.maximum(
        unL.abs() + celerity(hL, g), unR.abs() + celerity(hR, g)
    ).unsqueeze(-3)
    UL = torch.stack([hL, hL * unL, hL * utL], dim=-3)
    UR = torch.stack([hR, hR * unR, hR * utR], dim=-3)
    F = 0.5 * (FL + FR) - 0.5 * a * (UR - UL)
    return zero_dry_dry(F, hL, hR, h_eps)


# --------------------------------------------------------------------------
# HLL (Harten-Lax-van Leer)
# --------------------------------------------------------------------------

def hll_flux(
    hL: torch.Tensor,
    unL: torch.Tensor,
    utL: torch.Tensor,
    hR: torch.Tensor,
    unR: torch.Tensor,
    utR: torch.Tensor,
    g: float = G,
    h_eps: float = H_EPS,
) -> torch.Tensor:
    """HLL flux with Toro's wave-speed estimates (Toro 2001, §10.3):
    F_hll = (S_R F_L - S_L F_R + S_L S_R (U_R - U_L)) / (S_R - S_L)
    selected against F_L (S_L >= 0) and F_R (S_R <= 0)."""
    FL = physical_flux(hL, unL, utL, g)
    FR = physical_flux(hR, unR, utR, g)
    UL = torch.stack([hL, hL * unL, hL * utL], dim=-3)
    UR = torch.stack([hR, hR * unR, hR * utR], dim=-3)

    SL, SR = wave_speeds(hL, unL, hR, unR, g, h_eps)
    SLc = SL.unsqueeze(-3)
    SRc = SR.unsqueeze(-3)

    F_mid = safe_div(SRc * FL - SLc * FR + SLc * SRc * (UR - UL), SRc - SLc)
    F = torch.where(SLc >= 0, FL, torch.where(SRc <= 0, FR, F_mid))
    return zero_dry_dry(F, hL, hR, h_eps)


# --------------------------------------------------------------------------
# HLLC (Toro 2001, §10.4-10.5)
# --------------------------------------------------------------------------

def hllc_flux(
    hL: torch.Tensor,
    unL: torch.Tensor,
    utL: torch.Tensor,
    hR: torch.Tensor,
    unR: torch.Tensor,
    utR: torch.Tensor,
    g: float = G,
    h_eps: float = H_EPS,
) -> torch.Tensor:
    """HLLC flux: restores the contact/shear wave dropped by HLL.

    The tangential momentum is upwinded across the middle wave S*, which is
    what keeps 2D dam-break shear fronts sharp. Fully vectorized over all
    interfaces; pure and differentiable (no in-place ops, no .item(), guarded
    denominators) — this kernel doubles as the FVM-informed PINN loss core.

    Middle-wave speed (Toro 2001, eq. 10.58):
        S* = (S_L h_R (u_R - S_R) - S_R h_L (u_L - S_L))
             / (h_R (u_R - S_R) - h_L (u_L - S_L))
    Star states U*_K = h_K (S_K - u_K)/(S_K - S*) [1, S*, v_K]^T and
    F*_K = F_K + S_K (U*_K - U_K); the flux is picked by the signs of
    S_L, S*, S_R.

    Thin-layer/dry interfaces (either side <= THIN_FACTOR * h_eps) fall back
    to the HLL flux: in under-resolved films the two-rarefaction depth
    estimate far exceeds both side depths, S* lands far from the physical
    contact, and the star momentum flux flips sign — which pumps momentum
    into near-empty cells and blows up the front. The contact restoration
    HLLC exists for only matters in resolved wet regions.
    """
    FL = physical_flux(hL, unL, utL, g)
    FR = physical_flux(hR, unR, utR, g)
    UL = torch.stack([hL, hL * unL, hL * utL], dim=-3)
    UR = torch.stack([hR, hR * unR, hR * utR], dim=-3)

    SL, SR = wave_speeds(hL, unL, hR, unR, g, h_eps)

    num = SL * hR * (unR - SR) - SR * hL * (unL - SL)
    den = hR * (unR - SR) - hL * (unL - SL)
    S_star = safe_div(num, den)
    # degenerate case (both sides nearly still/dry): fall back to mean velocity
    S_star = torch.where(den.abs() < 1e-12, 0.5 * (unL + unR), S_star)

    def star_flux(
        F: torch.Tensor, U: torch.Tensor,
        h: torch.Tensor, un: torch.Tensor, ut: torch.Tensor, S: torch.Tensor,
    ) -> torch.Tensor:
        coef = h * safe_div(S - un, S - S_star)
        U_star = torch.stack([coef, coef * S_star, coef * ut], dim=-3)
        return F + S.unsqueeze(-3) * (U_star - U)

    FsL = star_flux(FL, UL, hL, unL, utL, SL)
    FsR = star_flux(FR, UR, hR, unR, utR, SR)

    SLc = SL.unsqueeze(-3)
    SRc = SR.unsqueeze(-3)
    Ssc = S_star.unsqueeze(-3)
    F = torch.where(
        SLc >= 0,
        FL,
        torch.where(Ssc >= 0, FsL, torch.where(SRc > 0, FsR, FR)),
    )

    # HLL fallback wherever either side is a thin layer or dry (see docstring)
    F_hll = safe_div(SRc * FL - SLc * FR + SLc * SRc * (UR - UL), SRc - SLc)
    F_hll = torch.where(SLc >= 0, FL, torch.where(SRc <= 0, FR, F_hll))
    h_thin = THIN_FACTOR * h_eps
    thin = ((hL <= h_thin) | (hR <= h_thin)).unsqueeze(-3)
    F = torch.where(thin, F_hll, F)

    return zero_dry_dry(F, hL, hR, h_eps)


#: Flux kernels keyed by name for config-driven selection.
FLUXES = {
    "rusanov": rusanov_flux,
    "hll": hll_flux,
    "hllc": hllc_flux,
}
