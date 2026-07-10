"""Loss assembly for the strong-form PINNs.

Autograd residuals of the 2D SWEs in both non-conservative (primitive) and
conservative form, so the two baseline PINNs differ only here. Bed-slope and
Manning-friction source terms are included; for the flat frictionless
benchmarks (B1) they vanish. IC/BC/data misfits are plain pointwise MSE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from .models import PINN


@dataclass
class Physics:
    g: float = 9.81
    manning_n: float = 0.0
    # bed gradient at collocation points; returns (z_x, z_y). None -> flat bed.
    bed_grad: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None = None
    h_floor: float = 1.0e-3     # depth floor for friction/velocity guards


def _grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """dy/dx with the graph retained (for higher-order / multi-term use)."""
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y),
        create_graph=True, retain_graph=True,
    )[0]


def _friction_terms(
    h: torch.Tensor, u: torch.Tensor, v: torch.Tensor, phys: Physics
) -> tuple[torch.Tensor, torch.Tensor]:
    """Manning friction accelerations (per unit mass) fr_x, fr_y; 0 if n=0."""
    if phys.manning_n == 0.0:
        z = torch.zeros_like(h)
        return z, z
    hs = torch.clamp(h, min=phys.h_floor)
    speed = torch.sqrt(u * u + v * v)
    coef = phys.g * phys.manning_n**2 * speed / hs ** (4.0 / 3.0)
    return coef * u, coef * v


def pde_residual(model: PINN, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor,
                 phys: Physics) -> torch.Tensor:
    """(N, 3) SWE residual at collocation points, in the model's formulation.

    x, y, t are (N, 1) leaf tensors with requires_grad=True.
    """
    xyt = torch.cat([x, y, t], dim=1)
    zx, zy = (phys.bed_grad(x, y) if phys.bed_grad is not None
              else (torch.zeros_like(x), torch.zeros_like(y)))

    if model.cfg.variables == "primitive":
        h, u, v = (o.unsqueeze(-1) for o in model.forward(xyt))
        fr_x, fr_y = _friction_terms(h, u, v, phys)
        cont = _grad(h, t) + _grad(h * u, x) + _grad(h * v, y)
        mom_x = (_grad(u, t) + u * _grad(u, x) + v * _grad(u, y)
                 + phys.g * (_grad(h, x) + zx) + fr_x)
        mom_y = (_grad(v, t) + u * _grad(v, x) + v * _grad(v, y)
                 + phys.g * (_grad(h, y) + zy) + fr_y)
        return torch.cat([cont, mom_x, mom_y], dim=1)

    # conservative: network outputs (h, hu, hv)
    h, hu, hv = (o.unsqueeze(-1) for o in model.forward(xyt))
    hs = torch.clamp(h, min=phys.h_floor)
    u, v = hu / hs, hv / hs
    fr_x, fr_y = _friction_terms(h, u, v, phys)
    F0, G0 = hu, hv
    F1 = hu * u + 0.5 * phys.g * h * h
    F2 = hu * v
    G1 = hv * u
    G2 = hv * v + 0.5 * phys.g * h * h
    cont = _grad(h, t) + _grad(F0, x) + _grad(G0, y)
    mom_x = _grad(hu, t) + _grad(F1, x) + _grad(F2, y) + phys.g * h * zx + h * fr_x
    mom_y = _grad(hv, t) + _grad(G1, x) + _grad(G2, y) + phys.g * h * zy + h * fr_y
    return torch.cat([cont, mom_x, mom_y], dim=1)


def mse(a: torch.Tensor, b: torch.Tensor | float = 0.0) -> torch.Tensor:
    if isinstance(b, float):
        return (a * a).mean()
    return ((a - b) ** 2).mean()


def ic_loss(model: PINN, xyt0: torch.Tensor,
            h0: torch.Tensor, u0: torch.Tensor, v0: torch.Tensor) -> torch.Tensor:
    """MSE of the network state at t=0 against the initial condition.

    Targets are given in the model's own variables: (h, u, v) for primitive,
    (h, hu, hv) for conservative — the caller supplies matching targets.
    """
    a, b, c = model.forward(xyt0)
    return mse(a, h0) + mse(b, u0) + mse(c, v0)


def data_loss(model: PINN, xyt: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """MSE against sparse gauge data (targets in the model's variables, (M,3))."""
    a, b, c = model.forward(xyt)
    pred = torch.stack([a, b, c], dim=-1)
    return mse(pred, targets)
