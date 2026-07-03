"""FVM-informed PINN.

The network predicts cell-averaged states on the solver grid at collocation
times; the loss is the residual of the discrete SSP-RK2 update built from the
differentiable HLLC flux + Audusse hydrostatic reconstruction (swe.solver.step),
so discrete conservation and well-balancing are inherited from the classical
solver rather than learned. Design choices (concurrent with Liu, arXiv
2605.11001; our regime is shock-dominated 2D dam breaks with wet/dry fronts and
IC variation, using HLLC instead of Roe):

  (a) the network outputs the free-surface perturbation xi = h - h_s, so a lake
      at rest reduces to predicting xi = 0;
  (b) softplus on the recovered depth h = softplus(xi + h_s) hard-enforces h>=0;
  (c) a Fourier-feature embedding precedes the first hidden layer.

The network runs in float32; states are cast to float64 for the flux step.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from swe.solver import Config, step
from swe.grid import pad_scalar
from .pinn import FourierFeatures, PINNConfig, _ACT


@dataclass
class FVMPINNConfig:
    hidden: int = 128
    layers: int = 5
    activation: str = "tanh"
    fourier_features: int = 32
    fourier_scale: float = 1.0
    x_range: tuple[float, float] = (0.0, 1.0)
    y_range: tuple[float, float] = (0.0, 1.0)
    t_range: tuple[float, float] = (0.0, 1.0)


class FVMPINN(nn.Module):
    """(x, y, t) -> (xi, hu, hv); depth recovered as softplus(xi + h_s)."""

    def __init__(self, cfg: FVMPINNConfig, h_s: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("h_s", h_s)                     # (ny, nx) still depth
        lo = torch.tensor([cfg.x_range[0], cfg.y_range[0], cfg.t_range[0]])
        hi = torch.tensor([cfg.x_range[1], cfg.y_range[1], cfg.t_range[1]])
        self.register_buffer("in_lo", lo)
        self.register_buffer("in_hi", hi)

        if cfg.fourier_features > 0:
            self.embed = FourierFeatures(3, cfg.fourier_features, cfg.fourier_scale)
            in_dim = 2 * cfg.fourier_features
        else:
            self.embed = None
            in_dim = 3
        act = _ACT[cfg.activation]
        layers: list[nn.Module] = [nn.Linear(in_dim, cfg.hidden), act()]
        for _ in range(cfg.layers - 1):
            layers += [nn.Linear(cfg.hidden, cfg.hidden), act()]
        layers += [nn.Linear(cfg.hidden, 3)]
        self.net = nn.Sequential(*layers)

    def _forward_points(self, xyt: torch.Tensor) -> torch.Tensor:
        z = 2 * (xyt - self.in_lo) / (self.in_hi - self.in_lo) - 1
        if self.embed is not None:
            z = self.embed(z)
        return self.net(z)

    def predict_grid(self, t: float, grid) -> torch.Tensor:
        """Predicted conserved state U=(h,hu,hv) on all cell centers at time t.

        Returns float64 (3, ny, nx) — ready for the differentiable flux step.
        """
        X, Y = grid.centers()
        xyt = torch.stack([
            X.reshape(-1).to(self.in_lo),
            Y.reshape(-1).to(self.in_lo),
            torch.full((grid.ny * grid.nx,), float(t), dtype=self.in_lo.dtype,
                       device=self.in_lo.device),
        ], dim=1)
        out = self._forward_points(xyt).reshape(grid.ny, grid.nx, 3)
        xi = out[..., 0]
        h = torch.nn.functional.softplus(xi + self.h_s)
        hu = out[..., 1]
        hv = out[..., 2]
        U = torch.stack([h, hu, hv], dim=0)
        return U.to(torch.float64)


@dataclass
class FVMResidualSpec:
    cfg: Config                       # solver config (scheme=hllc, order, g, n)
    times: list[float]                # collocation times (increasing)
    n_sub: int = 1                    # CFL-safe substeps per collocation interval
    z: torch.Tensor | None = None     # bed (ny, nx); None -> flat
    stochastic: bool = False          # sample one interval per call (SGD-style)
    step_dtype: torch.dtype = torch.float64   # precision of the FV step in the loss


def _interval_residual(model, spec, z_pad, k):
    grid = spec.cfg.grid
    t0, t1 = spec.times[k], spec.times[k + 1]
    dt = (t1 - t0) / spec.n_sub
    U = model.predict_grid(t0, grid).to(spec.step_dtype)
    for _ in range(spec.n_sub):
        U = step(U, z_pad, dt, spec.cfg)
    U_next = model.predict_grid(t1, grid).to(spec.step_dtype)
    diff = U_next - U
    return (diff**2).mean(), (diff**2).mean(dim=(-2, -1)).detach()


def fvm_residual_loss(
    model: FVMPINN, spec: FVMResidualSpec
) -> tuple[torch.Tensor, dict]:
    """Mean-squared residual of the discrete FV update between consecutive
    collocation times: || U_pred(t_{k+1}) - FV_step^{n_sub}(U_pred(t_k)) ||^2.

    Both states come from the network, so the gradient trains the network to be
    consistent with the classical discrete operator. With ``stochastic=True`` a
    single random interval is used per call (mini-batch SGD over intervals) —
    this is the practical training mode, since backprop through the sequential
    float64 solver steps is the dominant cost.
    """
    grid = spec.cfg.grid
    z = spec.z if spec.z is not None else torch.zeros(
        grid.ny, grid.nx, dtype=spec.step_dtype, device=grid.device)
    z_pad = pad_scalar(z.to(spec.step_dtype), spec.cfg.bc, grid.ng)
    n = len(spec.times) - 1

    if spec.stochastic:
        k = int(torch.randint(0, n, (1,)))
        loss, pc = _interval_residual(model, spec, z_pad, k)
        return loss.to(model.h_s.dtype), {"per_channel_mse": pc.tolist(), "interval": k}

    total = model.h_s.new_zeros(())
    per_channel = torch.zeros(3, dtype=torch.float64, device=grid.device)
    for k in range(n):
        loss, pc = _interval_residual(model, spec, z_pad, k)
        per_channel += pc.double()
        total = total + loss.to(total.dtype)
    total = total / max(n, 1)
    return total, {"per_channel_mse": (per_channel / max(n, 1)).tolist()}


def ic_anchor_loss(model: FVMPINN, grid, U0: torch.Tensor) -> torch.Tensor:
    """MSE of the predicted grid state at t=0 against the initial condition."""
    U_pred = model.predict_grid(0.0, grid)
    return ((U_pred - U0.to(torch.float64)) ** 2).mean().to(model.h_s.dtype)


def data_anchor_loss(
    model: FVMPINN,
    xyt: torch.Tensor,
    h_s_pts: torch.Tensor,
    targets_U: torch.Tensor,
) -> torch.Tensor:
    """MSE at sparse gauge points/times against reference (h, hu, hv).

    xyt: (M, 3) gauge coordinates; h_s_pts: (M,) still-water depth at those
    points (so the softplus depth can be recovered off-grid); targets_U: (M, 3)
    conserved (h, hu, hv) from the reference run.
    """
    out = model._forward_points(xyt.to(model.in_lo))
    h = torch.nn.functional.softplus(out[..., 0] + h_s_pts.to(out))
    pred = torch.stack([h, out[..., 1], out[..., 2]], dim=-1)
    return ((pred - targets_U.to(out)) ** 2).mean()
