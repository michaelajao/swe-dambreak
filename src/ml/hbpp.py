# Reused from Michael Ajao-Olarinoye and M. Asif Farooq's hbpp-euler-cartesian.
# MIT license retained in hbpp_license.txt. Architecture precedent:
# Safari et al., HBPP, Neurocomputing 702 (2026) 134601.
"""Reusable HBPP architecture copied from the author's Euler project.

The SWE pilot supplies its own three-coordinate input, conserved-state output
transform and finite-volume physics in fv_pilot.py. Euler field transforms,
residual interpolation and geometry sensors are not part of this adaptation.
The original optional regularizers remain available but are not used by the pilot.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .hbpp_layers import MLP, FourierFeatures, LearnableBasis


class HBPPOp(nn.Module):
    def __init__(
        self,
        branch_dim: int,
        coord_dim: int = 2,
        out_dim: int = 4,
        d: int = 128,
        n_basis: int = 32,
        branch_hidden: list[int] | None = None,
        trunk_hidden: list[int] | None = None,
        scaling_hidden: list[int] | None = None,
        value_hidden: list[int] | None = None,
        basis_hidden: list[int] | None = None,
        basis_kind: str = "mlp",
        n_fourier: int = 64,
        fourier_scale: float = 2.0,
        use_fourier: bool = True,
        use_scaling: bool = True,
        use_gate: bool = True,
        use_basis: bool = True,
        scale_axes: list[int] | None = None,
        warp_mode: str = "pointwise",
        mu_bound: float = 0.5,
        sigma_range: tuple[float, float] = (0.5, 1.5),
        value_skip: bool = True,
        activation: str = "gelu",
        seed: int = 0,
    ):
        super().__init__()
        self.d = d
        self.coord_dim = coord_dim
        self.use_fourier, self.use_scaling, self.use_gate, self.use_basis = use_fourier, use_scaling, use_gate, use_basis
        self.value_skip = value_skip
        self.scale_axes = list(scale_axes) if scale_axes is not None else list(range(min(2, coord_dim)))
        if warp_mode not in ("pointwise", "global"):
            raise ValueError(f"warp_mode must be 'pointwise' or 'global', got {warp_mode!r}")
        self.warp_mode = warp_mode
        self.mu_bound = mu_bound
        self.sig_lo, self.sig_hi = sigma_range

        self.branch = MLP(branch_dim, branch_hidden or [256, 256], d, activation)
        if use_fourier:
            self.ff = FourierFeatures(coord_dim, n_fourier, fourier_scale, seed)
            t_in = self.ff.out_dim
        else:
            self.ff = None
            t_in = coord_dim
        self.trunk = MLP(t_in, trunk_hidden or [128, 128], d, activation)
        na = len(self.scale_axes)
        # zero-initialised last layer: the warp starts as the identity (mu = 0, sigma = mid-range)
        s_in = 2 * d if warp_mode == "pointwise" else d
        self.scaling = MLP(s_in, scaling_hidden or [128, 64], 2 * na, activation, zero_last=True) if use_scaling else None
        self.basis = LearnableBasis(n_basis, coord_dim, basis_hidden, basis_kind, seed=seed) if use_basis else None
        v_in = d + (n_basis if use_basis else 0) + d
        self.value = MLP(v_in, value_hidden or [128, 128], d, activation)
        self.out = nn.Linear(d, out_dim)

    # ------------------------------------------------------------------
    def warp(self, K: torch.Tensor, Q: torch.Tensor, y: torch.Tensor):
        """y' = (y - mu)/sigma on the scaled axes; returns (y', mu, sigma)."""
        if self.scaling is None:
            return y, None, None
        if self.warp_mode == "pointwise":
            raw = self.scaling(torch.cat([K.expand_as(Q), Q], -1))
        else:                                   # one affine map per sample, broadcast over points
            raw = self.scaling(K).expand(*Q.shape[:-1], -1)
        na = len(self.scale_axes)
        mu = self.mu_bound * torch.tanh(raw[..., :na])
        sigma = self.sig_lo + (self.sig_hi - self.sig_lo) * torch.sigmoid(raw[..., na:])
        ax = self.scale_axes
        ys = (y[..., ax] - mu) / sigma
        parts = []
        k = 0
        for j in range(y.shape[-1]):
            if j in ax:
                parts.append(ys[..., k:k + 1])
                k += 1
            else:
                parts.append(y[..., j:j + 1])
        return torch.cat(parts, -1), mu, sigma

    def forward(self, branch: torch.Tensor, y: torch.Tensor, return_aux: bool = False):
        """branch (B, m), y (B, N, coord_dim) -> (B, N, out_dim)."""
        K = self.branch(branch).unsqueeze(1)                              # (B, 1, d)
        Q = self.trunk(self.ff(y) if self.ff is not None else y)          # (B, N, d)
        y2, mu, sigma = self.warp(K, Q, y)
        feats = [Q]
        if self.basis is not None:
            feats.append(self.basis(y2))
        feats.append(K.expand(-1, Q.shape[1], -1))
        V = self.value(torch.cat(feats, -1))
        if self.value_skip:
            V = V + Q
        if self.use_gate:
            alpha = torch.sigmoid((Q * K).sum(-1, keepdim=True) / math.sqrt(self.d))
        else:
            alpha = torch.ones_like(Q[..., :1])
        out = self.out(alpha * V)
        if return_aux:
            return out, {"mu": mu, "sigma": sigma, "alpha": alpha, "y_warped": y2}
        return out

    # ------------------------------------------------------------------
    def orthogonality_loss(self, n_points: int = 256, generator: torch.Generator | None = None) -> torch.Tensor:
        """mean((B B^T - I)^2) over row-normalised basis evaluations on random points."""
        if self.basis is None:
            return torch.zeros((), device=self.out.weight.device)
        dev = self.out.weight.device
        p = torch.rand(n_points, self.coord_dim, generator=generator).to(dev) * 2 - 1
        Bm = self.basis(p).T                                            # (n_basis, P)
        act = self.basis.active > 0
        Bm = Bm[act]
        Bn = Bm / Bm.norm(dim=1, keepdim=True).clamp(min=1e-8)
        G = Bn @ Bn.T
        return ((G - torch.eye(G.shape[0], device=dev)) ** 2).mean()

    def folding_penalty(self, branch: torch.Tensor, y: torch.Tensor, margin: float = 0.1) -> torch.Tensor:
        """Hinge on the no-folding condition of HBPP Lemma A.9, per scaled axis."""
        if self.scaling is None:
            return torch.zeros((), device=y.device)
        y = y.detach().requires_grad_(True)
        K = self.branch(branch).unsqueeze(1)
        Q = self.trunk(self.ff(y) if self.ff is not None else y)
        _, mu, sigma = self.warp(K, Q, y)
        pen = torch.zeros((), device=y.device)
        for k, ax in enumerate(self.scale_axes):
            if self.warp_mode == "global":      # mu, sigma constant in y: the map cannot fold
                continue
            dmu = torch.autograd.grad(mu[..., k].sum(), y, create_graph=True)[0][..., ax]
            dsg = torch.autograd.grad(sigma[..., k].sum(), y, create_graph=True)[0][..., ax]
            lhs = dmu.abs() + (y[..., ax] - mu[..., k]).abs() / sigma[..., k] * dsg.abs()
            pen = pen + torch.relu(lhs - (1.0 - margin)).pow(2).mean()
        return pen

    def set_active_basis(self, k: int):
        if self.basis is not None:
            self.basis.set_active(k)
