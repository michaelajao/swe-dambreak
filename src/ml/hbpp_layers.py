# Reused from Michael Ajao-Olarinoye and M. Asif Farooq's hbpp-euler-cartesian.
# MIT license retained in hbpp_license.txt. Architecture precedent:
# Safari et al., HBPP, Neurocomputing 702 (2026) 134601.
"""Shared building blocks: MLPs, random Fourier features, and a vectorised
ensemble of small MLPs (the learnable basis, evaluated in one batched pass as
the HBPP paper recommends for higher dimensions)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

ACTIVATIONS = {"tanh": nn.Tanh, "gelu": nn.GELU, "silu": nn.SiLU, "relu": nn.ReLU}


class MLP(nn.Module):
    """in -> hidden... -> out, activation between layers, none after the last."""

    def __init__(self, in_dim: int, hidden: list[int], out_dim: int, activation: str = "gelu",
                 zero_last: bool = False):
        super().__init__()
        dims = [in_dim, *hidden, out_dim]
        layers: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), ACTIVATIONS[activation]()]
        self.net = nn.Sequential(*layers[:-1])
        if zero_last:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


class FourierFeatures(nn.Module):
    """gamma(y) = [cos(2 pi y B), sin(2 pi y B)], B ~ N(0, scale^2) fixed (HBPP Def. 2.1)."""

    def __init__(self, in_dim: int, n_features: int, scale: float, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("B", torch.randn(in_dim, n_features, generator=g) * scale)
        self.out_dim = 2 * n_features

    def forward(self, y):
        proj = 2 * math.pi * y @ self.B
        return torch.cat([torch.cos(proj), torch.sin(proj)], -1)


class EnsembleMLP(nn.Module):
    """``n`` independent MLPs R^in -> R^out evaluated at the same inputs.

    x (..., in) -> (..., n, out). Weights are stacked (n, a, b) and applied with
    einsum, so all basis functions cost one batched matmul per layer.
    """

    def __init__(self, n: int, in_dim: int, hidden: list[int], out_dim: int = 1, activation: str = "tanh"):
        super().__init__()
        dims = [in_dim, *hidden, out_dim]
        self.W = nn.ParameterList()
        self.b = nn.ParameterList()
        for a, c in zip(dims[:-1], dims[1:]):
            w = torch.empty(n, a, c)
            for k in range(n):
                nn.init.xavier_normal_(w[k])
            self.W.append(nn.Parameter(w))
            self.b.append(nn.Parameter(torch.zeros(n, c)))
        self.act = ACTIVATIONS[activation]()
        self.n = n

    def forward(self, x):
        h = torch.einsum("...a,nab->...nb", x, self.W[0]) + self.b[0]
        for W, b in zip(self.W[1:], self.b[1:]):
            h = torch.einsum("...na,nab->...nb", self.act(h), W) + b
        return h


class LearnableBasis(nn.Module):
    """N learnable basis functions phi_i : R^d -> R (HBPP Variant B).

    ``kind='mlp'``  : phi_i is a small tanh MLP (paper Exp. 1).
    ``kind='sine'`` : phi_i(y) = a_i(y) sin(w_i . y + theta_i(y)) with a fixed,
                      distinct frequency w_i and amplitude/phase from a small MLP
                      (paper Exp. 2, the oscillatory inductive bias).
    """

    def __init__(self, n: int, in_dim: int, hidden: list[int] | None = None, kind: str = "mlp",
                 max_freq: float = 4.0, seed: int = 0):
        super().__init__()
        hidden = hidden or [32, 32]
        self.kind = kind
        self.n = n
        if kind == "mlp":
            self.net = EnsembleMLP(n, in_dim, hidden, 1)
        elif kind == "sine":
            self.net = EnsembleMLP(n, in_dim, hidden, 2)
            g = torch.Generator().manual_seed(seed)
            dirs = torch.randn(n, in_dim, generator=g)
            dirs = dirs / dirs.norm(dim=1, keepdim=True)
            mags = torch.linspace(0.5, max_freq, n)
            self.register_buffer("omega", math.pi * dirs * mags[:, None])
        else:
            raise ValueError(f"basis kind must be 'mlp' or 'sine', got {kind!r}")
        self.register_buffer("active", torch.ones(n))

    def forward(self, y):
        h = self.net(y)                                            # (..., n, 1|2)
        if self.kind == "mlp":
            out = h[..., 0]
        else:
            phase = torch.einsum("...d,nd->...n", y, self.omega)
            out = h[..., 0] * torch.sin(phase + h[..., 1])
        return out * self.active

    def set_active(self, k: int):
        self.active.zero_()
        self.active[: max(0, min(k, self.n))] = 1.0


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
