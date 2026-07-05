"""Tests for the ML entries: shapes, residual wiring, well-balancing, training."""

import torch

from ml.fvm_pinn import (
    FVMPINN,
    FVMPINNConfig,
    FVMResidualSpec,
    fvm_residual_loss,
    ic_anchor_loss,
)
from ml.losses import Physics, pde_residual
from ml.pinn import PINN, PINNConfig
from ml.train import TrainConfig, train
from swe.grid import TRANSMISSIVE, Grid
from swe.solver import Config
from swe.state import conserved


def _zero_last_layer(net, bias=None):
    last = net[-1]
    with torch.no_grad():
        last.weight.zero_()
        last.bias.zero_()
        if bias is not None:
            last.bias.copy_(torch.tensor(bias, dtype=last.bias.dtype))


def test_pinn_shapes_both_formulations():
    for variables in ("primitive", "conservative"):
        m = PINN(PINNConfig(variables=variables, hidden=16, layers=2))
        xyt = torch.rand(10, 3)
        a, b, c = m.forward(xyt)
        assert a.shape == (10,) and b.shape == (10,) and c.shape == (10,)
        h, u, v = m.state(xyt)
        assert h.shape == (10,)


def test_fourier_features_shape():
    m = PINN(PINNConfig(hidden=16, layers=2, fourier_features=8))
    assert m.forward(torch.rand(5, 3))[0].shape == (5,)


def test_strong_form_residual_zero_for_still_lake():
    """Constant still state (h=H, u=v=0, flat bed) has zero SWE residual in
    both formulations — validates the autograd residual wiring."""
    H = 2.0
    for variables in ("primitive", "conservative"):
        m = PINN(PINNConfig(variables=variables, hidden=16, layers=2))
        _zero_last_layer(m.net, bias=[H, 0.0, 0.0])
        x = torch.rand(20, 1, requires_grad=True)
        y = torch.rand(20, 1, requires_grad=True)
        t = torch.rand(20, 1, requires_grad=True)
        res = pde_residual(m, x, y, t, Physics(g=9.81))
        assert res.shape == (20, 3)
        assert res.abs().max() < 1e-5


def test_strong_form_residual_has_param_gradients():
    m = PINN(PINNConfig(hidden=16, layers=2))
    x = torch.rand(8, 1, requires_grad=True)
    y = torch.rand(8, 1, requires_grad=True)
    t = torch.rand(8, 1, requires_grad=True)
    loss = (pde_residual(m, x, y, t, Physics()) ** 2).mean()
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in m.parameters())


def _fvm_setup(n=16, H=2.0):
    grid = Grid.from_extent(n, n, (0, 10, 0, 10))
    h_s = torch.full((n, n), H, dtype=torch.float32)
    mcfg = FVMPINNConfig(hidden=16, layers=2, fourier_features=8,
                         x_range=(0, 10), y_range=(0, 10), t_range=(0, 1))
    model = FVMPINN(mcfg, h_s)
    scfg = Config(grid=grid, bc=TRANSMISSIVE, scheme="hllc", order=2)
    return grid, model, scfg, H


def test_fvm_predict_grid_shape_and_positive():
    grid, model, _, _ = _fvm_setup()
    U = model.predict_grid(0.3, grid)
    assert U.shape == (3, grid.ny, grid.nx)
    assert U.dtype == torch.float64
    assert U[0].min() > 0  # softplus keeps h strictly positive


def test_fvm_residual_zero_for_constant_still_prediction():
    """Zeroed final layer -> xi=0, hu=hv=0 -> uniform still lake; the discrete
    FV step reproduces it, so the FVM consistency residual is ~0 (inherited
    well-balancing)."""
    grid, model, scfg, H = _fvm_setup()
    _zero_last_layer(model.net)
    spec = FVMResidualSpec(cfg=scfg, times=[0.0, 0.1, 0.2], n_sub=2)
    loss, info = fvm_residual_loss(model, spec)
    assert float(loss) < 1e-16
    assert len(info["per_channel_mse"]) == 3


def test_fvm_residual_and_ic_have_gradients():
    grid, model, scfg, H = _fvm_setup()
    spec = FVMResidualSpec(cfg=scfg, times=[0.0, 0.05, 0.1], n_sub=1)
    U0 = conserved(torch.full((grid.ny, grid.nx), 2.0, dtype=torch.float64),
                   torch.zeros(grid.ny, grid.nx, dtype=torch.float64),
                   torch.zeros(grid.ny, grid.nx, dtype=torch.float64))
    loss = fvm_residual_loss(model, spec)[0] + ic_anchor_loss(model, grid, U0)
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters())


def test_data_anchor_uses_same_reconstruction_as_predict_grid():
    """The gauge misfit must be measured on the network's actual (h,hu,hv),
    i.e. the same softplus depth and h*vel reparametrization as predict_grid.
    Feeding the model's own grid prediction as the target => ~zero loss."""
    from ml.fvm_pinn import data_anchor_loss

    grid, model, _, _ = _fvm_setup(n=16)
    model.cfg.vel_scale = 8.0  # exercise the bounded-velocity path
    t = 0.5
    X, Y = grid.centers()
    xyt = torch.stack([X.reshape(-1), Y.reshape(-1),
                       torch.full((grid.ny * grid.nx,), t)], dim=1)
    U = model.predict_grid(t, grid)                     # (3, ny, nx)
    targets = torch.stack([U[0].reshape(-1), U[1].reshape(-1), U[2].reshape(-1)], dim=1)
    h_s = model.h_s.reshape(-1)
    loss = data_anchor_loss(model, xyt, h_s, targets)
    assert float(loss) < 1e-8


def test_train_reduces_loss(tmp_path):
    """Short FVM-PINN training on a lake-at-rest reduces the loss and writes
    artifacts."""
    grid, model, scfg, H = _fvm_setup(n=12)
    spec = FVMResidualSpec(cfg=scfg, times=[0.0, 0.1, 0.2], n_sub=1)
    U0 = conserved(torch.full((grid.ny, grid.nx), H, dtype=torch.float64),
                   torch.zeros(grid.ny, grid.nx, dtype=torch.float64),
                   torch.zeros(grid.ny, grid.nx, dtype=torch.float64))

    def loss_fn():
        r, info = fvm_residual_loss(model, spec)
        ic = ic_anchor_loss(model, grid, U0)
        return r + ic, {"residual": float(r), "ic": float(ic)}

    l0 = float(loss_fn()[0])
    out = train(model, loss_fn, TrainConfig(iters=40, lr=1e-3, log_every=10,
                                            out_dir=str(tmp_path / "run")))
    assert out["best_loss"] < l0
    assert (tmp_path / "run" / "history.csv").exists()
    assert (tmp_path / "run" / "best.pt").exists()
