# Design proposal: `state.py` / `grid.py` data layout

Status: **awaiting approval** (pre-Phase-1 gate requested in the project brief).
Date: 2026-07-03

## 1. Core tensor layout

One conserved-state tensor, no wrapper class in the hot path:

```
U : torch.Tensor, float64, shape (..., 3, Ny, Nx)
    channel axis -3:  0 = h, 1 = hu, 2 = hv
    y axis        -2:  Ny = ny + 2*ng   (rows, includes ghost cells)
    x axis        -1:  Nx = nx + 2*ng   (cols, includes ghost cells, contiguous/fast axis)
```

Design decisions and rationale:

- **Channel-first, negative-axis indexing everywhere.** All solver ops index with
  `U[..., 0, :, :]`, `U[..., :, j, i]`, `dim=-1/-2`. A leading batch dimension
  `(B, 3, Ny, Nx)` then works for free — this is what the FVM-PINN loss (Phase 3)
  and the neural-operator follow-up need: `FV_step` applied to a batch of
  network-predicted states at many collocation times in one call. Nothing in the
  classical solver ever assumes `U.ndim == 3`.
- **float64 mandatory in the solver.** `constants.py` holds `G = 9.81`,
  `H_EPS = 1e-6`, `CFL = 0.45`; `Grid` carries `dtype`/`device` and factories
  allocate accordingly. ML models may run float32 but must cast to float64 before
  entering any `swe.*` function (enforced by an assert in `solver.step`).
- **Ghost width `ng = 2`, always**, even for first-order runs. MUSCL needs 2; keeping
  it fixed means static shapes for `torch.compile` and no scheme-dependent branching.
- **1D cases are `ny = 1`** (degenerate 2D). Ghost rows in y are filled by the y-BC
  (transmissive); with y-uniform data all y-fluxes cancel identically, so the 1D
  Ritter/Stoker validations run through the exact same 2D code path.

## 2. `grid.py` — `Grid` dataclass

```python
@dataclass(frozen=True)
class Grid:
    nx: int; ny: int              # interior cell counts
    x0: float; y0: float          # domain lower-left corner
    dx: float; dy: float
    ng: int = 2
    dtype: torch.dtype = torch.float64
    device: torch.device = "cpu"
```

Derived (cached) members:

- `xc, yc` — 1D cell-center coordinates, interior only: `x0 + (i + 0.5) * dx`.
- `xc_g, yc_g` — center coordinates including ghosts (needed to evaluate bed/IC
  functions in ghost cells).
- `interior` — the slice tuple `(..., ng:ng+ny, ng:ng+nx)` exposed as a helper so
  call sites never hand-roll ghost offsets: `U_int = U[..., grid.i_int, grid.j_int]`
  via `grid.interior(U)`.

**Bed elevation** `z` lives at **cell centers**, shape `(Ny, Nx)` (ghosts included,
filled by zeroth-order extrapolation). Interface bed values are *not* stored:
Audusse et al. (2004) hydrostatic reconstruction builds
`z*_{i+1/2} = max(z_L, z_R)` from the (possibly MUSCL-reconstructed) traces at each
interface, so `wellbalance.py` computes them on the fly from the same reconstruction
that produced the state traces. This is the arrangement that makes lake-at-rest an
algebraic identity.

`z` is owned by the benchmark config, not by `Grid` — `Grid` is pure geometry.
Topography, IC, and friction coefficient live in a `Case` config object consumed by
`solver.run(cfg)`.

## 3. Boundary conditions — out-of-place ghost fill

BCs are **pure functions**: `apply_bc(U, grid, bc) -> U_new`. Ghost strips are
computed from interior slices and assembled with `torch.cat` — never in-place
assignment, so the whole step stays autograd-safe and compile-friendly.

Per-side spec (`bc.left/right/bottom/top`), initial vocabulary:

| type | ghost fill rule |
|---|---|
| `transmissive` | copy nearest interior cell (zeroth-order extrapolation) |
| `reflective` | mirror `h` and tangential momentum; negate normal momentum |
| `periodic` | wrap from opposite side |

Order of application: x-sides first, then y-sides (y pass reads the already-filled
x-ghost columns), which gives corner ghosts consistent values without special-casing.
BCs are re-applied before **every** RK stage flux evaluation, not once per step.

The Phase-2 progressive breach (time-varying internal wall) is *not* a boundary
condition in this design: it will be a reflective-face mask applied inside the flux
computation (`torch.where` on interface fluxes), so `grid.py`/`state.py` need no
hook for it now beyond noting that fluxes accept an optional interface mask.

## 4. `state.py` — primitive conversion with dry-cell guard

Pure functions, no class:

```python
def primitives(U, h_eps=H_EPS) -> tuple[h, u, v]
def conserved(h, u, v) -> U
```

The dry guard must be NaN-safe **in the backward pass**, not just the forward pass.
`torch.where(h > eps, hu/h, 0)` still backprops NaN through the untaken branch when
`h == 0`, so the denominator is made safe *before* the select:

```python
h_safe = torch.clamp(h, min=h_eps)
wet    = h > h_eps
u      = torch.where(wet, hu / h_safe, torch.zeros_like(h))
```

A config-selectable alternative (`desingularize: kurganov`) implements the
Kurganov–Petrova smoothed velocity `u = √2·h·hu / √(h⁴ + max(h⁴, ε⁴))` for the ML
phase, where a hard `where` can give poor gradients at the wet/dry front. Default
for classical runs is the plain guard above.

`wetdry.py` (separate module, per the brief) owns the positivity clamp and
velocity-zeroing applied *after* each update stage; `state.py` only guarantees safe
conversion of whatever it is given.

## 5. Interface/flux array conventions (forward-looking, fixes shapes now)

- x-interfaces: `nx + 1` per row → flux tensor `(..., 3, ny, nx+1)`; y-interfaces
  `(..., 3, ny+1, nx)`. Reconstruction produces left/right traces
  `UL, UR : (..., 3, ny, nx+1)` (and bottom/top for y) from the ghost-padded `U`.
- All flux kernels are `f(UL, UR) -> F` — pure, vectorized over the whole interface
  tensor, no cell loops, no `.item()`, no in-place ops.
- CFL: `dt = CFL * min(dx, dy)-style` computed from `max(|u|+√(gh), |v|+√(gh))`
  reduced **over wet cells only** via a `wet` mask and `torch.where(wet, speed, 0)`
  before the max (a global max over dry cells with garbage velocities would be
  wrong; a masked-select would be data-dependent shape — the `where`-then-`amax`
  form is compile-safe). The dt scalar is the one place a graph break is accepted
  (adaptive stepping is inherently data-dependent); for the FVM-PINN loss dt is a
  *fixed input*, so the differentiable path never touches the reduction.

## 6. Phase 0 note — coauthor data (already in `data/raw/`)

Inventory so far: 6 IC variants (step, rectangular, circular, Gaussian, parabolic,
triangular) × 3 schemes (HLL, Lax–Wendroff, MUSCL-"RS") × 5 snapshots
(t = 0.0, 0.5, 1.0, 1.5, 2.0 s). Each file is a **headerless 501×501 CSV of a single
field** — depth (or free surface) only, values ≈ 10 at t=0 for Variant 1.

Open questions for the coauthor (needed before the Phase-1 reconciliation gate;
will be tabulated formally in `reports/phase0_inventory.md`):

1. Domain extents and whether the 501 values are cell centers or nodes
   (501 is node-like: 500 cells + 1).
2. Is the stored field `h` or free surface `η = h + z`? Any topography/friction?
3. Where are `hu, hv` (or `u, v`)? Depth-only outputs cannot support momentum
   metrics or the sparse-data gauge term for momentum.
4. g, wet-dry threshold, boundary treatment, CFL used.
5. What does "RS" in MUSCLRS stand for (Riemann solver? which one?), and what
   limiter?

## 7. What I will build immediately after approval

1. `pyproject.toml` (uv, pinned: torch, numpy, matplotlib, pyyaml, pytest, tqdm) +
   package skeleton.
2. `constants.py`, `grid.py`, `state.py` + pytest units (BC fills, dry-guard values
   *and gradients*, 1D-as-ny=1 round trip).
3. Phase 0 proper: `qc.py`, `coauthor.py`, `reports/phase0_inventory.md` — the QC
   diagnostics (mass drift, min h, azimuthal symmetry on Variant 3, NaN scan) run
   fine on depth-only data even while the convention questions above are pending.
4. Then Phase 1 fluxes/reconstruction/well-balancing per the brief.
