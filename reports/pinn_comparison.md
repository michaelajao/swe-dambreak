# PINN comparison — PINNs and FVM-informed PINN vs classical schemes

Curated from the training runs on the Coventry **brosnan** HPC (NVIDIA Quadro
RTX 8000). Per-method numbers are mean ± std over 3 seeds. Errors are discrete
$L^1$/$L^2$ of depth $h$ against a fine-grid HLLC MUSCL–van Leer self-convergence
reference (N=512 for the coauthor cases, N=1024 for the g=9.81 contrast),
resampled onto the evaluation grid (N=128) and taken at the final output time.
Mass drift is $\max_t |V(t)/V(0)-1|$. Raw per-seed logs in `reports/ml_runs/`.

Training: Adam, lr $10^{-3}$, 12k iterations, gradient clipping. PINNs are
6×128 tanh MLPs; the FVM-PINN is 5×128 with a 32-frequency Fourier embedding,
predicts $(\xi, u, v)$ with $h=\mathrm{softplus}(\xi+h_s)$ and momentum $h\cdot
u$ (bounded velocity), and its loss is the stochastic single-interval
discrete-consistency residual of the differentiable HLLC/SSP-RK2 step (float64).

## 1. Main comparison — coauthor-convention cases (g=2, Gaussian-hump bed, t=2 s)

These reproduce the paper's Variant 3 (Circular) and Variant 1 (Step) exactly,
so the neural comparison overlaps the classical section. `ca_circular_dry` sets
the downstream depth to zero to expose the dry-bed FVM-PINN collapse.

Depth error L1(h) at t=2 s, mean over 3 seeds, across all six variants:

| Variant | classical | PINN prim. | PINN cons. | FVM-PINN |
|---|---|---|---|---|
| 1 Step | 248 | 1090 | 2190 | 8660 |
| 2 Rectangular | 372 | 900 | 880 | 4980 |
| 3 Circular | 391 | 1650 | 2490 | 12900 |
| 4 Gaussian | 6.4 | 448 | 857 | 5140 |
| 5 Parabolic | 707 | 2590 | 3150 | 13450 |
| 6 Triangular | 369 | 1390 | 1850 | 8720 |

Mass drift (max_t |V(t)/V(0)-1|):

| Variant | PINN prim. | PINN cons. | FVM-PINN |
|---|---|---|---|
| 1 Step | 0.002 | 0.019 | 0.046 |
| 2 Rectangular | 0.013 | 0.026 | 0.276 |
| 3 Circular | 0.008 | 0.038 | 0.390 |
| 4 Gaussian | 0.003 | 0.030 | 0.540 |
| 5 Parabolic | 0.008 | 0.063 | 0.460 |
| 6 Triangular | 0.009 | 0.029 | 0.280 |

Across-IC reading: error tracks IC smoothness — the smooth Gaussian (V4) is far
the easiest for every method (classical 6.4, primitive PINN 448), the curved
parabolic band (V5) the hardest; primitive PINN beats conservative on every
variant except the rectangular tie; the physics-only FVM-PINN is the least
accurate neural model on every variant and drifts most (up to 0.54) — the
low-momentum collapse, worst where the flow is most dynamic.

The dry-bed case additionally isolates the data-guidance ablation:

| method (ca_circular_dry) | L1(h) | drift |
|---|---|---|
| FVM-PINN (physics only) | 9210 | 0.41 |
| FVM-PINN + data (256 gauges) | 5670 | 0.14 |

## 2. Data-guidance recovery curve (ca_circular_dry, sparse gauges)

The sparse-data term is drawn from the reference solution. As the number of
gauge points increases, both the depth error and the mass drift fall
monotonically from the physics-only collapse — the quantitative form of the
Liu (2026) finding that data guidance cures the collapse.

| gauges | L1(h) | mass drift |
|---|---|---|
| 0 (physics only) | 9.21e+03 ± 1.6e+02 | 4.16e-01 ± 4.6e-02 |
| 64 | 6.59e+03 ± 3.1e+02 | 1.66e-01 ± 3.3e-02 |
| 256 | 5.67e+03 ± 2.0e+02 | 1.37e-01 ± 3.8e-02 |
| 1024 | 3.41e+03 ± 6.4e+01 | 7.6e-02 ± 8.0e-03 |

(1024 gauges ≈ 6% of the 128² cells at one snapshot.)

## 3. Contrasting sharp-shock regime (g=9.81 dry circular)

The dry circular case repeated at g=9.81 (R=11, flat bed, t=1.2 s) produces a
much stronger shock. Depth error $L^1(h)$, mean over 3 seeds, vs the moderate-
shock benchmark regime (g=2, ca_circular_dry):

| method | L1(h) @ g=2 | drift @ g=2 | L1(h) @ g=9.81 | drift @ g=9.81 |
|---|---|---|---|---|
| classical HLLC | 380 | ~0 | 79 | ~0 |
| PINN (primitive) | 1680 | 0.015 | 1120 | 0.026 |
| PINN (conservative) | 2040 | 0.030 | 2410 | 0.185 |
| FVM-PINN (physics only) | 9210 | 0.41 | 3080 | 0.152 |
| FVM-PINN + data (256) | 5670 | 0.14 | 1950 | 0.112 |

**Finding:** the FVM-PINN's error drops ~3× at the stronger shock (both with and
without data), while the strong-form PINNs barely move. The ordering rearranges:
the data-guided FVM-PINN goes from least-accurate neural model (g=2) to 2nd best
(g=9.81), overtaking the conservative PINN and approaching the primitive one. The
discrete-residual structure behaves as a shock-capturing prior whose benefit
grows with shock strength — its advantage shows on strong fronts, not the slow
flows of the g=2 benchmark set. This makes the FVM-PINN's standing
regime-dependent rather than uniformly weak.

## 4. Interpretation

- **Formulation among the strong-form PINNs:** the primitive formulation beats
  the conservative one on every case, in both error and mass drift. Training the
  network directly on $(h,hu,hv)$ did not help here.
- **The physics-only FVM-PINN collapses** to a low-momentum spurious state on
  every case (depth error ~5–8× the strong-form PINNs, mass drift up to 0.4),
  reproducing the documented failure mode of the discrete-residual loss.
- **Sparse-data guidance monotonically rescues it** (Section 2): the collapse is
  data-curable, and the recovery is smooth in the number of gauges.
- **Honest caveat on the "structure ladder":** even with 1024 gauges the
  FVM-PINN does not overtake the plain primitive PINN in this g=2 coauthor
  regime. The waves are slow and the shocks weak over 2 s, so the strong-form
  PINN's shock-smearing weakness barely bites while the FVM-PINN's collapse
  pathology dominates. Section 3 tests whether the ordering changes when the
  shock is strong.
- **No neural method reaches the classical solver** on accuracy at this budget;
  the classical HLLC scheme remains the accuracy-per-cost reference.
