# SciML comparison — PINNs and FVM-informed PINN vs classical schemes

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

**ca_circular_wet (paper Variant 3)**

| method | L1(h) | drift | notes |
|---|---|---|---|
| classical HLLC–van Leer | 3.91e+02 | 0.0 | same-grid reference tier |
| PINN (primitive) | 1.65e+03 ± 3.1e+01 | 7.5e-03 | best neural |
| PINN (conservative) | 2.49e+03 ± 1.5e+02 | 3.8e-02 | |
| FVM-PINN (physics only) | 1.29e+04 ± 6.7e+02 | 3.9e-01 | low-momentum collapse |

**ca_step (paper Variant 1)**

| method | L1(h) | drift |
|---|---|---|
| classical HLLC–van Leer | 2.48e+02 | 0.0 |
| PINN (primitive) | 1.09e+03 ± 5.1e+01 | 1.7e-03 |
| PINN (conservative) | 2.19e+03 ± 7.7e+02 | 1.9e-02 |
| FVM-PINN (physics only) | 8.66e+03 ± 1.1e+03 | 4.6e-02 |

**ca_circular_dry (dry downstream)**

| method | L1(h) | drift |
|---|---|---|
| classical HLLC–van Leer | 3.80e+02 | ~1e-16 |
| PINN (primitive) | 1.68e+03 ± 1.5e+02 | 1.5e-02 |
| PINN (conservative) | 2.04e+03 ± 3.3e+02 | 3.0e-02 |
| FVM-PINN (physics only) | 9.02e+03 ± 1.6e+02 | 4.1e-01 |
| FVM-PINN + data (256 gauges) | 5.67e+03 ± 2.0e+02 | 1.4e-01 |

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

*Pending — run in progress. This case (g=9.81, R=11, flat bed, t=1.2 s) has a
much stronger shock, where the FVM-PINN's shock-capturing structure should be
worth more relative to the strong-form PINNs than in the near-shock-free g=2
coauthor regime.*

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
