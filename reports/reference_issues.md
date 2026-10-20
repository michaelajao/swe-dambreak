# Two defects in the stored reference solutions (for the solver authors)

Found on 2026-10-20 while re-scoring the neural runs. Both affect the HLL and
MUSCL-RS drops under `data/`, which the paper uses as the reference for every
neural table. Numbers: `reports/bed_diagnostic.md`,
`reports/classical_quantification.md`; scripts:
`experiments/run_bed_diagnostic.py`, `experiments/run_classical_quantification.py`.

## 1. Step case (variant 1): waves travel sqrt(2) too fast

Away from the hump (row y = 10 m, where Z < 1e-3 m) the stored step solutions
should be a plain 1D Riemann problem with g = 2. Front positions measured from
x = 50 m, stored MUSCL-RS against our solver with g = 2:

| t [s] | shock, theirs / ours | rarefaction head, theirs / ours |
|---|---|---|
| 0.5 | 3.1 / 2.1 m (1.48) | 3.5 / 2.3 m (1.52) |
| 1.0 | 6.1 / 4.3 m (1.42) | 6.5 / 4.5 m (1.44) |
| 1.5 | 9.1 / 6.5 m (1.40) | 9.7 / 6.7 m (1.45) |
| 2.0 | 12.1 / 8.7 m (1.39) | 12.7 / 8.9 m (1.43) |

The ratio is constant in time, so it is a wave-speed error, not a time offset.
It equals sqrt(2) within the grid resolution, which is what g = 4 or a pressure
flux of g h^2 instead of g h^2 / 2 would produce. Our solver reproduces the
Stoker solution at g = 9.81 to first order (reports/solver_validation.md), and
the rarefaction head of our run moves at sqrt(g h_l) = 4.47 m/s as it should.
The other five variants do not show the speed error. Please check the step
run's g and flux.

## 2. Bed source too weak or absent (variants 2-6, clearest on 4)

Depth RMSE [m] at t = 2 s between the stored solutions and our well-balanced
solver run over the hump and over a flat bed with the same initial depth:

| Variant | bed effect (ours hump vs flat) | MUSCL-RS vs hump | MUSCL-RS vs flat |
|---|---|---|---|
| 2 Rectangular | 0.094 | 0.108 | 0.068 |
| 3 Circular | 0.169 | 0.208 | 0.127 |
| 4 Gaussian | 0.090 | 0.112 | 0.068 |
| 5 Parabolic | 0.112 | 0.143 | 0.109 |
| 6 Triangular | 0.128 | 0.148 | 0.106 |

The stored MUSCL-RS solution is closer to the flat-bed solution than to the
hump solution on all five. On the Gaussian case the centreline profile is on
top of our flat-bed profile (peak 6.0 m, against 5.35 m over the hump); see
`reports/figures/reference_bed_and_step_profiles.png`. HLL behaves similarly
on variants 2 and 4 but follows the hump on 3, 5 and 6. Please check how Z
and its gradient enter the MUSCL-RS source term.

## 3. Lax-Wendroff runs

In a closed box the LW runs lose 1.5% (V2), 8.6% (V3), 20% (V5) and 6.7% (V6)
of their volume, and the depth falls to 1e-4 m on V2, V3, V5 and V6, where the
initial minimum is 0.2 m or 1 m (a positivity clamp is being hit). Their RMSE
against MUSCL-RS is 0.6-1.6 m, above the frozen-initial-state error on V4.

## Effect on the neural results

Scored against our own fine-grid HLLC solution (512^2, block-averaged to
128^2), the neural results change in magnitude but not in their conclusions.
With the bed term in the residual the primitive PINN Gaussian error is 0.028 m
against our reference and 0.112 m against MUSCL-RS; without it, 0.095 and
0.071 m. The MUSCL-RS Gaussian reference therefore rewards the network that
omits the bed. Full tables: `reports/ml_runs/model_evaluation_ours_bed.md`,
`model_evaluation_ours_nobed.md`, `model_evaluation_bed.md`.
