# Classical schemes quantified (LW, HLL, MUSCL-RS on the six variants)

Computed from the stored 501x501 depth snapshots. Mass drift = max_t |V(t)/V(0)-1| (closed reflective box). min h = minimum depth over all snapshots (outer depth is 0.2 m on variant 2, 1 m elsewhere). Symmetry = relative L1 asymmetry at t = 2 s. RMSE in metres at t = 2 s on the node grid against MUSCL-RS, and, regridded, against our well-balanced HLLC MUSCL-van Leer solution on 500x500 cells (inter-solver difference, not an exact solution).

| Variant | Scheme | mass drift | min h [m] | sym x | sym y | sym rot90 | RMSE vs MUSCL-RS | RMSE vs ours |
|---|---|---|---|---|---|---|---|---|
| 1 Step | LW | 6.63e-04 | 0.977 | 1.1e+00 | 9.5e-16 | 7.2e-01 | 1.320 | 1.572 |
| 1 Step | HLL | 3.27e-07 | 0.955 | 1.4e+00 | 6.8e-17 | 7.9e-01 | 0.069 | 0.605 |
| 1 Step | MUSCLRS | 2.85e-07 | 0.948 | 1.4e+00 | 2.7e-16 | 7.9e-01 | 0.000 | 0.621 |
| 2 Rectangular | LW | 1.51e-02 | 0.000195 | 2.6e-12 | 3.6e-12 | 3.6e-12 | 0.583 | 0.558 |
| 2 Rectangular | HLL | 3.66e-15 | 0.199 | 5.3e-02 | 5.3e-02 | 5.3e-02 | 0.039 | 0.116 |
| 2 Rectangular | MUSCLRS | 3.77e-15 | 0.199 | 5.3e-02 | 5.3e-02 | 5.3e-02 | 0.000 | 0.108 |
| 3 Circular | LW | 8.60e-02 | 0.0001 | 3.3e-01 | 3.3e-01 | 3.3e-01 | 1.553 | 1.275 |
| 3 Circular | HLL | 2.17e-08 | 1 | 1.4e-04 | 1.4e-04 | 1.4e-04 | 0.226 | 0.093 |
| 3 Circular | MUSCLRS | 4.55e-15 | 0.99 | 4.4e-02 | 4.4e-02 | 4.4e-02 | 0.000 | 0.208 |
| 4 Gaussian | LW | 3.58e-05 | 1.39e-10 | 1.1e-15 | 1.1e-15 | 1.1e-15 | 0.816 | 0.756 |
| 4 Gaussian | HLL | 1.09e-07 | 1.7e-10 | 7.7e-02 | 7.7e-02 | 7.7e-02 | 0.011 | 0.109 |
| 4 Gaussian | MUSCLRS | 1.08e-07 | 1.7e-10 | 7.8e-02 | 7.8e-02 | 7.8e-02 | 0.000 | 0.112 |
| 5 Parabolic | LW | 2.04e-01 | 0.0001 | 1.1e-07 | 7.5e-01 | 7.1e-01 | 1.619 | 1.412 |
| 5 Parabolic | HLL | 1.91e-03 | 0.872 | 1.3e-16 | 5.3e-01 | 5.0e-01 | 0.175 | 0.148 |
| 5 Parabolic | MUSCLRS | 2.05e-03 | 0.945 | 2.6e-02 | 5.3e-01 | 5.1e-01 | 0.000 | 0.143 |
| 6 Triangular | LW | 6.70e-02 | 0.000114 | 3.2e-09 | 5.9e-01 | 4.8e-01 | 1.287 | 1.152 |
| 6 Triangular | HLL | 2.85e-08 | 0.993 | 9.7e-17 | 5.4e-01 | 4.3e-01 | 0.155 | 0.090 |
| 6 Triangular | MUSCLRS | 4.33e-15 | 0.946 | 3.2e-02 | 5.4e-01 | 4.3e-01 | 0.000 | 0.148 |
