# Neural models vs the reference depth solution

Runs: `runs/revision_2026_10/strongform_bed`. Reference: our HLLC MUSCL-van Leer on 512^2 cells, block-averaged onto the 128^2 evaluation grid at t = 2.0 s. RMSE(h) in metres over the grid cells; L1(h) integrated over cell areas (m^3); mass drift = max_t |V(t)/V(0) - 1| over the five output times. Frozen = the initial depth field held fixed. Classical row: our HLLC MUSCL-van Leer at the same evaluation resolution (inter-solver difference; cf. reports/reconciliation.md). Neural entries are mean ± sample sd over seeds.

## Depth RMSE [m] at t = 2 s

| Variant | Frozen | Classical (HLLC) | PINN (primitive) | PINN (conservative) |
|---|---|---|---|---|
| 1 Step | 1.342 | 0.071 | 0.306 ± 0.006 | 0.424 ± 0.021 |
| 2 Rectangular | 0.712 | 0.090 | 0.182 ± 0.003 | 0.125 ± 0.001 |
| 3 Circular | 1.466 | 0.108 | 0.354 ± 0.009 | 0.485 ± 0.030 |
| 4 Gaussian | 0.580 | 0.002 | 0.028 ± 0.010 | 0.059 ± 0.001 |
| 5 Parabolic | 2.391 | 0.166 | 0.561 ± 0.031 | 0.514 ± 0.010 |
| 6 Triangular | 1.478 | 0.118 | 0.312 ± 0.005 | 0.354 ± 0.009 |

## L1(h) [m^3] at t = 2 s

| Variant | Classical (HLLC) | PINN (primitive) | PINN (conservative) |
|---|---|---|---|
| 1 Step | 248 | 1.02e+03 | 2.26e+03 |
| 2 Rectangular | 372 | 869 | 808 |
| 3 Circular | 392 | 1.63e+03 | 3.18e+03 |
| 4 Gaussian | 6.38 | 208 | 442 |
| 5 Parabolic | 707 | 2.65e+03 | 3.51e+03 |
| 6 Triangular | 369 | 1.24e+03 | 1.88e+03 |

## Relative mass drift

| Variant | PINN (primitive) | PINN (conservative) |
|---|---|---|
| 1 Step | 0.001 | 0.019 |
| 2 Rectangular | 0.005 | 0.041 |
| 3 Circular | 0.012 | 0.049 |
| 4 Gaussian | 0.003 | 0.011 |
| 5 Parabolic | 0.008 | 0.067 |
| 6 Triangular | 0.009 | 0.022 |

## Per-seed values (RMSE / L1 / drift)

- 1 Step / PINN (primitive): 0.3101 / 1003 / 0.0004; 0.3091 / 1057 / 0.0005; 0.2987 / 1001 / 0.0034
- 1 Step / PINN (conservative): 0.4299 / 2475 / 0.0218; 0.4003 / 1826 / 0.0125; 0.4412 / 2493 / 0.0221
- 2 Rectangular / PINN (primitive): 0.1820 / 845.6 / 0.0009; 0.1789 / 852.9 / 0.0104; 0.1845 / 908.1 / 0.0039
- 2 Rectangular / PINN (conservative): 0.1250 / 852.9 / 0.0432; 0.1244 / 759 / 0.0358; 0.1268 / 812.4 / 0.0449
- 3 Circular / PINN (primitive): 0.3598 / 1628 / 0.0137; 0.3433 / 1617 / 0.0163; 0.3579 / 1642 / 0.0048
- 3 Circular / PINN (conservative): 0.4695 / 2875 / 0.0518; 0.4663 / 3057 / 0.0152; 0.5192 / 3606 / 0.0790
- 4 Gaussian / PINN (primitive): 0.0349 / 283.5 / 0.0023; 0.0172 / 113.7 / 0.0016; 0.0332 / 225.9 / 0.0053
- 4 Gaussian / PINN (conservative): 0.0576 / 423.7 / 0.0068; 0.0605 / 444.6 / 0.0101; 0.0591 / 456.3 / 0.0149
- 5 Parabolic / PINN (primitive): 0.5405 / 2485 / 0.0061; 0.5455 / 2499 / 0.0152; 0.5957 / 2964 / 0.0024
- 5 Parabolic / PINN (conservative): 0.5206 / 3562 / 0.0544; 0.5031 / 3370 / 0.0562; 0.5186 / 3585 / 0.0903
- 6 Triangular / PINN (primitive): 0.3165 / 1170 / 0.0041; 0.3134 / 1268 / 0.0197; 0.3059 / 1277 / 0.0041
- 6 Triangular / PINN (conservative): 0.3606 / 2009 / 0.0206; 0.3566 / 1813 / 0.0258; 0.3436 / 1811 / 0.0187
