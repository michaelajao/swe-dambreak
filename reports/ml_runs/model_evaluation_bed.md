# Neural models vs the reference depth solution

Runs: `runs/revision_2026_10/strongform_bed`. Reference: MUSCLRS depth files (501x501 nodes) at t = 2.0 s, bilinearly regridded onto the 128^2 evaluation grid. RMSE(h) in metres over the grid cells; L1(h) integrated over cell areas (m^3); mass drift = max_t |V(t)/V(0) - 1| over the five output times. Frozen = the initial depth field held fixed. Classical row: our HLLC MUSCL-van Leer at the same evaluation resolution (inter-solver difference; cf. reports/reconciliation.md). Neural entries are mean ± sample sd over seeds.

## Depth RMSE [m] at t = 2 s

| Variant | Frozen | Classical (HLLC) | PINN (primitive) | PINN (conservative) |
|---|---|---|---|---|
| 1 Step | 1.579 | 0.601 | 0.723 ± 0.012 | 0.754 ± 0.023 |
| 2 Rectangular | 0.713 | 0.136 | 0.181 ± 0.002 | 0.170 ± 0.001 |
| 3 Circular | 1.472 | 0.222 | 0.324 ± 0.007 | 0.536 ± 0.040 |
| 4 Gaussian | 0.498 | 0.113 | 0.112 ± 0.002 | 0.141 ± 0.002 |
| 5 Parabolic | 2.386 | 0.192 | 0.558 ± 0.031 | 0.516 ± 0.010 |
| 6 Triangular | 1.468 | 0.174 | 0.286 ± 0.005 | 0.372 ± 0.011 |

## L1(h) [m^3] at t = 2 s

| Variant | Classical (HLLC) | PINN (primitive) | PINN (conservative) |
|---|---|---|---|
| 1 Step | 2.19e+03 | 2.96e+03 | 3.86e+03 |
| 2 Rectangular | 609 | 892 | 1.02e+03 |
| 3 Circular | 863 | 1.53e+03 | 3.45e+03 |
| 4 Gaussian | 398 | 524 | 756 |
| 5 Parabolic | 773 | 2.61e+03 | 3.49e+03 |
| 6 Triangular | 584 | 1.17e+03 | 1.94e+03 |

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

- 1 Step / PINN (primitive): 0.7335 / 3031 / 0.0004; 0.7103 / 2842 / 0.0005; 0.7251 / 3008 / 0.0034
- 1 Step / PINN (conservative): 0.7669 / 4099 / 0.0218; 0.7266 / 3462 / 0.0125; 0.7676 / 4008 / 0.0221
- 2 Rectangular / PINN (primitive): 0.1795 / 855.1 / 0.0009; 0.1822 / 906.9 / 0.0104; 0.1821 / 914 / 0.0039
- 2 Rectangular / PINN (conservative): 0.1718 / 1055 / 0.0432; 0.1689 / 983.7 / 0.0358; 0.1704 / 1014 / 0.0449
- 3 Circular / PINN (primitive): 0.3258 / 1507 / 0.0137; 0.3165 / 1524 / 0.0163; 0.3302 / 1552 / 0.0048
- 3 Circular / PINN (conservative): 0.5314 / 3190 / 0.0518; 0.4986 / 3209 / 0.0152; 0.5781 / 3949 / 0.0790
- 4 Gaussian / PINN (primitive): 0.1105 / 586.6 / 0.0023; 0.1119 / 465.7 / 0.0016; 0.1139 / 520.1 / 0.0053
- 4 Gaussian / PINN (conservative): 0.1428 / 726.6 / 0.0068; 0.1407 / 761.2 / 0.0101; 0.1392 / 780.1 / 0.0149
- 5 Parabolic / PINN (primitive): 0.5373 / 2449 / 0.0061; 0.5435 / 2459 / 0.0152; 0.5945 / 2934 / 0.0024
- 5 Parabolic / PINN (conservative): 0.5200 / 3533 / 0.0544; 0.5041 / 3352 / 0.0562; 0.5228 / 3591 / 0.0903
- 6 Triangular / PINN (primitive): 0.2903 / 1090 / 0.0041; 0.2868 / 1200 / 0.0197; 0.2811 / 1224 / 0.0041
- 6 Triangular / PINN (conservative): 0.3816 / 2079 / 0.0206; 0.3747 / 1869 / 0.0258; 0.3610 / 1886 / 0.0187
