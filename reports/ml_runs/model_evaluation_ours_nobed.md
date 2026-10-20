# Neural models vs the reference depth solution

Runs: `runs/ml`. Reference: our HLLC MUSCL-van Leer on 512^2 cells, block-averaged onto the 128^2 evaluation grid at t = 2.0 s. RMSE(h) in metres over the grid cells; L1(h) integrated over cell areas (m^3); mass drift = max_t |V(t)/V(0) - 1| over the five output times. Frozen = the initial depth field held fixed. Classical row: our HLLC MUSCL-van Leer at the same evaluation resolution (inter-solver difference; cf. reports/reconciliation.md). Neural entries are mean ± sample sd over seeds.

## Depth RMSE [m] at t = 2 s

| Variant | Frozen | Classical (HLLC) | PINN (primitive) | PINN (conservative) | FVM-PINN (data-free) |
|---|---|---|---|---|---|
| 1 Step | 1.342 | 0.071 | 0.314 ± 0.009 | 0.347 ± 0.010 | 1.403 ± 0.014 |
| 2 Rectangular | 0.712 | 0.090 | 0.176 ± 0.004 | 0.146 ± 0.002 | 0.916 ± 0.059 |
| 3 Circular | 1.466 | 0.108 | 0.380 ± 0.006 | 0.419 ± 0.029 | 2.127 ± 0.036 |
| 4 Gaussian | 0.580 | 0.002 | 0.095 ± 0.003 | 0.089 ± 0.008 | 1.160 ± 0.036 |
| 5 Parabolic | 2.391 | 0.166 | 0.569 ± 0.006 | 0.491 ± 0.013 | 1.769 ± 0.008 |
| 6 Triangular | 1.478 | 0.118 | 0.341 ± 0.004 | 0.367 ± 0.015 | 1.525 ± 0.078 |

## L1(h) [m^3] at t = 2 s

| Variant | Classical (HLLC) | PINN (primitive) | PINN (conservative) | FVM-PINN (data-free) |
|---|---|---|---|---|
| 1 Step | 248 | 982 | 1.44e+03 | 9.71e+03 |
| 2 Rectangular | 372 | 801 | 872 | 4.85e+03 |
| 3 Circular | 392 | 1.67e+03 | 2.7e+03 | 1.35e+04 |
| 4 Gaussian | 6.38 | 379 | 596 | 5.17e+03 |
| 5 Parabolic | 707 | 2.72e+03 | 3.18e+03 | 1.35e+04 |
| 6 Triangular | 369 | 1.29e+03 | 2.11e+03 | 8.91e+03 |

## Relative mass drift

| Variant | PINN (primitive) | PINN (conservative) | FVM-PINN (data-free) |
|---|---|---|---|
| 1 Step | 0.001 | 0.007 | 0.066 |
| 2 Rectangular | 0.004 | 0.032 | 0.252 |
| 3 Circular | 0.008 | 0.055 | 0.340 |
| 4 Gaussian | 0.001 | 0.013 | 0.542 |
| 5 Parabolic | 0.008 | 0.066 | 0.454 |
| 6 Triangular | 0.004 | 0.035 | 0.269 |

## Per-seed values (RMSE / L1 / drift)

- 1 Step / PINN (primitive): 0.3106 / 998.7 / 0.0007; 0.3073 / 924.9 / 0.0008; 0.3242 / 1023 / 0.0029
- 1 Step / PINN (conservative): 0.3544 / 1485 / 0.0043; 0.3520 / 1515 / 0.0114; 0.3359 / 1335 / 0.0058
- 1 Step / FVM-PINN (data-free): 1.4019 / 9834 / 0.0575; 1.3903 / 9559 / 0.0805; 1.4177 / 9724 / 0.0594
- 2 Rectangular / PINN (primitive): 0.1760 / 851 / 0.0024; 0.1800 / 747.7 / 0.0022; 0.1724 / 805.8 / 0.0062
- 2 Rectangular / PINN (conservative): 0.1480 / 867.4 / 0.0348; 0.1463 / 887.8 / 0.0317; 0.1432 / 860 / 0.0304
- 2 Rectangular / FVM-PINN (data-free): 0.8483 / 4516 / 0.2467; 0.9435 / 4958 / 0.2512; 0.9570 / 5080 / 0.2582
- 3 Circular / PINN (primitive): 0.3864 / 1709 / 0.0083; 0.3809 / 1660 / 0.0102; 0.3741 / 1639 / 0.0042
- 3 Circular / PINN (conservative): 0.4454 / 2930 / 0.0390; 0.3875 / 2368 / 0.0540; 0.4244 / 2793 / 0.0713
- 3 Circular / FVM-PINN (data-free): 2.1505 / 1.318e+04 / 0.3226; 2.1454 / 1.367e+04 / 0.3626; 2.0859 / 1.364e+04 / 0.3347
- 4 Gaussian / PINN (primitive): 0.0969 / 378.5 / 0.0020; 0.0920 / 351.4 / 0.0011; 0.0972 / 407.9 / 0.0007
- 4 Gaussian / PINN (conservative): 0.0971 / 698.1 / 0.0191; 0.0815 / 480.8 / 0.0077; 0.0892 / 609.5 / 0.0131
- 4 Gaussian / FVM-PINN (data-free): 1.1784 / 5287 / 0.5967; 1.1179 / 5005 / 0.5342; 1.1828 / 5221 / 0.4962
- 5 Parabolic / PINN (primitive): 0.5750 / 2813 / 0.0036; 0.5679 / 2723 / 0.0128; 0.5632 / 2632 / 0.0064
- 5 Parabolic / PINN (conservative): 0.4990 / 3174 / 0.0536; 0.4975 / 3250 / 0.0711; 0.4761 / 3115 / 0.0737
- 5 Parabolic / FVM-PINN (data-free): 1.7606 / 1.339e+04 / 0.4888; 1.7687 / 1.339e+04 / 0.5256; 1.7763 / 1.363e+04 / 0.3469
- 6 Triangular / PINN (primitive): 0.3367 / 1302 / 0.0056; 0.3442 / 1257 / 0.0047; 0.3427 / 1302 / 0.0031
- 6 Triangular / PINN (conservative): 0.3585 / 2046 / 0.0246; 0.3594 / 1950 / 0.0313; 0.3844 / 2348 / 0.0501
- 6 Triangular / FVM-PINN (data-free): 1.4508 / 8437 / 0.3078; 1.6061 / 8912 / 0.2175; 1.5189 / 9384 / 0.2831
