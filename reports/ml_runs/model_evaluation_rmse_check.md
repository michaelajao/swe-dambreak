# Neural models vs the reference depth solution

Runs: `runs/ml`. Reference: MUSCLRS depth files (501x501 nodes) at t = 2.0 s, bilinearly regridded onto the 128^2 evaluation grid. RMSE(h) in metres over the grid cells; L1(h) integrated over cell areas (m^3); mass drift = max_t |V(t)/V(0) - 1| over the five output times. Frozen = the initial depth field held fixed. Classical row: our HLLC MUSCL-van Leer at the same evaluation resolution (inter-solver difference; cf. reports/reconciliation.md). Neural entries are mean ± sample sd over seeds.

## Depth RMSE [m] at t = 2 s

| Variant | Frozen | Classical (HLLC) | PINN (primitive) | PINN (conservative) | FVM-PINN (data-free) |
|---|---|---|---|---|---|
| 1 Step | 1.579 | 0.601 | 0.732 ± 0.003 | 0.717 ± 0.008 | 1.573 ± 0.006 |
| 2 Rectangular | 0.713 | 0.136 | 0.167 ± 0.003 | 0.132 ± 0.002 | 0.953 ± 0.058 |
| 3 Circular | 1.472 | 0.222 | 0.348 ± 0.002 | 0.441 ± 0.025 | 2.172 ± 0.036 |
| 4 Gaussian | 0.498 | 0.113 | 0.071 ± 0.002 | 0.091 ± 0.010 | 1.169 ± 0.038 |
| 5 Parabolic | 2.386 | 0.192 | 0.564 ± 0.006 | 0.484 ± 0.010 | 1.774 ± 0.008 |
| 6 Triangular | 1.468 | 0.174 | 0.304 ± 0.008 | 0.370 ± 0.018 | 1.543 ± 0.078 |

## L1(h) [m^3] at t = 2 s

| Variant | Classical (HLLC) | PINN (primitive) | PINN (conservative) | FVM-PINN (data-free) |
|---|---|---|---|---|
| 1 Step | 2.19e+03 | 3e+03 | 3.23e+03 | 1.1e+04 |
| 2 Rectangular | 609 | 790 | 836 | 4.99e+03 |
| 3 Circular | 863 | 1.55e+03 | 2.77e+03 | 1.35e+04 |
| 4 Gaussian | 398 | 332 | 601 | 5.17e+03 |
| 5 Parabolic | 773 | 2.67e+03 | 3.13e+03 | 1.35e+04 |
| 6 Triangular | 584 | 1.15e+03 | 2.14e+03 | 8.94e+03 |

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

- 1 Step / PINN (primitive): 0.7315 / 2968 / 0.0007; 0.7292 / 2959 / 0.0008; 0.7356 / 3069 / 0.0029
- 1 Step / PINN (conservative): 0.7223 / 3308 / 0.0043; 0.7219 / 3194 / 0.0114; 0.7074 / 3196 / 0.0058
- 1 Step / FVM-PINN (data-free): 1.5667 / 1.112e+04 / 0.0575; 1.5773 / 1.101e+04 / 0.0805; 1.5763 / 1.094e+04 / 0.0594
- 2 Rectangular / PINN (primitive): 0.1641 / 818.4 / 0.0024; 0.1708 / 742.7 / 0.0022; 0.1667 / 809.4 / 0.0062
- 2 Rectangular / PINN (conservative): 0.1331 / 816.1 / 0.0348; 0.1329 / 851.1 / 0.0317; 0.1298 / 840.6 / 0.0304
- 2 Rectangular / FVM-PINN (data-free): 0.8863 / 4655 / 0.2467; 0.9804 / 5093 / 0.2512; 0.9936 / 5214 / 0.2582
- 3 Circular / PINN (primitive): 0.3496 / 1548 / 0.0083; 0.3464 / 1545 / 0.0102; 0.3491 / 1564 / 0.0042
- 3 Circular / PINN (conservative): 0.4593 / 2945 / 0.0390; 0.4124 / 2459 / 0.0540; 0.4509 / 2893 / 0.0713
- 3 Circular / FVM-PINN (data-free): 2.1969 / 1.319e+04 / 0.3226; 2.1892 / 1.368e+04 / 0.3626; 2.1303 / 1.364e+04 / 0.3347
- 4 Gaussian / PINN (primitive): 0.0719 / 331.8 / 0.0020; 0.0690 / 297 / 0.0011; 0.0726 / 368.7 / 0.0007
- 4 Gaussian / PINN (conservative): 0.1007 / 709.2 / 0.0191; 0.0809 / 479.1 / 0.0077; 0.0916 / 614.1 / 0.0131
- 4 Gaussian / FVM-PINN (data-free): 1.1893 / 5289 / 0.5967; 1.1257 / 5007 / 0.5342; 1.1935 / 5222 / 0.4962
- 5 Parabolic / PINN (primitive): 0.5702 / 2771 / 0.0036; 0.5638 / 2672 / 0.0128; 0.5592 / 2577 / 0.0064
- 5 Parabolic / PINN (conservative): 0.4872 / 3102 / 0.0536; 0.4929 / 3210 / 0.0711; 0.4729 / 3081 / 0.0737
- 5 Parabolic / FVM-PINN (data-free): 1.7660 / 1.339e+04 / 0.4888; 1.7743 / 1.341e+04 / 0.5256; 1.7816 / 1.364e+04 / 0.3469
- 6 Triangular / PINN (primitive): 0.2957 / 1153 / 0.0056; 0.3048 / 1109 / 0.0047; 0.3112 / 1196 / 0.0031
- 6 Triangular / PINN (conservative): 0.3562 / 2059 / 0.0246; 0.3630 / 1967 / 0.0313; 0.3906 / 2391 / 0.0501
- 6 Triangular / FVM-PINN (data-free): 1.4690 / 8468 / 0.3078; 1.6247 / 8936 / 0.2175; 1.5357 / 9406 / 0.2831
