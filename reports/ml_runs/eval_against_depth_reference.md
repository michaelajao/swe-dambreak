# Neural models vs reference depth data (CSV_FILE_h)

Reference: MUSCLRS depth files (501x501 nodes) at t = 2.0 s, bilinearly regridded onto the 128^2 evaluation grid. Error: discrete L1(h) integrated over cell areas. Classical row: our HLLC MUSCL-van Leer at the same evaluation resolution (inter-solver difference; cf. reports/reconciliation.md).

| Variant | classical HLLC (ours) | PINN prim. | PINN cons. | FVM-PINN |
|---|---|---|---|---|
| 1 Step | 2.19e+03 | 3e+03 | 3.23e+03 | 1.1e+04 |
| 2 Rectangular | 609 | 790 | 836 | 4.99e+03 |
| 3 Circular | 863 | 1.55e+03 | 2.77e+03 | 1.35e+04 |
| 4 Gaussian | 398 | 332 | 601 | 5.17e+03 |
| 5 Parabolic | 773 | 2.67e+03 | 3.13e+03 | 1.35e+04 |
| 6 Triangular | 584 | 1.15e+03 | 2.14e+03 | 8.94e+03 |

## Per-seed values

- 1 Step / PINN (primitive): 2968, 2959, 3069
- 1 Step / PINN (conservative): 3308, 3194, 3196
- 1 Step / FVM-PINN (physics): 1.112e+04, 1.101e+04, 1.094e+04
- 2 Rectangular / PINN (primitive): 818.4, 742.7, 809.4
- 2 Rectangular / PINN (conservative): 816.1, 851.1, 840.6
- 2 Rectangular / FVM-PINN (physics): 4655, 5093, 5214
- 3 Circular / PINN (primitive): 1548, 1545, 1564
- 3 Circular / PINN (conservative): 2945, 2459, 2893
- 3 Circular / FVM-PINN (physics): 1.319e+04, 1.368e+04, 1.364e+04
- 4 Gaussian / PINN (primitive): 331.8, 297, 368.7
- 4 Gaussian / PINN (conservative): 709.2, 479.1, 614.1
- 4 Gaussian / FVM-PINN (physics): 5289, 5007, 5222
- 5 Parabolic / PINN (primitive): 2771, 2672, 2577
- 5 Parabolic / PINN (conservative): 3102, 3210, 3081
- 5 Parabolic / FVM-PINN (physics): 1.339e+04, 1.341e+04, 1.364e+04
- 6 Triangular / PINN (primitive): 1153, 1109, 1196
- 6 Triangular / PINN (conservative): 2059, 1967, 2391
- 6 Triangular / FVM-PINN (physics): 8468, 8936, 9406
