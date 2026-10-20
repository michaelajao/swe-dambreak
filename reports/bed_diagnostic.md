# Bed-response diagnostic for the reference drop

Depth RMSE [m] at t = 2 s between the stored solutions (regridded) and our well-balanced HLLC MUSCL-van Leer solver on 500x500 cells, run over the hump and over a flat bed with the same initial depth. 'ours hump vs flat' is the size of the bed effect itself.

| Variant | ours hump vs flat | HLL vs ours hump | HLL vs ours flat | MUSCL-RS vs ours hump | MUSCL-RS vs ours flat |
|---|---|---|---|---|---|
| 1 Step | 0.105 | 0.605 | 0.632 | 0.621 | 0.650 |
| 2 Rectangular | 0.094 | 0.116 | 0.083 | 0.108 | 0.068 |
| 3 Circular | 0.169 | 0.093 | 0.200 | 0.208 | 0.127 |
| 4 Gaussian | 0.090 | 0.109 | 0.067 | 0.112 | 0.068 |
| 5 Parabolic | 0.112 | 0.148 | 0.178 | 0.143 | 0.109 |
| 6 Triangular | 0.128 | 0.090 | 0.144 | 0.148 | 0.106 |
