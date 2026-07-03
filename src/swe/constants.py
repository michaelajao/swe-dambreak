"""Physical and numerical constants shared across the solver.

Do not override these ad hoc; experiment-level overrides go through configs.
"""

import torch

#: Gravitational acceleration [m s^-2].
G: float = 9.81

#: Wet/dry depth threshold [m]. Cells with h <= H_EPS are treated as dry:
#: velocities are zeroed and no momentum flux is admitted.
H_EPS: float = 1.0e-6

#: Default CFL number for adaptive time stepping (SSP-RK2 with first- or
#: second-order reconstruction; cf. Kurganov & Petrova 2007 guidance).
CFL_DEFAULT: float = 0.45

#: Solver arithmetic precision. ML models may use float32, but every function
#: in the ``swe`` package computes in float64.
DTYPE: torch.dtype = torch.float64
