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

#: Thin-layer threshold factor: cells with h <= THIN_FACTOR * H_EPS are
#: treated as under-resolved films at wet/dry fronts. There, MUSCL drops to
#: first order and HLLC falls back to HLL (Toro's S* contact estimate
#: degrades when the two-rarefaction depth estimate far exceeds both side
#: depths, flipping the sign of the star momentum flux). Standard wet/dry
#: front practice; the resulting 1e-3 m threshold matches GeoClaw's default
#: dry tolerance. See tests/test_solver.py::test_dry_dam_break_positivity.
THIN_FACTOR: float = 1000.0

#: Solver arithmetic precision. ML models may use float32, but every function
#: in the ``swe`` package computes in float64.
DTYPE: torch.dtype = torch.float64
