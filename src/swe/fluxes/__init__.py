"""Riemann flux kernels, keyed by name for config-driven selection."""

from .hll import hll_flux
from .hllc import hllc_flux
from .rusanov import rusanov_flux

FLUXES = {
    "rusanov": rusanov_flux,
    "hll": hll_flux,
    "hllc": hllc_flux,
}

__all__ = ["FLUXES", "rusanov_flux", "hll_flux", "hllc_flux"]
