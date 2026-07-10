"""Scientific-ML entries: strong-form PINNs and the FVM-informed PINN."""

from .models import FVMPINN, FVMPINNConfig, FVMResidualSpec, PINN, PINNConfig
from .train import TrainConfig, train

__all__ = [
    "PINN", "PINNConfig",
    "FVMPINN", "FVMPINNConfig", "FVMResidualSpec",
    "TrainConfig", "train",
]
