"""Scientific-ML entries: strong-form PINNs and the FVM-informed PINN."""

from .fvm_pinn import FVMPINN, FVMPINNConfig, FVMResidualSpec
from .pinn import PINN, PINNConfig
from .train import TrainConfig, train

__all__ = [
    "PINN", "PINNConfig",
    "FVMPINN", "FVMPINNConfig", "FVMResidualSpec",
    "TrainConfig", "train",
]
