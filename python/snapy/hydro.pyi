"""
Stub file for snapy.hydro module
"""

from typing import Callable, Dict, List, Optional, Tuple, overload
import torch
from .boundary import InternalBoundaryOptions
from .coordinate import CoordinateOptions
from .eos import EquationOfStateOptions
from .forcing import ConstGravityOptions, CoriolisOptions, DiffusionOptions
from .implicit import ImplicitOptions
from .reconstruction import ReconstructOptions
from .riemann import RiemannSolverOptions


# Type aliases
bcfunc_t = Optional[Callable[[torch.Tensor, int, "BoundaryFuncOptions"], None]]

class HydroOptions:
    """
    Hydrodynamics configuration options.

    This class manages hydrodynamics parameters.
    """

    def __init__(self) -> None:
        """Initialize HydroOptions with default values."""
        ...

    def __repr__(self) -> str: ...

    @staticmethod
    def from_yaml(filename: str, verbose: bool = False) -> "HydroOptions":
        """
        Load HydroOptions from a YAML file.

        Args:
            filename: Path to YAML file
            verbose: Enable verbose output (optional)

        Returns:
            HydroOptions loaded from file
        """
        ...

    @overload
    def verbose(self) -> bool:
        """Get verbose flag."""
        ...

    @overload
    def verbose(self, value: bool) -> "HydroOptions":
        """Set verbose flag."""
        ...

    @overload
    def disable_flux_x1(self) -> bool:
        """Get disable flux x1 flag."""
        ...

    @overload
    def disable_flux_x1(self, value: bool) -> "HydroOptions":
        """Set disable flux x1 flag."""
        ...

    @overload
    def disable_flux_x2(self) -> bool:
        """Get disable flux x2 flag."""
        ...

    @overload
    def disable_flux_x2(self, value: bool) -> "HydroOptions":
        """Set disable flux x2 flag."""
        ...

    @overload
    def disable_flux_x3(self) -> bool:
        """Get disable flux x3 flag."""
        ...

    @overload
    def disable_flux_x3(self, value: bool) -> "HydroOptions":
        """Set disable flux x3 flag."""
        ...

    @overload
    def grav(self) -> ConstGravityOptions:
        """Get gravity options."""
        ...

    @overload
    def grav(self, value: ConstGravityOptions) -> "HydroOptions":
        """Set gravity options."""
        ...

    @overload
    def coriolis(self) -> CoriolisOptions:
        """Get Coriolis options."""
        ...

    @overload
    def coriolis(self, value: CoriolisOptions) -> "HydroOptions":
        """Set Coriolis options."""
        ...

    @overload
    def diffusion(self) -> DiffusionOptions:
        """Get hydro diffusion options."""
        ...

    @overload
    def diffusion(self, value: DiffusionOptions) -> "HydroOptions":
        """Set hydro diffusion options."""
        ...

    @overload
    def eos(self) -> "EquationOfStateOptions":
        """Get equation of state options."""
        ...

    @overload
    def eos(self, value: "EquationOfStateOptions") -> "HydroOptions":
        """Set equation of state options."""
        ...

    @overload
    def recon1(self) -> "ReconstructOptions":
        """Get reconstruction options for dimension 1."""
        ...

    @overload
    def recon1(self, value: "ReconstructOptions") -> "HydroOptions":
        """Set reconstruction options for dimension 1."""
        ...

    @overload
    def recon23(self) -> "ReconstructOptions":
        """Get reconstruction options for dimensions 2 and 3."""
        ...

    @overload
    def recon23(self, value: "ReconstructOptions") -> "HydroOptions":
        """Set reconstruction options for dimensions 2 and 3."""
        ...

    @overload
    def riemann(self) -> "RiemannSolverOptions":
        """Get Riemann solver options."""
        ...

    @overload
    def riemann(self, value: "RiemannSolverOptions") -> "HydroOptions":
        """Set Riemann solver options."""
        ...

    @overload
    def icorr(self) -> "ImplicitOptions":
        """Get implicit correction options."""
        ...

    @overload
    def icorr(self, value: "ImplicitOptions") -> "HydroOptions":
        """Set implicit correction options."""
        ...

    @overload
    def wb_wall_clamp(self) -> bool:
        """Clamp the well-balanced x1 reference stencils to interior cells at a
        physical wall. Default True; `dynamics: {wb-wall-clamp: false}` opts out."""
        ...

    @overload
    def wb_wall_clamp(self, value: bool) -> "HydroOptions": ...

    @overload
    def conversion_order(self) -> int:
        """Order of the x1 cell-average <-> point conversion: 2 (default) or 4.

        4 is fourth order on uniform and smoothly stretched x1 grids and falls to second order where adjacent
        cell widths jump."""
        ...

    @overload
    def conversion_order(self, value: int) -> "HydroOptions": ...

class Hydro:
    """
    Hydrodynamics implementation.

    This module handles hydrodynamic calculations.
    """

    @overload
    def __init__(self) -> None:
        """Construct a new default module."""
        ...

    @overload
    def __init__(self, options: HydroOptions, block = None) -> None:
        """
        Construct a Hydro module.

        Args:
            options: Hydrodynamics configuration options
            block: Parent block module (optional)
        """
        ...

    def __repr__(self) -> str: ...

    options: HydroOptions

    def forward(self, *args) -> torch.Tensor:
        """Forward pass through the module."""
        ...

    def module(self, name: str) -> torch.nn.Module:
        """Get a named sub-module."""
        ...

    def buffer(self, name: str) -> torch.Tensor:
        """Get a named buffer."""
        ...

    def max_time_step(self, *args) -> float:
        """Calculate maximum stable time step."""
        ...

def balance_column(w: torch.Tensor, dx1f: torch.Tensor, grav: float,
                   wall_clamp: bool = True, rtol: float = 1.e-10,
                   max_iter: int = 120, geometry: str = "",
                   stretched: bool = False, conversion_order: int = 2,
                   eos: Optional[torch.nn.Module] = None
                   ) -> Tuple[torch.Tensor, float, int]: ...
