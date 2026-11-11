"""
Stub file for snapy.forcing module
"""

from typing import Callable, Dict, List, Optional, Tuple, overload
import torch

# Forcing
class ConstGravityOptions:
    """
    Constant gravity forcing options.

    This class manages constant gravity parameters.
    """

    def __init__(self) -> None:
        """Initialize ConstGravityOptions with default values."""
        ...

    def __repr__(self) -> str: ...

    @overload
    def grav1(self) -> float:
        """Get gravity in x1 direction."""
        ...

    @overload
    def grav1(self, value: float) -> "ConstGravityOptions":
        """Set gravity in x1 direction."""
        ...

    @overload
    def grav2(self) -> float:
        """Get gravity in x2 direction."""
        ...

    @overload
    def grav2(self, value: float) -> "ConstGravityOptions":
        """Set gravity in x2 direction."""
        ...

    @overload
    def grav3(self) -> float:
        """Get gravity in x3 direction."""
        ...

    @overload
    def grav3(self, value: float) -> "ConstGravityOptions":
        """Set gravity in x3 direction."""
        ...

    @overload
    def gravity_work(self) -> str:
        """Get the x1 gravity-work form: "face" (default) or "cell"."""
        ...

    @overload
    def gravity_work(self, value: str) -> "ConstGravityOptions":
        """Set the x1 gravity-work form."""
        ...

    @overload
    def gravity_work_fixer(self) -> bool:
        """Get whether the global E+PE fixer runs (with gravity_work "cell")."""
        ...

    @overload
    def gravity_work_fixer(self, value: bool) -> "ConstGravityOptions":
        """Set whether the global E+PE fixer runs (with gravity_work "cell")."""
        ...

class CoriolisOptions:
    """
    Coriolis forcing options.

    This class manages Coriolis force parameters.
    """

    def __init__(self) -> None:
        """Initialize CoriolisOptions with default values."""
        ...

    def __repr__(self) -> str: ...

    @overload
    def omega1(self) -> float:
        """Get rotation rate omega1."""
        ...

    @overload
    def omega1(self, value: float) -> "CoriolisOptions":
        """Set rotation rate omega1."""
        ...

    @overload
    def omega2(self) -> float:
        """Get rotation rate omega2."""
        ...

    @overload
    def omega2(self, value: float) -> "CoriolisOptions":
        """Set rotation rate omega2."""
        ...

    @overload
    def omega3(self) -> float:
        """Get rotation rate omega3."""
        ...

    @overload
    def omega3(self, value: float) -> "CoriolisOptions":
        """Set rotation rate omega3."""
        ...

class DiffusionOptions:
    """Isotropic hydro diffusion options."""

    def __init__(self) -> None: ...

    def __repr__(self) -> str: ...

    @overload
    def temperature_inner(self) -> float: ...
    @overload
    def temperature_inner(self, value: float) -> "DiffusionOptions": ...
    @overload
    def temperature_outer(self) -> float: ...
    @overload
    def temperature_outer(self, value: float) -> "DiffusionOptions": ...
    @overload
    def wall_velocity(self) -> str: ...
    @overload
    def wall_velocity(self, value: str) -> "DiffusionOptions": ...
    @overload
    def implicit_radial(self) -> bool: ...
    @overload
    def implicit_radial(self, value: bool) -> "DiffusionOptions": ...

    @overload
    def nu_iso(self) -> float:
        """Get kinematic viscosity."""
        ...

    @overload
    def nu_iso(self, value: float) -> "DiffusionOptions":
        """Set kinematic viscosity."""
        ...

    @overload
    def kappa_iso(self) -> float:
        """Get thermal diffusivity in units of length squared per time."""
        ...

    @overload
    def kappa_iso(self, value: float) -> "DiffusionOptions":
        """Set thermal diffusivity in units of length squared per time."""
        ...

    @overload
    def dynamic(self) -> bool:
        """Read nu_iso as a dynamic viscosity mu and kappa_iso as a conductivity k."""
        ...

    @overload
    def dynamic(self, value: bool) -> "DiffusionOptions": ...

    @overload
    def nu_scale_x1_table(self) -> torch.Tensor:
        """Get the x1 profile of nu_iso as a (2, n) table of x1 knots and scale values."""
        ...

    @overload
    def nu_scale_x1_table(self, value: torch.Tensor) -> "DiffusionOptions":
        """Set the x1 profile of nu_iso as a (2, n) float64 table: x1 knots
        (strictly increasing, n >= 2) and scale values (strictly positive),
        linear between knots and held beyond the ends. Works with any number
        of blocks along x1. Set before the MeshBlock is built."""
        ...

    @overload
    def kappa_scale_x1_table(self) -> torch.Tensor:
        """Get the x1 profile of kappa_iso as a (2, n) table."""
        ...

    @overload
    def kappa_scale_x1_table(self, value: torch.Tensor) -> "DiffusionOptions":
        """Set the x1 profile of kappa_iso as a (2, n) table, as nu_scale_x1_table."""
        ...

    @overload
    def nu_scale_x1(self) -> torch.Tensor:
        """Get the per-cell x1 profile of nu_iso."""
        ...

    @overload
    def nu_scale_x1(self, value: torch.Tensor) -> "DiffusionOptions":
        """Set the x1 profile of nu_iso per cell: a 1-D float64 tensor with one
        value per x1 cell centre of the block, ghosts included. Refused when
        x1 is split into more than one block (use nu_scale_x1_table)."""
        ...

    @overload
    def kappa_scale_x1(self) -> torch.Tensor:
        """Get the per-cell x1 profile of kappa_iso."""
        ...

    @overload
    def kappa_scale_x1(self, value: torch.Tensor) -> "DiffusionOptions":
        """Set the x1 profile of kappa_iso per cell, as nu_scale_x1."""
        ...
