"""
Stub file for snapy.implicit module
"""

from typing import Callable, Dict, List, Optional, Tuple, overload
import torch


# Implicit Solver
class ImplicitOptions:
    """
    Implicit solver configuration options.

    This class manages implicit time integration parameters.
    """

    def __init__(self) -> None:
        """Initialize ImplicitOptions with default values."""
        ...

    @staticmethod
    def from_yaml(filename: str, verbose: bool = False) -> Optional["ImplicitOptions"]:
        """
        Load ImplicitOptions from a YAML file.

        Args:
            filename: Path to YAML file
            verbose: Enable verbose output

        Returns:
            ImplicitOptions loaded from file, or None when the file sets no
            implicit scheme
        """
        ...

    @overload
    def precision(self) -> str: ...

    @overload
    def precision(self, value: str) -> "ImplicitOptions": ...

    @overload
    def fallback_fp64(self) -> bool: ...

    @overload
    def fallback_fp64(self, value: bool) -> "ImplicitOptions": ...

    def __repr__(self) -> str: ...

    def type(self) -> str:
        """Get the implicit solver type (read-only)."""
        ...

    @overload
    def scheme(self) -> int:
        """Get the scheme type."""
        ...

    @overload
    def scheme(self, value: int) -> "ImplicitOptions":
        """Set the scheme type."""
        ...

    @overload
    def advection_cfl(self) -> float:
        """CFL bound on |v| in an implicit direction (default 1.0). Requires an
        implicit direction; setting it without one is a hard error."""
        ...

    @overload
    def advection_cfl(self, value: float) -> "ImplicitOptions": ...

    @overload
    def shear_cfl(self) -> float:
        """Bound dt at every x1 face whose horizontal wind jumps by at least the
        face sound speed. 0 (the default) turns it off; requires an implicit
        direction."""
        ...

    @overload
    def shear_cfl(self, value: float) -> "ImplicitOptions": ...

class ImplicitHydro:
    """
    Implicit hydrodynamics solver.

    This module handles implicit time integration for hydrodynamics.
    """

    @overload
    def __init__(self) -> None:
        """Construct a new default module."""
        ...

    @overload
    def __init__(self, options: ImplicitOptions, hydro = None) -> None:
        """
        Construct an ImplicitHydro module.

        Args:
            options: Implicit solver configuration options
            hydro: Parent hydro module (optional)
        """
        ...

    def __repr__(self) -> str: ...

    options: ImplicitOptions

    def forward(self, *args) -> torch.Tensor:
        """Forward pass through the module."""
        ...

    def module(self, name: str) -> torch.nn.Module:
        """Get a named sub-module."""
        ...

    def buffer(self, name: str) -> torch.Tensor:
        """Get a named buffer."""
        ...
