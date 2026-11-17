"""V3 lane O skeleton: the options generator (signatures only; not wired).

Reads docs/v3/skeleton/O/options_table.yaml (in V3: src/options/*.yaml) and
writes the five generated artifacts listed in that file's header. Runs at
CMake configure time (add_custom_command with the table as DEPENDS) and from
setup.py before the extension build, so the C++ library, the Python module
and the stubs always come from the same table revision.

Ownership: the table is the single source of truth. Hand edits to a
generated file are overwritten; CI fails if a checked-in .pyi differs from
the generated one (the stubs are checked in for IDEs; everything else is a
build product).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class OptionRow:
    """One row of the table (columns documented in options_table.yaml)."""

    module: str
    name: str
    key: Optional[str]
    type: str
    default: object
    env: Optional[str] = None
    python: bool = True
    doc: str = ""


@dataclass(frozen=True)
class ModuleTable:
    module: str
    yaml_node: str
    cpp_struct: str
    options: tuple  # tuple[OptionRow, ...]
    checks: tuple = ()  # cross-option rules, "expr : message"


def load_tables(paths: list[Path]) -> list[ModuleTable]:
    """Parse and validate: unique names and keys per module, defaults that
    parse as their type, enum defaults inside the enum, env only on bool or
    enum options."""
    ...


def emit_cpp_struct(t: ModuleTable) -> str:
    """ADD_ARG lines with the default, report(), clone(); header guard."""
    ...


def emit_cpp_parser(t: ModuleTable) -> str:
    """from_yaml(): check_keys(node, path, {keys}) then one typed read per
    row with the row default; env override with TORCH_WARN_ONCE; checks."""
    ...


def emit_pybind(t: ModuleTable) -> str:
    """ADD_OPTION(T, struct, name) for rows with python=True."""
    ...


def emit_pyi(t: ModuleTable) -> str:
    """Getter/setter overloads with the doc line as docstring."""
    ...


def emit_rst(t: ModuleTable) -> str:
    """list-table: key, type, default, env, doc."""
    ...


def import_from_headers(src: Path) -> list[ModuleTable]:
    """One-time migration: read ADD_ARG lines and from_yaml defaults at the
    pin and produce the initial table; reports every option whose create()
    default and deck default differ (the #352 class of bug)."""
    ...


def main(argv: list[str]) -> int:
    """gen_options.py --tables DIR --out BUILD_DIR [--check-stubs DIR]"""
    ...
