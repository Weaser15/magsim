from __future__ import annotations

from collections.abc import Collection
from pathlib import Path
from typing import Any

INDENT = "    "

# --- Utility functions -------------------------------------------------------


def add_indent(text: str, level: int = 1) -> str:
    """Add indentation to a single string."""
    prefix = INDENT * level
    return "\n".join(prefix + l for l in text.splitlines())


def set_indent(text: str, level: int) -> str:
    """Set indentation level of a single string."""
    prefix = INDENT * level
    return "\n".join(prefix + l.lstrip() for l in text.splitlines())


# --- Templates ---------------------------------------------------------------


def fmt(value: Any) -> str:
    return str(value)


def define_var(var: str, value: Any):
    return f"{var} := {value}"


def set_var(var: str, value: Any):
    return f"{var} = {value}"


def call(fn: str, *args):
    return f"{fn}({', '.join(fmt(a) for a in args)})"


def var_call(var: str, fn: str, *args):
    return f"{var}.{call(fn, *args)}"


# --- Mumax3 ------------------------------------------------------------------


def define_region(region: int, value: Any) -> str:
    return call("DefRegion", region, value)


def set_var_region(var: str, region: int, value: Any) -> str:
    return var_call(var, "SetRegion", region, value)


def get_var_region(var: str, region: int) -> str:
    return var_call(var, "GetRegion", region)


def table_add(var: str) -> str:
    return call("TableAdd", var)


def table_add_var(var: str, name: str, unit: str) -> str:
    return call("TableAddVar", f'"{name}"', f'"{unit}"')


def add_field_term(var: str) -> str:
    return call("AddFieldTerm", var)


def add_edens_term(var: str) -> str:
    return call("AddEdensTerm", var)


def set_grid_size(nx: int | str, ny: int | str, nz: int | str) -> str:
    return call("SetGridSize", nx, ny, nz)


def set_cell_size(dx: float | str, dy: float | str, dz: float | str) -> str:
    return call("SetCellSize", dx, dy, dz)


def set_pbc(nx: int, ny: int, nz: int) -> str:
    return call("SetPBC", nx, ny, nz)


def load_mask(filepath: str | Path, absolute: bool = False) -> str:
    resolved = filepath if absolute else Path(filepath).name
    return call("ImageShape", f'"{resolved}"')


def load_file(filepath: str | Path, absolute: bool = False) -> str:
    resolved = filepath if absolute else Path(filepath).name
    return call("LoadFile", f'"{resolved}"')


def set_geom(value: Any) -> str:
    return call("SetGeom", value)


def autosave(var: str, interval: float | str) -> str:
    return call("AutoSave", var, interval)


def table_autosave(value: float | str) -> str:
    return call("TableAutoSave", value)


def print_values(*values) -> str:
    return call("Print", *values)


def comment(text: str) -> str:
    return f"// {text}"


def header_comment(text: str, total_length: int = 80, fill_symbol: str = "-"):
    nfill = total_length - len(text) - 2 - 3
    new_text = " ".join([fill_symbol * 3, text, fill_symbol * nfill])
    return comment(new_text)


def save(var: str) -> str:
    return call("Save", var)


def table_save() -> str:
    return call("TableSave")


def relax() -> str:
    return call("Relax")


def minimize() -> str:
    return call("Minimize")


def run(time: float | str) -> str:
    return call("Run", time)


def set_vector(var: str, x: int | str, y: int | str, z: int | str, value: Any) -> str:
    return var_call(var, "SetVector", x, y, z, value)


def vector(x: Any, y: Any, z: Any) -> str:
    return call("Vector", x, y, z)


def snapshot(var: str) -> str:
    return call("Snapshot", var)


def intersect(var1: str, var2: Any) -> str:
    return var_call(var1, "Intersect", var2)


def layers(z1: int | str, z2: int | str) -> str:
    return call("Layers", z1, z2)


def add(var1: Any, var2: Any) -> str:
    return call("Add", var1, var2)


def mul(var1: Any, var2: Any) -> str:
    return call("Mul", var1, var2)


def shifted(var: str, x: int | str, y: int | str, z: int | str) -> str:
    return call("Shifted", var, x, y, z)


def const(value: float | str) -> str:
    return call("Const", value)


def dot(var1: str, var2: str) -> str:
    return call("Dot", var1, var2)


def new_vector_mask(nx: int | str, ny: int | str, nz: int | str):
    return call("NewVectorMask", nx, ny, nz)


def index_to_coord(x: int | str, y: int | str, z: int | str):
    return call("Index2Coord", x, y, z)


def sinc(value: Any):
    return call("Sinc", value)


def sqrt(value: Any):
    return call("Sqrt", value)


def timestamp(message: str = "Time:"):
    return print_values(message, call("Now"))


def crop(
    var: Any,
    x1: int | str,
    x2: int | str,
    y1: int | str,
    y2: int | str,
    z1: int | str,
    z2: int | str,
):
    return call("Crop", var, x1, x2, y1, y2, z1, z2)


# --- Statements --------------------------------------------------------------


def for_loop(
    var: str,
    start: Any,
    condition: Any,
    step: Any,
    commands: Collection[str],
    operator: str | None = None,
) -> str:
    if operator is not None:
        condition = f"{var} {operator} {condition}"

    header = f"for {var} := {start}; {condition}; {var} += {step}" + " {\n"
    body = "\n".join(add_indent(command.rstrip()) for command in commands)
    result = header + body + "\n}"
    return result + "\n"


def if_statement(
    condition: str,
    commands: Collection[str],
    else_commands: Collection[str] | None = None,
) -> str:
    header = f"if {condition}" + " {\n"
    body = "\n".join(add_indent(command.rstrip()) for command in commands)
    result = header + body + "\n}"

    if else_commands is not None:
        else_body = "\n".join(add_indent(command.rstrip()) for command in else_commands)
        result += " else {\n" + else_body + "\n}"

    return result + "\n"
