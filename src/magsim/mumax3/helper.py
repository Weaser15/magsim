from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

from . import cmd
from .blocks import (
    conditional_save_states,
    hysteresis_block,
    init_simspace,
    save_states,
    tableadd_multiple,
)
from .grid import Grid
from .layer import LayerGroup, layer_dict


def tableadd_common(regions: list[int] | None = None) -> list[str]:
    variables = ["B_ext", "E_demag", "E_exch", "E_Zeeman", "E_anis", "E_total"]
    if regions is not None:
        variables += [cmd.get_var_region("m", i) for i in regions if i is not None]
    return [
        cmd.header_comment("Add Variables to Table"),
        *tableadd_multiple(variables),
        cmd.table_add_var("save_m", "save_m", ""),
        "",
    ]


def create_stack_by_name(
    stack: Iterable[tuple[str, float]], dz: float, l_dict: dict[str, Callable] | None = None
):
    l_dict = layer_dict if l_dict is None else l_dict
    layer_list = [l_dict[s[0].lower()](s[1]) for s in stack]
    layers = LayerGroup(layer_list, dz)
    return layers


def hysteresis(
    layers: LayerGroup,
    grid: Grid,
    B_start: float,
    B_stop: float,
    B_step: float,
    phi: float,
    save_names: tuple[str] = ("m",),
    B_save: float | None = None,
    *,
    save_by_regions: bool = False,
    comp: int | None = None,
    relaxation_mechanism: list[str] | None = None,
    save_bool_name: str = "save_m",
    extra_commands: list[str] | None = None,
):
    if relaxation_mechanism is None:
        relaxation_mechanism = [cmd.minimize()]
    if extra_commands is None:
        extra_commands = []

    commands = []
    commands.append(cmd.timestamp())
    commands += init_simspace(layers, grid)
    commands += tableadd_common(regions=layers.get_regions())

    hyst_commands = []
    hyst_commands += relaxation_mechanism
    # Save magnetisation states (and others, e.g. demag)
    save_layers = layers.get_floor_layer_names() if save_by_regions else None
    if B_save is None:
        hyst_commands += save_states(
            save_names,
            comp=comp,
            save_m=True,
            tablesave=True,
            save_m_name=save_bool_name,
            layers=save_layers,
        )
    else:
        hyst_commands += conditional_save_states(
            save_names,
            B_save,
            comp=comp,
            save_m=True,
            tablesave=True,
            save_m_name=save_bool_name,
            layers=save_layers,
        )
    hyst_commands += extra_commands

    commands += hysteresis_block(B_start, B_stop, B_step, phi, hyst_commands)
    commands.append(cmd.timestamp())
    return commands


def excitation(
    layers: LayerGroup,
    grid: Grid,
    dirpath: Path | str,
    index: int,
    *,
    save_bool_name: str = "save_m",
):
    commands = []
    commands.append(cmd.timestamp())
    commands += init_simspace(
        layers, grid, dirpath=dirpath, index=index, saveboolname=save_bool_name
    )
