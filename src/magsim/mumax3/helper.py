from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

from . import cmd
from .blocks import (
    conditional_save_states,
    hysteresis_block,
    init_simspace,
    save_states,
)
from .grid import Grid
from .layer import LayerGroup, layer_dict


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

    hyst_commands = []
    hyst_commands += relaxation_mechanism
    # Save magnetisation states (and others, e.g. demag)
    if B_save is None:
        hyst_commands += save_states(save_names, True, True, save_bool_name)
    else:
        hyst_commands += conditional_save_states(save_names, B_save, True, True, save_bool_name)
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
