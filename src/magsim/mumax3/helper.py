from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np

from . import cmd
from .layer import LayerGroup, MagneticLayer


def hysteresis(
    B_start: float, B_stop: float, B_step: float, phi: float, commands: list[str]
) -> list[str]:

    operator = ">=" if B_start - B_stop > 0 else "<="

    B_ext_str = "B_ext"
    B_start_str = "B_start"
    B_stop_str = "B_stop"
    B_step_str = "B_step"
    xangle = "cos(phi)"
    yangle = "sin(phi)"
    var = "B"

    forloop_commands = [
        cmd.set_var(B_ext_str, cmd.vector(f"{var} * {xangle}", f"{var} * {yangle}", 0)),
        *commands,
    ]

    hyst_commands = [
        cmd.header_comment("Start Hysteresis"),
        cmd.define_var(B_start_str, B_start),
        cmd.define_var(B_stop_str, B_stop),
        cmd.define_var(B_step_str, B_step),
        cmd.define_var("phi", f"{phi} * (pi / 180.0)"),
        "",
        cmd.set_var(
            B_ext_str, cmd.vector(f"{B_start_str} * {xangle}", f"{B_start_str} * {yangle}", 0)
        ),
        cmd.relax(),
        "",
        cmd.for_loop(
            var=var,
            start=B_start_str,
            condition=B_stop_str,
            step=B_step_str,
            operator=operator,
            commands=forloop_commands,
        ),
        "",
    ]
    return hyst_commands


def run_relax(
    layers: LayerGroup | MagneticLayer,
    t1: float,
    alpha2: float,
    t2: float,
    relax_func: str = cmd.minimize(),
) -> list[str]:

    if isinstance(layers, LayerGroup):
        old_region_alphas = layers.get_magnetic_layers_property("alpha")
        set_to_old = [
            cmd.set_var_region("alpha", region, alpha) for region, alpha in old_region_alphas
        ]
        set_to_new = [
            cmd.set_var_region("alpha", region, alpha2) for region, _ in old_region_alphas
        ]
    elif isinstance(layers, MagneticLayer):
        set_to_old = [cmd.set_var("alpha", layers.material.alpha)]
        set_to_new = [cmd.set_var("alpha", alpha2)]

    commands = [
        *set_to_old.copy(),
        cmd.run(t1),
        relax_func,
        "",
        *set_to_new.copy(),
        cmd.run(t2),
        "",
        *set_to_old.copy(),
    ]
    return commands


def conditional_run_relax(
    layers: LayerGroup | MagneticLayer,
    t1: float,
    alpha2: float,
    t2: float,
    B_run: float,
    relax_func: str = cmd.minimize(),
) -> list[str]:

    if_commands = run_relax(layers=layers, t1=t1, alpha2=alpha2, t2=t2, relax_func=relax_func)
    return [
        cmd.if_statement(f"abs(B) <= {B_run}", commands=if_commands, else_commands=[relax_func])
    ]


def save_states(
    names: Iterable[str], save_m: bool = True, tablesave: bool = True, save_m_name: str = "save_m"
):
    commands = [cmd.save(name) for name in names]
    if tablesave:
        commands.append(cmd.table_save())
    if save_m:
        commands = [cmd.set_var(save_m_name, 1), *commands, cmd.set_var(save_m_name, 0)]
    return commands


def conditional_save_states(
    names: Iterable[str],
    B_save: float,
    save_m: bool = True,
    tablesave: bool = True,
    save_m_name: str = "save_m",
):
    if_commands = save_states(
        names=names, save_m=save_m, tablesave=tablesave, save_m_name=save_m_name
    )
    return [
        cmd.if_statement(
            f"abs(B) <= {B_save}", commands=if_commands, else_commands=[cmd.table_save()]
        )
    ]


def define_excitation(
    fc: float,
    t0: float,
    amp: float,
    kc: float | None = None,
    xpulse: int = 0,
    ypulse: int = 0,
) -> list[str]:
    commands = []
    commands += [
        cmd.define_var("fc", fc),
        cmd.define_var("t0", t0),
        cmd.define_var("t_step", "1. / (2 * fc)"),
        cmd.define_var("t_pulse", "t0 * 0.1"),
        cmd.define_var("amp", amp),
        "",
    ]

    if kc is not None:
        commands += [
            cmd.define_var("kc", kc),
            cmd.define_var("xpulse", xpulse),
            cmd.define_var("ypulse", ypulse),
            cmd.define_var("Bmask", 0.999),
            cmd.define_var("field_mask", cmd.new_vector_mask("nx", "ny", "nz")),
            "",
        ]
    return commands


def run_excitation(
    use_kc: bool = False,
    save_var: str = "m",
    save_m: bool = True,
) -> list[str]:

    commands = []
    if use_kc:
        for_commands = [
            cmd.define_var("r", cmd.index_to_coord("i", "j", 0)),
            cmd.define_var("x", "r.X()"),
            cmd.define_var("y", "r.Y()"),
            cmd.set_var(
                "Bmask",
                cmd.sinc(
                    f"kc * {cmd.sqrt('(x - xpulse) * (x - xpulse) + (y - ypulse) * (y - ypulse)')}"
                ),
            ),
            cmd.set_vector("field_mask", "i", "j", 0, cmd.vector(0, 0, "Bmask")),
        ]
        for_loop2 = cmd.for_loop(
            "j", start=0, condition="ny", step=1, operator="<", commands=for_commands
        )
        for_loop1 = cmd.for_loop(
            "i", start=0, condition="nx", step=1, operator="<", commands=[for_loop2]
        )
        commands += [
            for_loop1,
            "B_ext." + cmd.add("field_mask", f"amp * {cmd.sinc('2 * pi * fc * (t - t_pulse)')}"),
        ]
    else:
        commands += [cmd.set_var("B_ext", f"amp * {cmd.sinc('2 * pi * fc * (t - t_pulse)')}")]

    run_commands = [cmd.table_autosave("t_step"), cmd.autosave(save_var, "t_step"), cmd.run("t0")]
    if save_m:
        run_commands = [cmd.set_var("save_m", 1), *run_commands, cmd.set_var("save_m", 0)]
    commands += run_commands
    return commands


def excitation(
    fc: float,
    t0: float,
    amp: float,
    kc: float | None = None,
    xpulse: int = 0,
    ypulse: int = 0,
    save_var: str = "m",
    save_m: bool = True,
) -> list[str]:
    commands = []

    commands += define_excitation(fc=fc, t0=t0, amp=amp, kc=kc, xpulse=xpulse, ypulse=ypulse)
    commands += run_excitation(use_kc=kc is not None, save_var=save_var, save_m=save_m)

    return commands


def load_magstate(
    dirpath: Path | str, index: int, var: str = "B_bias", saveboolname: str = "save_m ()"
) -> list[str]:
    """Load the magstate with correct field."""
    dirpath = Path(dirpath)
    tablepath = dirpath / "table.txt"
    table = np.loadtxt(tablepath, unpack=True)

    with tablepath.open("r") as f:
        cols = f.readline().split("\t")
    field_indices = tuple(cols.index(f"B_ext{i} (T)") for i in "xyz")
    if saveboolname not in cols:
        actual_index = index
    else:
        arg = cols.index(saveboolname)
        boolcol = table[arg]
        actual_index = np.where(boolcol)[0][index]
    field = (float(i) for i in table[field_indices, actual_index])

    filepath = dirpath / f"m{index:06d}.ovf"
    commands = [
        cmd.header_comment("Load Magnetisation from File"),
        cmd.set_var("m", cmd.load_file(filepath.absolute())),
        cmd.define_var(var, cmd.vector(*field)),
        cmd.set_var("B_ext", var),
        "",
    ]
    return commands
