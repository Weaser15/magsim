from __future__ import annotations

from collections.abc import Iterable
from itertools import product
from pathlib import Path

import numpy as np

from . import cmd
from .grid import Grid
from .layer import LayerGroup, MagneticLayer


def tableadd_multiple(variables: Iterable[str]) -> list[str]:
    return [cmd.table_add(var) for var in variables]


def init_simspace(
    layers: LayerGroup,
    grid: Grid,
    *,
    dirpath: Path | str | None = None,
    index: int | None = None,
    var: str = "B_bias",
    saveboolname: str = "save_m ()",
    absolute: bool = True,
) -> list[str]:
    commands = []
    commands += grid.to_script(*layers.get_nz_and_dz())
    shape = "mask" if grid.mask else None
    commands += layers.to_script(shape=shape)
    if (dirpath is not None) and (index is not None):
        commands += load_magstate(dirpath, index, var, saveboolname, absolute)
    elif (dirpath is None) and (index is None):
        pass
    else:
        print("Both dirpath and index must be defined to initialise from magstate!")
    return commands


def hysteresis_block(
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
        cmd.header_comment("Define Hysteresis Parameters"),
        cmd.define_var(B_start_str, B_start),
        cmd.define_var(B_stop_str, B_stop),
        cmd.define_var(B_step_str, B_step),
        cmd.define_var("phi", f"{phi} * (pi / 180.0)"),
        "",
        cmd.header_comment("Relax to Initial State"),
        cmd.set_var(
            B_ext_str, cmd.vector(f"{B_start_str} * {xangle}", f"{B_start_str} * {yangle}", 0)
        ),
        cmd.relax(),
        "",
        cmd.header_comment("Begin Hysteresis"),
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
    names: Iterable[str],
    save_m: bool = True,
    tablesave: bool = True,
    save_m_name: str = "save_m",
    autosave_interval: str | float | None = None,
    comp: int | None = None,
    layers: int | str | list | None = None,
) -> list[str]:

    if isinstance(layers, int | str | None):
        layers = [layers]

    commands = []
    for name, layer in product(names, layers):
        if autosave_interval is None:
            commands.append(cmd.save(cmd.crop_layers(cmd.comp(name, comp), layer)))
        else:
            commands.append(
                cmd.autosave(cmd.crop_layers(cmd.comp(name, comp), layer), autosave_interval)
            )
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
    comp: int | None = None,
    layers: int | str | list | None = None,
) -> list[str]:
    if_commands = save_states(
        names=names,
        save_m=save_m,
        tablesave=tablesave,
        save_m_name=save_m_name,
        comp=comp,
        layers=layers,
    )
    return [
        cmd.if_statement(
            f"abs(B) <= {B_save}", commands=if_commands, else_commands=[cmd.table_save()]
        )
    ]


def define_excitation(
    fc: float,
    t_run: float,
    t0: float,
    amp: float,
    direction: tuple[float, float, float] = (0, 0, 1.0),
    maxerr: float = 1e-7,
    maxdt: float | str = "t_step / 10",
    kc: float | None = None,
    xpulse: int = 0,
    ypulse: int = 0,
) -> list[str]:
    commands = [cmd.header_comment("Define Excitation Parameters")]
    commands += [
        cmd.define_var("fc", fc),
        cmd.define_var("t_run", t_run),
        cmd.define_var("t_step", "1. / (2 * fc)"),
        cmd.define_var("t0", t0),
        cmd.define_var("amp", amp),
        cmd.define_var("direction", cmd.vector(*direction)),
        cmd.set_var("MaxDt", maxdt),
        cmd.set_var("MaxErr", maxerr),
        "",
    ]

    if kc is not None:
        commands += [
            cmd.header_comment("Define Spatial Mask Parameters"),
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
    save_vars: Iterable[str] = ("m",),
    save_m: bool = True,
    comp: int | None = 2,
    layers: int | str | list | None = None,
) -> list[str]:

    commands = [cmd.header_comment("Run Excitation")]
    sinc_pulse = cmd.sinc("2 * pi * fc * (t - t0)")
    if use_kc:
        set_mask = cmd.set_vector("field_mask", "i", "j", "k2", cmd.vector(0, 0, "Bmask"))
        for_loop3 = cmd.for_loop(
            "k", start=0, condition="nz", step=1, operator="<", commands=[set_mask]
        )

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
            for_loop3,
        ]

        for_loop2 = cmd.for_loop(
            "j", start=0, condition="ny", step=1, operator="<", commands=for_commands
        )
        for_loop1 = cmd.for_loop(
            "i", start=0, condition="nx", step=1, operator="<", commands=[for_loop2]
        )
        commands += [
            for_loop1,
            "B_ext." + cmd.add("field_mask", f"amp * {sinc_pulse}"),
        ]
    else:
        commands += [
            cmd.set_var(
                "B_ext",
                cmd.var_add("B_bias", cmd.var_mul(cmd.var_mul("direction", "amp"), sinc_pulse)),
            )
        ]

    run_commands = [
        cmd.table_autosave("t_step"),
        *save_states(
            save_vars,
            save_m=False,
            tablesave=False,
            autosave_interval="t_step",
            comp=comp,
            layers=layers,
        ),
        cmd.run("t_run - t_step"),
    ]
    if save_m:
        run_commands = [cmd.set_var("save_m", 1), *run_commands, cmd.set_var("save_m", 0)]
    commands += run_commands
    return commands


def excitation_block(
    fc: float,
    t_run: float,
    amp: float,
    t0: float,
    maxerr: float = 1e-7,
    maxdt: float | str = "t_step / 10",
    kc: float | None = None,
    xpulse: int = 0,
    ypulse: int = 0,
    save_vars: Iterable[str] = ("m",),
    comp: int | None = 2,
    layers: int | str | list | None = None,
    save_m: bool = True,
) -> list[str]:
    commands = []

    commands += define_excitation(
        fc=fc,
        t_run=t_run,
        amp=amp,
        t0=t0,
        maxerr=maxerr,
        maxdt=maxdt,
        kc=kc,
        xpulse=xpulse,
        ypulse=ypulse,
    )
    commands += run_excitation(
        use_kc=kc is not None, save_vars=save_vars, save_m=save_m, comp=comp, layers=layers
    )

    return commands


def load_magstate(
    dirpath: Path | str,
    index: int,
    var: str = "B_bias",
    saveboolname: str = "save_m ()",
    absolute: bool = True,
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
        cmd.m_load_file(filepath, absolute),
        cmd.define_var(var, cmd.vector(*field)),
        cmd.set_var("B_ext", var),
        cmd.relax(),
        "",
    ]
    return commands
