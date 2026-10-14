import numpy as np

from .grid import Grid
from .layer import LayerGroup, MagneticLayer, RKKYLayer
from .material import MagneticMaterial, empty


def convert_iterable(var: dict | list) -> list:
    # Convert dict to a list
    if isinstance(var, dict):
        start, stop, step = var.get("start"), var.get("stop"), var.get("step")
        if not isinstance(start, int | float):
            raise ValueError("`start` not defined in iterable!")
        return np.round(np.arange(start, stop, step), 9).tolist()
    # Else return the list
    return list(var)


def initialise_space(layers, grid, materials, mask):
    # Set up grid
    gd = Grid.from_mask(
        mask,
        grid["px_size_nm"] * 1e-9,
        grid["dx_nm"] * 1e-9,
        grid["dy_nm"] * 1e-9,
        tuple(grid["pbc"]),
        8,
    )
    dz = grid["dz_nm"] * 1e-9

    # Set up materials
    mats = {}
    for key, mat in materials.items():
        mat = mat.copy()
        mat_type = mat.pop("type")
        if mat_type.lower() == "magnetic":
            material = MagneticMaterial(**mat)
        else:
            raise NotImplementedError("Only supports None or `magnetic`!")
        mats[key] = material

    # Set up layers
    lrs = []
    for layer in layers:
        thickness = layer["thickness_nm"] * 1e-9
        material = empty if layer["material"] is None else mats[layer["material"]]
        if "rkky" in layer:
            lr = RKKYLayer(J1=layer["J1"], J2=layer["J2"], thickness=thickness)
        else:
            lr = MagneticLayer(thickness=thickness, material=material)
        lrs.append(lr)
    lrs = LayerGroup(lrs)
    lrs.assign_dz(dz)

    return lrs, gd
