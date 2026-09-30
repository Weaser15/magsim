from .grid import Grid
from .layer import LayerGroup, MagneticLayer, RKKYLayer
from .material import MagneticMaterial, empty


def initialise_space(layers, grid, materials, mask):
    # Set up grid
    gd = Grid.from_mask(
        mask,
        grid["nm_per_px"] * 1e-9,
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
        material = layer["material"]
        if material is None:
            material = empty
        if "rkky" in layer:
            lr = RKKYLayer(J1=layer["J1"], J2=layer["J2"], thickness=thickness)
        else:
            lr = MagneticLayer(thickness=thickness, material=material)
        lrs.append(lr)
    lrs = LayerGroup(lrs)
    lrs.assign_dz(dz)

    return lrs, gd
