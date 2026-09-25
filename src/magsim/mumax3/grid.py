from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from . import cmd


@dataclass(frozen=True)
class Grid:
    nx: int | str
    ny: int | str
    dx: float | str
    dy: float | str
    pbc: tuple[int, int, int] | str
    edgesmooth: int
    mask: str | Path | None = None

    @classmethod
    def from_mask(
        cls,
        filepath: Path | str,
        resolution: float,
        dx: float,
        dy: float,
        pbc: tuple[int, int, int],
        edgesmooth: int,
    ) -> Grid:
        filepath = Path(filepath)
        image_shape = np.asarray(Image.open(filepath)).shape
        nx, ny = round(image_shape[1] * resolution / dx), round(image_shape[0] * resolution / dy)
        return cls(nx=nx, ny=ny, dx=dx, dy=dy, mask=filepath, pbc=pbc, edgesmooth=edgesmooth)

    def to_script(self, nz: int, dz: float) -> list[str]:
        commands = [
            cmd.header_comment("Define Grid Geometry"),
            cmd.comment("Define Number of Cells"),
            cmd.define_var("nx", self.nx),
            cmd.define_var("ny", self.ny),
            cmd.define_var("nz", nz),
            cmd.set_grid_size("nx", "ny", "nz"),
            "",
            cmd.header_comment("Define Cell Sizes"),
            cmd.define_var("dx", self.dx),
            cmd.define_var("dy", self.dy),
            cmd.define_var("dz", dz),
            cmd.set_cell_size("dx", "dy", "dz"),
            "",
        ]
        # Add the mask if it is included.
        if (mask := self.mask) is not None:
            commands += [
                cmd.header_comment("Define Mask"),
                cmd.define_var("mask", cmd.load_mask(mask, absolute=False)),
                cmd.set_geom("mask"),
                "",
            ]
        return commands
