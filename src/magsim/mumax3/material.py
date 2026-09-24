from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import cmds


def optional_vector(value: tuple | None) -> str | None:
    return cmds.vector(*value) if value is not None else None


@dataclass(frozen=True)
class MagneticMaterial:
    name: str
    Msat: float
    Aex: float
    alpha: float
    GammaLL: float | None = None
    AnisU: tuple[float, float, float] | None = None
    Ku1: float | None = None
    Ku2: float | None = None
    AnisC1: tuple[float, float, float] | None = None
    AnisC2: tuple[float, float, float] | None = None
    Kc1: float | None = None
    Kc2: float | None = None
    Kc3: float | None = None

    def to_script(self, region: int | None) -> list[str]:

        def emit(var: str, value: Any):
            if region is None:
                return cmds.set_var(var, value)
            else:
                return cmds.set_var_region(var, region, value)

        var_value = {
            "Msat": self.Msat,
            "Aex": self.Aex,
            "alpha": self.alpha,
            "GammaLL": self.GammaLL,
            "AnisU": optional_vector(self.AnisU),
            "Ku1": self.Ku1,
            "Ku2": self.Ku2,
            "AnisC1": optional_vector(self.AnisC1),
            "AnisC2": optional_vector(self.AnisC2),
            "Kc1": self.Kc1,
            "Kc2": self.Kc2,
            "Kc3": self.Kc3,
        }
        return [emit(var, value) for var, value in var_value.items() if value is not None]
