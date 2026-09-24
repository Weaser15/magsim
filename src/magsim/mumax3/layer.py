from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, MutableSequence
from dataclasses import dataclass, replace
from typing import Any

from . import cmd
from .material import MagneticMaterial

NCELLS_LAYER_TEMPLATE = "n_l{}"
FLOOR_LAYER_TEMPLATE = "z_l{}"


@dataclass(frozen=True)
class Layer(ABC):
    thickness: float

    def ncells(self, dz: float) -> int:
        return round(self.thickness / dz)

    @abstractmethod
    def to_script(self, layer_index: int | None = None) -> list[str]: ...


@dataclass(frozen=True)
class MagneticLayer(Layer):
    material: MagneticMaterial
    region: int | None = None

    def to_script(self, layer_index: int | None = None, shape: str | None = None) -> list[str]:
        commands = []
        if self.region is None and layer_index is None:
            # Case 1: No regions defined - only one material.
            commands.append(cmd.header_comment("Define Global Material: {self.material.name}"))
        elif self.region is None or layer_index is None:
            raise ValueError(
                f"Layer is partially resolved - region: {self.region}, layer_index: {layer_index}"
            )
        else:
            # Case 2: Regions defined, so multiple materials.
            z_start = FLOOR_LAYER_TEMPLATE.format(layer_index)
            z_end = f"{z_start} + {NCELLS_LAYER_TEMPLATE.format(layer_index)}"
            layer_shape = cmd.layers(z_start, z_end)
            if shape is not None:
                layer_shape = cmd.intersect(shape, layer_shape)

            commands += [
                cmd.header_comment(
                    f"Define Layer: {layer_index}, Region: {self.region}, Material: {self.material.name}"
                ),
                cmd.define_region(self.region, layer_shape),
            ]
        return [*commands, *self.material.to_script(self.region), ""]


@dataclass(frozen=True)
class RKKYLayer(Layer):
    J1: float
    J2: float

    def to_script(
        self,
        layer_index: int | None = None,
        region_below: int | None = None,
        region_above: int | None = None,
    ) -> list[str]:

        # These exist to match method signature
        if layer_index is None:
            raise ValueError("Need layer index to be defined to generate RKKY Layer!")
        if region_below is None or region_above is None:
            raise ValueError("Both regions need to not be None!")

        commands = []
        # Define string script names
        J1 = f"J1_l{layer_index}"
        J2 = f"J2_l{layer_index}"
        commands += [
            cmd.header_comment(f"Layer: {layer_index}, RKKY Coupling"),
            cmd.define_var(J1, self.J1),
            cmd.define_var(J2, self.J2),
            "",
        ]

        # Interface
        prev_idx = layer_index - 1
        next_idx = layer_index + 1

        prev_ncells = NCELLS_LAYER_TEMPLATE.format(prev_idx)
        next_ncells = NCELLS_LAYER_TEMPLATE.format(next_idx)

        prev_interface = f"interface_dw_l{layer_index}"
        next_interface = f"interface_up_l{layer_index}"

        m_prev_in_next = f"mshift_up_l{layer_index}"
        m_next_in_prev = f"mshift_dw_l{layer_index}"

        mshift = f"mshift_l{layer_index}"

        commands += [
            cmd.comment(f"Layer: {layer_index}, Define RKKY Interface"),
            cmd.define_var(
                next_interface,
                cmd.mul(
                    cmd.shifted(cmd.const(1), 0, 0, f"{prev_ncells} + 1"),
                    cmd.shifted(cmd.const(1), 0, 0, f"-1 * ({next_ncells} - 1)"),
                ),
            ),
            cmd.define_var(
                prev_interface,
                cmd.mul(
                    cmd.shifted(cmd.const(1), 0, 0, f"-1 * ({next_ncells} + 1)"),
                    cmd.shifted(cmd.const(1), 0, 0, f"{prev_ncells} - 1"),
                ),
            ),
            cmd.define_var(
                m_prev_in_next,
                cmd.mul(
                    cmd.shifted("m", 0, 0, f"{prev_ncells} + 1"),
                    next_interface,
                ),
            ),
            cmd.define_var(
                m_next_in_prev,
                cmd.mul(
                    cmd.shifted("m", 0, 0, f"-1 * ({next_ncells} + 1)"),
                    prev_interface,
                ),
            ),
            cmd.define_var(
                mshift,
                cmd.add(m_prev_in_next, m_next_in_prev),
            ),
            "",
        ]

        # Define J1 Field
        C_up_J1 = f"C_up_J1_l{layer_index}"
        C_dw_J1 = f"C_dw_J1_l{layer_index}"
        C_J1 = f"C_J1_l{layer_index}"
        B_J1 = f"B_J1_l{layer_index}"
        E_J1 = f"E_J1_l{layer_index}"

        commands += [
            cmd.comment(f"Layer: {layer_index}: J1 Field"),
            cmd.define_var(
                C_up_J1,
                cmd.mul(
                    cmd.const(f"-{J1} / ({cmd.get_var_region('Msat', region_above)} * dz)"),
                    next_interface,
                ),
            ),
            cmd.define_var(
                C_dw_J1,
                cmd.mul(
                    cmd.const(f"-{J1} / ({cmd.get_var_region('Msat', region_below)} * dz)"),
                    prev_interface,
                ),
            ),
            cmd.define_var(C_J1, cmd.add(C_up_J1, C_dw_J1)),
            cmd.define_var(B_J1, cmd.mul(C_J1, mshift)),
            cmd.define_var(E_J1, cmd.mul(cmd.const(-1), cmd.dot(B_J1, "M_full"))),
            "",
        ]

        # Define J2 Field
        C_up_J2 = f"C_up_J2_l{layer_index}"
        C_dw_J2 = f"C_dw_J2_l{layer_index}"
        C_J2 = f"C_J2_l{layer_index}"
        B_J2 = f"B_J2_l{layer_index}"
        E_J2 = f"E_J2_l{layer_index}"

        commands += [
            cmd.comment(f"Layer: {layer_index}: J2 Field"),
            cmd.define_var(
                C_up_J2,
                cmd.mul(
                    cmd.const(f"-2*{J2} / ({cmd.get_var_region('Msat', region_above)} * dz)"),
                    next_interface,
                ),
            ),
            cmd.define_var(
                C_dw_J2,
                cmd.mul(
                    cmd.const(f"-2*{J2} / ({cmd.get_var_region('Msat', region_below)} * dz)"),
                    prev_interface,
                ),
            ),
            cmd.define_var(C_J2, cmd.add(C_up_J2, C_dw_J2)),
            cmd.define_var(B_J2, cmd.mul(mshift, cmd.mul(C_J2, cmd.dot(mshift, "m")))),
            cmd.define_var(E_J2, cmd.mul(cmd.const(-0.5), cmd.dot(B_J2, "M_full"))),
            "",
        ]

        commands += [
            cmd.comment(f"Layer: {layer_index}: Add Terms"),
            cmd.add_field_term(B_J1),
            cmd.add_edens_term(E_J1),
            cmd.add_field_term(B_J2),
            cmd.add_edens_term(E_J2),
        ]

        return commands


class LayerGroup(MutableSequence):
    def __init__(self, layers: Iterable[Layer] | None = None):
        self._layers = list(layers) if layers is not None else []

    def __getitem__(self, index):
        return self._layers[index]

    def __setitem__(self, index: int | slice, value: Layer | Iterable[Layer]) -> None:
        if isinstance(index, slice):
            values = list(value) if not isinstance(value, Layer) else [value]
            if not all(isinstance(v, Layer) for v in values):
                raise ValueError("LayerGroup can only store Layers!")
            self._layers[index] = values
        else:
            if not isinstance(value, Layer):
                raise ValueError("LayerGroup can only store Layers!")
            self._layers[index] = value

    def __delitem__(self, index):
        del self._layers[index]

    def __len__(self):
        return len(self._layers)

    def __add__(self, other: Layer | LayerGroup) -> LayerGroup:
        if isinstance(other, Layer):
            return LayerGroup([*self, other])
        elif all(isinstance(v, Layer) for v in other):
            return LayerGroup([*self, *other])
        else:
            raise ValueError("Can only add LayerGroup with Layer and LayerGroup!")

    def __iadd__(self, other: Layer | Iterable[Layer] | LayerGroup) -> LayerGroup:
        if isinstance(other, Layer):
            self._layers.append(other)
        elif all(isinstance(v, Layer) for v in other):
            self._layers.extend(other)
        else:
            raise ValueError("LayerGroup can only store Layers!")
        return self

    def __iter__(self):
        return (l for l in self._layers)

    def insert(self, index: int, value: Layer) -> None:
        if not isinstance(value, Layer):
            raise ValueError("LayerGroup can only store Layers!")
        self._layers.insert(index, value)

    def total_thickness(self):
        return sum(layer.thickness for layer in self._layers)

    def round(self, dz: float):
        new_layers = [replace(layer, thickness=layer.ncells(dz) * dz) for layer in self._layers]
        return LayerGroup(new_layers)

    def to_script(self, dz: float, shape: str | None):
        layers = self.round(dz)
        nlayers = len(layers)

        commands = []
        commands.append(cmd.header_comment("Define Layers"))
        ncells = [layer.ncells(dz) for layer in layers]
        n_vars = [NCELLS_LAYER_TEMPLATE.format(i + 1) for i in range(nlayers)]

        commands += [cmd.define_var(n_vars[i], ncells[i]) for i in range(nlayers)]
        commands += [
            cmd.define_var(
                FLOOR_LAYER_TEMPLATE.format(i + 1), " + ".join(n_vars[:i]) if i > 0 else 0
            )
            for i in range(nlayers)
        ]
        commands.append("")

        for i, layer in enumerate(layers):
            if not isinstance(layer, MagneticLayer):
                continue
            commands += layer.to_script(layer_index=i + 1, shape=shape)
            commands.append("")

        for i, layer in enumerate(layers):
            if isinstance(layer, MagneticLayer):
                continue

            kwargs = {}
            kwargs["layer_index"] = i + 1
            if isinstance(layer, RKKYLayer):
                layer_below, layer_above = layers[i - 1], layers[i + 1]
                if not (
                    isinstance(layer_below, MagneticLayer)
                    and isinstance(layer_above, MagneticLayer)
                ):
                    raise ValueError(
                        "RKKY Coupling requires layers on either side to be magnetic!"
                    )
                kwargs["region_below"] = layer_below.region
                kwargs["region_above"] = layer_above.region
            commands += layer.to_script(**kwargs)
            commands.append("")
        return commands

    def get_magnetic_layers_property(self, key: str) -> list[tuple[int, Any]]:
        properties = []
        for layer in self:
            if isinstance(layer, MagneticLayer):
                region = layer.region
                property_ = getattr(layer.material, key)
                properties.append((region, property_))
        return properties
