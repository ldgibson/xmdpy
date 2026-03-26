from collections.abc import Callable, Container, Generator
from dataclasses import dataclass, field
from enum import StrEnum
from functools import partial
import itertools
import shlex
from typing import Any, cast
import warnings

import numpy as np

from xmdpy.types import Int1DArray, PathLike, SingleDType, TrajArray, TrajNDArray

from .names import DATA_VAR_DIMS, DEFAULT_COORDS, Coord, DataVar
from .on_disk_array import OnDiskArray
from .parsing_utils import count_lines, frame_generator
from .trajectory_formats import TrajectoryFormat


def cell_formatter(name: str, value: str) -> tuple[str, TrajNDArray]:
    cell = np.array(value.split(), dtype=float)
    if cell.size == 9:
        cell = cell.reshape(3, 3)
    elif cell.size == 3:
        cell = np.eye(3, dtype=float) * cell
    else:
        raise ValueError(f"Invalid number of cell parameters in '{name}=\"{value}\"'")
    return "cell", cell


def int_formatter(name: str, value: str) -> tuple[str, int]:
    return name, int(value)


def float_formatter(name: str, value: str) -> tuple[str, float]:
    return name, float(value)


VALID_PROPERTY_TYPES: dict[str, type] = {"S": str, "I": int, "R": float, "L": bool}


def properties_formatter(
    name: str, value: str
) -> tuple[str, list[tuple[str, type, int]]]:
    prop_info: list[tuple[str, type, int]] = []
    prop_data = value.split(":")
    for prop_name, prop_type, n_fields in itertools.batched(prop_data, 3):
        prop_info.append(
            (prop_name.lower(), VALID_PROPERTY_TYPES[prop_type], int(n_fields))
        )
    return "properties", prop_info


def pbc_formatter(name: str, value: str) -> tuple[str, tuple[bool, ...]]:
    return name, tuple(dim == "T" for dim in value.split())


def unknown_formatter(name: str, value: str) -> tuple[str, str]:
    warnings.warn(f"Unknown entry: {name}, skipping formatting...")
    return name, value


EXT_DATA_FORMATTERS: dict[str, Callable[[str, str], tuple[str, Any]]] = {
    "time": float_formatter,
    "timestep": int_formatter,
    "lattice": cell_formatter,
    "temperature": float_formatter,
    "potential_energy": float_formatter,
    "properties": properties_formatter,
    "pbc": pbc_formatter,
}


def format_ext_data(name: str, value: str) -> tuple[str, Any]:
    formatter_fn = EXT_DATA_FORMATTERS.get(name.lower(), unknown_formatter)
    return formatter_fn(name, value)


def extxyz_comment_splitter(line: str) -> Generator[list[str]]:
    for entry in shlex.split(line):
        yield entry.split("=", maxsplit=1)


def parse_extxyz_comment(
    line: str, ignore: Container[str] | None = None
) -> dict[str, Any]:
    if ignore is None:
        ignore = set()

    ext_data: dict[str, Any] = {}
    for name, value in extxyz_comment_splitter(line):
        if name in ignore:
            continue

        name, value = format_ext_data(name, value)
        ext_data[name] = value

    return ext_data


def get_extxyz_dims_and_details(
    filename: PathLike,
) -> tuple[int, list[str], dict[str, Any]]:
    n_lines = count_lines(filename)

    atoms = []

    with open(filename, "rb") as f:
        n_atoms = int(f.readline().strip())

        comment = f.readline().decode()
        ext_data = parse_extxyz_comment(comment)

        for _ in range(n_atoms):
            line = f.readline()
            fields = line.split()
            atoms.append(fields[0].decode())

    n_frames = int(n_lines / (n_atoms + 2))

    return n_frames, atoms, ext_data


def read_extxyz_frames(
    frames: Int1DArray,
    atoms: Int1DArray,
    xyz_dim: Int1DArray,
    filename: PathLike,
    usecol: slice,
    total_atoms: int,
    dtype: SingleDType = "float64",
) -> TrajArray:
    offset = 2
    lines_per_frame = total_atoms + offset

    for dim in (frames, atoms, xyz_dim):
        if not isinstance(dim, np.ndarray):
            raise TypeError(f"invalid index type: {type(dim)}")

    skipped_lines = set(range(offset)).union(
        {atom_id + offset for atom_id in range(total_atoms) if atom_id not in atoms}
    )

    # TODO: update reader to allow arrays with shapes other than 3 for last dimension
    array = np.zeros((len(frames), len(atoms), 3), dtype=dtype)

    with open(filename, "rb") as file_handle:
        for i, coords in enumerate(
            frame_generator(
                file_handle,
                frames,
                lines_per_frame,
                skip_lines_in_frame=skipped_lines,
                usecol=usecol,
            )
        ):
            array[i] = coords

    return array[:, :, xyz_dim]


class PropertyName(StrEnum):
    ATOMS = "species"
    POSITIONS = "pos"
    VELOCITIES = "vel"
    FORCES = "forces"


DATA_VAR_NAME: dict[PropertyName, DataVar] = {
    PropertyName.POSITIONS: DataVar.POSITIONS,
    PropertyName.VELOCITIES: DataVar.VELOCITIES,
    PropertyName.FORCES: DataVar.FORCES,
}

METHOD_PROPERTY_NAME: dict[PropertyName, str] = {
    PropertyName.POSITIONS: "positions",
    PropertyName.VELOCITIES: "velocities",
    PropertyName.FORCES: "forces",
}


# TODO: Extend this to detect which array fields are present dynamically
# TODO: Allow for array fields that have shapes other than (n_frames, n_atoms, 3)
@dataclass
class OnDiskEXTXYZTrajectory:
    filename: PathLike
    dt: float = 1
    dtype: SingleDType = "float64"

    n_frames: int = field(init=False)
    n_atoms: int = field(init=False)
    atoms: list[str] = field(init=False)
    array_names: list[PropertyName] = field(init=False)
    array_cols: dict[PropertyName, slice] = field(init=False)

    def __post_init__(self) -> None:
        self.n_frames, self.atoms, ext_data = get_extxyz_dims_and_details(self.filename)
        self.n_atoms = len(self.atoms)
        prop_data = ext_data.get(
            "properties",
            [(PropertyName.ATOMS, str, 1), (PropertyName.POSITIONS, float, 3)],
        )
        self.array_names = [
            PropertyName(name.lower())
            for name, _, _ in prop_data
            if name.lower() != PropertyName.ATOMS
        ]
        self.array_cols = self._get_data_cols(prop_data)

    def _get_data_cols(
        self, property_data: list[tuple[str, type, int]]
    ) -> dict[PropertyName, slice]:
        cols: dict[PropertyName, slice] = {}
        col_counter = 0

        for name, _, ncols in property_data:
            if name not in PropertyName:
                raise ValueError(f"Cannot recognize property '{name}'")

            cols[PropertyName(name)] = slice(col_counter, col_counter + ncols, 1)
            col_counter += ncols
        return cols

    @property
    def positions(self) -> OnDiskArray:
        parser_fn = partial(
            read_extxyz_frames,
            filename=self.filename,
            usecol=self.array_cols[PropertyName.POSITIONS],
            total_atoms=self.n_atoms,
            dtype=self.dtype,
        )
        return OnDiskArray(parser_fn, (self.n_frames, self.n_atoms, 3))

    @property
    def velocities(self) -> OnDiskArray:
        parser_fn = partial(
            read_extxyz_frames,
            filename=self.filename,
            usecol=self.array_cols[PropertyName.VELOCITIES],
            total_atoms=self.n_atoms,
            dtype=self.dtype,
        )
        return OnDiskArray(parser_fn, (self.n_frames, self.n_atoms, 3))

    @property
    def forces(self) -> OnDiskArray:
        parser_fn = partial(
            read_extxyz_frames,
            filename=self.filename,
            usecol=self.array_cols[PropertyName.FORCES],
            total_atoms=self.n_atoms,
            dtype=self.dtype,
        )
        return OnDiskArray(parser_fn, (self.n_frames, self.n_atoms, 3))

    def get_data_vars(self) -> tuple[tuple[str, tuple[str, ...], OnDiskArray], ...]:
        data_vars = []
        for prop_name in self.array_names:
            method_name = METHOD_PROPERTY_NAME.get(prop_name, None)

            if method_name is None:
                raise KeyError(f"mapping for {prop_name} to method not found")

            data_var_array = getattr(self, method_name, None)

            if data_var_array is None:
                raise AttributeError(f"method for {prop_name} not found")

            data_var = DATA_VAR_NAME.get(prop_name, None)
            if data_var is None:
                raise KeyError(f"mapping for {prop_name} to DataVar not found")

            data_vars.append(
                (data_var, DATA_VAR_DIMS[data_var], cast(OnDiskArray, data_var_array))
            )
        return tuple(data_vars)

    def get_coords(self) -> tuple[tuple[str, tuple[str, ...], TrajNDArray], ...]:
        return (
            (Coord.TIME, DATA_VAR_DIMS[Coord.TIME], np.arange(self.n_frames) * self.dt),
            (Coord.ATOMID, DATA_VAR_DIMS[Coord.ATOMID], np.arange(self.n_atoms)),
            (Coord.ATOM, DATA_VAR_DIMS[Coord.ATOM], np.asarray(self.atoms)),
            (Coord.SPACE, DATA_VAR_DIMS[Coord.SPACE], DEFAULT_COORDS[Coord.SPACE]),
        )

    def get_attrs(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "file_format": TrajectoryFormat.EXTXYZ,
        }
