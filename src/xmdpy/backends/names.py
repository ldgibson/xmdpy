from enum import StrEnum
from typing import Any


class Dim(StrEnum):
    TIME = "time"
    ATOMID = "atom_id"
    SPACE = "xyz_dim"
    CELL = "cell_vector"


class Coord(StrEnum):
    TIME = "time"
    ATOMID = "atom_id"
    ATOM = "atoms"
    SPACE = "xyz_dim"
    CELL = "cell_vector"


class DataVar(StrEnum):
    POSITIONS = "xyz"
    CELL = "cell"
    VELOCITIES = "velocities"
    CHARGES = "charges"
    DIPOLE = "dipole"


DATA_VAR_DIMS: dict[Coord | DataVar, tuple[Dim, ...]] = {
    Coord.TIME: (Dim.TIME,),
    Coord.ATOMID: (Dim.ATOMID,),
    Coord.ATOM: (Dim.ATOMID,),
    Coord.SPACE: (Dim.SPACE,),
    Coord.CELL: (Dim.CELL,),
    DataVar.POSITIONS: (Dim.TIME, Dim.ATOMID, Dim.SPACE),
    DataVar.VELOCITIES: (Dim.TIME, Dim.ATOMID, Dim.SPACE),
    DataVar.CELL: (Dim.TIME, Dim.CELL, Dim.SPACE),
    DataVar.CHARGES: (Dim.TIME, Dim.ATOMID),
    DataVar.DIPOLE: (Dim.TIME, Dim.ATOMID, Dim.SPACE),
}


DEFAULT_COORDS: dict[Coord, Any] = {
    Coord.SPACE: list("xyz"),
    Coord.CELL: list("ABC"),
}
