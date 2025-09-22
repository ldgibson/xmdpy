from dataclasses import dataclass

from xmdpy.types import (
    CellArray,
    TrajArray,
)


@dataclass
class Trajectory:
    atoms: tuple[str]
    positions: TrajArray
    cell: CellArray | None = None
