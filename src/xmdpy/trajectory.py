from dataclasses import dataclass

from xmdpy.types import CellNDArray, TrajNDArray


@dataclass
class Trajectory:
    atoms: tuple[str]
    positions: TrajNDArray
    cell: CellNDArray | None = None
