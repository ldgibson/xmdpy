from dataclasses import dataclass

import numpy as np

from xmdpy.types import CellNDArray, TrajNDArray


@dataclass
class Trajectory:
    symbols: (
        list[str]
        | np.ndarray[tuple[int, int], np.dtype[np.str_]]
        | np.ndarray[tuple[int], np.dtype[np.str_]]
    )
    positions: TrajNDArray
    cell: CellNDArray | None = None
