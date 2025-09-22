import os
from typing import Any

import numpy as np

type SingleDType = np.dtype | type | str
type FloatLike = np.floating | np.integer

TrajArray = np.ndarray[tuple[int, int, int], np.dtype[FloatLike]]
CellArray = np.ndarray[tuple[int, int, int], np.dtype[FloatLike]]
CellArray3x3 = np.ndarray[tuple[int, int], np.dtype[FloatLike]]

IntArray = np.ndarray[tuple[int], np.dtype[np.integer]]

type PathLike = str | bytes | os.PathLike[Any]
