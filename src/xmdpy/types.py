import os
from typing import Any

import numpy as np

type SingleDType = np.dtype | type | str
type FloatLike = np.floating | np.integer
type AnyTrajData = FloatLike | np.str_

TrajArray = np.ndarray[tuple[int, int, int], np.dtype[FloatLike]]
CellArray = np.ndarray[tuple[int, int, int], np.dtype[FloatLike]]
CellArray3x3 = np.ndarray[tuple[int, int], np.dtype[FloatLike]]

Int1DArray = np.ndarray[tuple[int], np.dtype[np.integer]]

type _TrajNDArray[Shape: tuple[int, ...]] = np.ndarray[Shape, np.dtype[AnyTrajData]]
TrajNDArray = _TrajNDArray[tuple[int, ...]]

type PathLike = str | bytes | os.PathLike[Any]
