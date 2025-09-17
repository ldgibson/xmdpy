import numpy as np

type SingleDType = np.dtype | type | str
type FloatLike = np.floating | np.integer

TrajArray = np.ndarray[tuple[int, int, int], np.dtype[FloatLike]]
CellArray = np.ndarray[tuple[int, int, int], np.dtype[FloatLike]]
CellArray3x3 = np.ndarray[tuple[int, int], np.dtype[FloatLike]]
