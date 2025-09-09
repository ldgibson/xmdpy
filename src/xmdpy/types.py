import numpy as np

type SingleDType = np.dtype | type | str
type FloatLike = np.floating | np.integer

TrajNDArray = np.ndarray[tuple[int, int, int], np.dtype[FloatLike]]
CellNDArray = np.ndarray[tuple[int, int, int], np.dtype[FloatLike]]
