from collections.abc import Callable

import numpy as np

from xmdpy.types import Int1DArray, SingleDType, TrajNDArray

# * The current limit on the dimensionality of an array's indexers is 4D with
# * static typing. Currently, the first index should always refer to the time
# * dimension of the trajectory. The atomic positions in a trajectory are
# * indexed first by time, then by atom ID, and lastly by spatial dimension.


type TrajectoryParserFn = (
    Callable[[Int1DArray], TrajNDArray]
    | Callable[[Int1DArray, Int1DArray], TrajNDArray]
    | Callable[[Int1DArray, Int1DArray, Int1DArray], TrajNDArray]
    | Callable[[Int1DArray, Int1DArray, Int1DArray, Int1DArray], TrajNDArray]
)
type OuterIndex = int | np.integer | slice | Int1DArray


def normalize_index_to_array(index: OuterIndex | None, upper_bound: int) -> Int1DArray:
    if isinstance(index, np.ndarray):
        if index.ndim > 1:
            raise IndexError("index cannot be multidimensional")

        if np.any(index >= upper_bound):
            raise IndexError("index extends beyond upper bound")

        return np.asarray(index, dtype=np.int64)

    if isinstance(index, (int, np.integer)):
        if index >= upper_bound:
            raise IndexError("index extends beyond upper bound")

        return np.array([index])

    if index is None:
        start, stop, step = 0, upper_bound, 1

    elif isinstance(index, slice):
        start = index.start or 0
        stop = index.stop or upper_bound
        step = index.step or 1

    else:
        raise IndexError(f"invalid index type: {type(index)}")

    if start < 0:
        start = upper_bound + start

    if stop < 0:
        stop = upper_bound + stop

    if start >= upper_bound or stop > upper_bound:
        raise IndexError("index extends beyond upper bound")

    return np.arange(start, stop, step, dtype=np.int64)


class OnDiskArray:
    """Represents general arrays within a trajectory"""

    def __init__(
        self,
        parser_fn: TrajectoryParserFn,
        shape: tuple[int, ...],
        dtype: SingleDType = "float64",
    ) -> None:
        self.parser_fn = parser_fn
        self.shape = shape
        self.dtype = np.dtype(dtype)

    def __getitem__(self, key: OuterIndex | tuple[OuterIndex, ...], /) -> TrajNDArray:
        if not isinstance(key, tuple):
            key = (key,)

        indices = tuple(
            normalize_index_to_array(index, self.shape[i])
            for i, index in enumerate(key)
        )

        arr = self.parser_fn(*indices)

        if any(isinstance(dim, int) for dim in key):
            arr = np.squeeze(arr)

        return arr
