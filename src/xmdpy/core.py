from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import numpy as np
import numpy.typing as npt
import xarray as xr

__all__ = ["Cell", "normalize_cell"]

# if TYPE_CHECKING:
type SingleDType = np.dtype | type | str
type FloatLike = np.floating | np.integer

type NumFrames = int
type NumAtoms = int

TrajNDArray = np.ndarray[tuple[NumFrames, NumAtoms, Literal[3]], np.dtype[FloatLike]]
AtomsNDArray = np.ndarray[tuple[NumAtoms, Literal[3]], np.dtype[FloatLike]]
CellNDArray = np.ndarray[tuple[NumFrames, Literal[3], Literal[3]], np.dtype[FloatLike]]


class ShapeError(Exception):
    def __init__(
        self,
        invalid_shape: tuple[int, ...],
        target_shape: tuple[int, ...] | None = None,
    ) -> None:
        self.message = f"Invalid shape: {invalid_shape}."

        if target_shape is not None:
            self.message += f" Shape must be: {target_shape}."
        super().__init__(self.message)


def normalize_cell(
    cell: npt.ArrayLike,
    n_frames: NumFrames | None = None,
    dtype: SingleDType | None = None,
    copy: bool | None = None,
) -> CellNDArray:
    cell = np.asarray(cell, dtype=dtype, copy=copy)

    if n_frames is None:
        n_frames = 1

    # Handle different cases by number of dimensions and shape of array
    match (cell.ndim, cell.shape):
        # Scalar
        case (0, ()):
            normalized_cell = np.broadcast_to(np.eye(3) * cell, (n_frames, 3, 3))

        # Cell lengths
        case (1, (3,)):
            normalized_cell = np.broadcast_to(np.diag(cell), (n_frames, 3, 3))

        # Cell lengths per-frame
        case (2, (N, 3)) if N != 3:
            if n_frames != N:
                raise ValueError(
                    f"Number of frames ({n_frames=}) does not match shape of array ({cell.shape=})"
                )
            normalized_cell = np.stack([np.diag(c) for c in cell])

        # Cell vectors (avoids case with cell lengths given for 3 frames)
        case (2, (3, 3)):
            normalized_cell = np.broadcast_to(cell, (n_frames, 3, 3))

        # Cell vectors per-frame
        case (3, (N, 3, 3)):
            normalized_cell = cell

        # All other shapes are invalid
        case (_, _):
            raise ShapeError(invalid_shape=cell.shape, target_shape=(-1, 3, 3))

    return normalized_cell


class Cell:
    def __init__(
        self,
        array: npt.ArrayLike,
        n_frames: int | None = None,
        shape_tol: float = 1e-6,
        dtype: SingleDType = np.float64,
        copy: bool | None = None,
    ) -> None:
        self._array = normalize_cell(array, n_frames=n_frames, dtype=dtype, copy=copy)
        self._shape_tol = shape_tol

    def __repr__(self) -> str:
        array_string = np.array2string(self._array, prefix="Cell(", suffix=")")
        return f"Cell({array_string})"

    def __len__(self) -> NumFrames:
        return self._array.shape[0]

    def __getitem__(self, key):
        return self._array.__getitem__(key)

    def __array__(
        self, dtype: SingleDType | None = None, copy: bool | None = None
    ) -> CellNDArray:
        return np.asarray(self._array, dtype=dtype, copy=copy)

    @property
    def array(self) -> CellNDArray:
        return self._array

    @property
    def volume(self) -> np.ndarray[tuple[NumFrames,], np.dtype[np.float64]]:
        return np.linalg.det(self._array)

    @property
    def lengths(self) -> np.ndarray[tuple[NumFrames, Literal[3]], np.dtype[np.float64]]:
        return np.linalg.norm(self._array, axis=-1)

    @property
    def angles(self) -> None:
        raise NotImplementedError("Cell angles not yet implemented.")

    def is_orthorhombic(self) -> bool:
        """Checks if cell is orthorhombic."""
        return not (np.nonzero(self.array.reshape(-1, 9))[1] % 4).any()

    def is_symmetric(self) -> bool:
        """Checks if cell is symmetric."""
        return np.allclose(
            self._array, np.swapaxes(self._array, 1, 2), atol=self._shape_tol
        )

    def is_constant(self) -> bool:
        """Checks if cell is constant."""
        if self.__len__() == 1:
            return True
        return np.allclose(self._array[0][np.newaxis, :, :], self._array)

    def to_xarray(self, frame_index: Iterable[int] | int | None = None) -> xr.DataArray:
        """Exports instance to a `xarray.DataArray` with optional `frame_index`.

        Dimensions in DataArray will be `"cell_vector"` and `"xyz_dim"` by
        default. If a `frame_index` is supplied, and/or the length of the
        `Cell` is greater than one, then a `"frame"` dimension will also be
        included.

        Args:
            frame_index (Iterable[int] | int | None, optional): If specified,
            `frame_index` is used as index for `frame` dimension of resulting
            DataArray. Otherwise, the index will try to be inferred from the
            length of the `Cell` instance. If the length is greater than 1,
            then `frame_index` will be set to `range(length)`. If the length
            is equal to 1, then no `frame` dimension is added to the DataArray.
            Defaults to None.

        Returns:
            xr.DataArray: DataArray with labeled coordinates and dimensions.
        """
        if frame_index is None:
            frame_index = range(self.__len__())

        elif isinstance(frame_index, int):
            frame_index = [frame_index]

        return xr.DataArray(
            name="cell",
            data=self._array,
            dims=["frame", "cell_vector", "xyz_dim"],
            coords={
                "frame": frame_index,
                "cell_vector": list("ABC"),
                "xyz_dim": list("xyz"),
            },
        )
