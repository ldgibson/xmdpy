from collections.abc import Iterable
from itertools import product

import numpy as np
import numpy.typing as npt
import xarray as xr

from .accessor import TrajectoryAccessor

__all__ = ["Cell", "TrajectoryAccessor"]

type SingleDType = np.dtype | type | str


class ShapeError(Exception):
    def __init__(
        self, array: npt.ArrayLike, target_shape: tuple[int] | None = None
    ) -> None:
        self.message = (
            f"Provided array ({array.__name__}) has invalid shape: {array.shape}."
        )

        if target_shape is not None:
            self.message += f"\nArray must have shape: {target_shape}."
        super().__init__(self.message)


def normalize_cell(
    cell: npt.ArrayLike,
    n_frames: int | None = None,
    dtype: SingleDType | None = None,
    copy: bool | None = None,
) -> npt.NDArray:
    cell = np.asarray(cell, dtype=dtype, copy=copy)

    if n_frames is None:
        n_frames = 1

    # Scalar
    if cell.shape == ():
        return np.broadcast_to(np.eye(3) * cell, (n_frames, 3, 3))

    # Cell lengths
    if cell.shape == (3,):
        return np.broadcast_to(np.diag(cell), (n_frames, 3, 3))

    # Cell vectors (avoids case with cell lengths given for 3 frames)
    if cell.shape == (3, 3) and n_frames != 3:
        return np.broadcast_to(cell, (n_frames, 3, 3))

    # Cell lengths per-frame
    if cell.ndim == 2 and cell.shape[1:] == (3,):
        return np.stack([np.diag(c) for c in cell])

    # Cell vectors per-frame
    if cell.ndim == 3 and cell.shape[1:] == (3, 3):
        return cell

    raise ShapeError(cell)


class Cell:
    def __init__(
        self,
        array: npt.ArrayLike,
        n_frames: int | None = None,
        shape_tol: float = 1e-6,
        dtype: SingleDType | None = None,
        copy: bool | None = None,
    ) -> None:
        self.array = normalize_cell(array, n_frames=n_frames, dtype=dtype, copy=copy)
        self._shape_tol = shape_tol

    def __repr__(self) -> str:
        array_string = np.array2string(self.array, prefix="Cell(", suffix=")")
        return f"Cell({array_string})"

    def __len__(self) -> int:
        return self.array.shape[0]

    def __getitem__(self, key):
        return self.array.__getitem__(key)

    def __array__(
        self, dtype: SingleDType | None = None, copy: bool | None = None
    ) -> npt.NDArray:
        return np.asarray(self.array, dtype=dtype, copy=copy)

    @property
    def volume(self) -> float:
        return np.linalg.det(self.array)

    @property
    def lengths(self) -> npt.NDArray:
        return np.linalg.norm(self.array, axis=-1)

    @property
    def angles(self) -> None:
        raise NotImplementedError("Cell angles not yet implemented.")

    def is_orthorhombic(self):
        """Checks if cell is orthorhombic."""
        off_diag_indices = np.where(~np.eye(3, dtype=bool))
        off_diag = self.array[:, *off_diag_indices]
        return np.all(abs(off_diag) < self._shape_tol)

    def is_symmetric(self) -> bool:
        """Checks if cell is symmetric."""
        return np.all(self.array == np.swapaxes(self.array, 1, 2))

    def is_constant(self) -> bool:
        """Checks if cell is constant."""
        if self.__len__() == 1:
            return True
        return np.all(self.array[0][np.newaxis, :, :] == self.array[:])

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
        dims = ["cell_vector", "xyz_dim"]
        coords = {
            "cell_vector": ["A", "B", "C"],
            "xyz_dim": ["x", "y", "z"],
        }

        if isinstance(frame_index, int):
            frame_index = [frame_index]

        if frame_index is None and self.__len__() > 1:
            frame_index = range(self.__len__())

        if frame_index is not None:
            dims.insert(0, "frame")
            coords["frame"] = frame_index

        return xr.DataArray(
            name="cell",
            data=self.array,
            dims=dims,
            coords=coords,
        )


FORMAT_STYLES = {
    "BOLD": "\033[1m",  # ANSI escape sequence for bold text
    "NOBOLD": "\033[22m",
    # "RED": "\033[31m",
    # "END": "\033[0m",  # ANSI escape sequence to reset formatting
}


def formatted_2d_array_string(
    array: npt.ArrayLike,
    indices: list[tuple[int, int]] = [],
    float_format: str = "10.6f",
    highlight_style: str = "BOLD",
    default_style: str = "",
) -> str:
    """
    Generates a string from an array of floats with specified elements highlighted.

    Args:
        array (npt.ArrayLike): Input array.
        indices (list[tuple[int, int]], optional): Indices of elements to be highlighted. Defaults to [].
        float_format (str, optional): Format string for floats. Defaults to "10.6f".
        highlight_style (str, optional): Font style for highlighted elements. Defaults to "BOLD".
        default_style (str, optional): Default font style. Defaults to "".

    Raises:
        ValueError: If requested font style is not available.

    Returns:
        str: Strong of formatted array
    """
    if isinstance(highlight_style, str):
        if highlight_style.upper() != "BOLD":
            raise NotImplementedError(
                "Styles other than BOLD and NOBOLD are not implemented."
            )
        highlight_style_str = FORMAT_STYLES[highlight_style.upper()]
    else:
        raise ValueError(
            f"Must use styles listed in FORMAT_STYLES: {FORMAT_STYLES.keys()}"
        )

    if not default_style:
        default_style_str = FORMAT_STYLES["NOBOLD"]
    elif isinstance(default_style, str):
        if default_style.upper() != "NOBOLD":
            raise NotImplementedError(
                "Styles other than BOLD and NOBOLD not implemented."
            )
        default_style_str = FORMAT_STYLES[default_style.upper()]
    else:
        raise ValueError(
            f"Must use styles listed in FORMAT_STYLES: {FORMAT_STYLES.keys()}"
        )

    array_str = default_style_str

    for count, (i, j) in enumerate(product(range(3), range(3)), start=1):
        if j == 0:
            array_str += "    "

        # Apply formatting to nonzero off-diagonal elements
        if (i, j) in indices:
            array_str += highlight_style_str

        array_str += f"{array[i, j]:{float_format}}"

        # Reset formatting
        if (i, j) in indices:
            array_str += default_style_str

        if count % 3 == 0:
            array_str += "\n"
        else:
            array_str += "  "

    return array_str


class NotOrthorhombicError(Exception):
    """Custom exception for raising errors when a cell is not orthorhombic."""

    def __init__(
        self,
        cell: Cell,
        message: str = "",
        max_frames: int = 3,
        bad_frames: Iterable[int] = [0],
    ) -> None:
        self.cell = cell
        self.bad_frames = bad_frames
        self.max_frames = max_frames
        self.message = self.build_message(message)
        super().__init__(self.message)

    def build_message(self, message: str) -> str:
        cell_strings = self.get_bad_cell_strings()
        return message + "\n" + "".join(cell_strings)

    def get_bad_cell_strings(self) -> list[str]:
        bad_cell_strings = []

        off_diag_idx = ~np.eye(3, dtype=bool)

        for frame_count, frame_id in enumerate(self.bad_frames):
            if frame_count >= self.max_frames:
                bad_cell_strings.append("...")
                break

            nonzero_idx = np.where(off_diag_idx & (abs(self.cell[frame_id]) > 0.0))
            nonzero_idx_list = list(zip(*nonzero_idx))

            cell_str = formatted_2d_array_string(
                array=self.cell[frame_id],
                indices=nonzero_idx_list,
            )
            bad_cell_strings.append(f"FrameID = {frame_id}\n" + cell_str)
        return bad_cell_strings
