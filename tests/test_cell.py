import numpy as np
import pytest
import xarray as xr

from xmdpy.cell import Cell, ShapeError, normalize_cell


@pytest.fixture
def expected_cell_array() -> np.ndarray:
    return np.broadcast_to(np.diag([4, 4, 4]).astype(np.int64), (10, 3, 3))


@pytest.fixture
def expected_cell() -> Cell:
    cell = np.broadcast_to(np.diag([4, 4, 4]).astype(np.int64), (10, 3, 3))
    return Cell(cell)


@pytest.mark.parametrize(
    "cell",
    [
        4,
        [4, 4, 4],
        [[4, 4, 4] for _ in range(10)],
        np.array([4, 4, 4]),
        np.array([[4, 4, 4] for _ in range(10)]),
        np.array([[4, 0, 0], [0, 4, 0], [0, 0, 4]]),
    ],
)
def test_normalize_cell_valid(cell, expected_cell_array) -> None:
    result = normalize_cell(cell, n_frames=10, dtype=np.int64)
    np.testing.assert_array_equal(result, expected_cell_array)


@pytest.mark.parametrize(
    "cell",
    [
        np.full((10,), 10.0),
        np.full((1, 2, 3), 10.0),
        np.full((10, 1, 3, 3), 10.0),
    ],
)
def test_normalize_cell_raises_shape_error(cell) -> None:
    with pytest.raises(ShapeError):
        normalize_cell(cell, n_frames=10)


@pytest.mark.parametrize(
    "cell",
    [
        np.full((1, 3), 10.0),
        np.full((4, 3), 10.0),
    ],
)
def test_normalize_cell_raises_value_error(cell) -> None:
    with pytest.raises(ValueError):
        normalize_cell(cell, n_frames=10)


@pytest.mark.parametrize(
    "cell",
    [
        4,
        [4, 4, 4],
        np.array([4, 4, 4]),
        np.array([[4, 0, 0], [0, 4, 0], [0, 0, 4]]),
    ],
)
def test_normalize_cell_default_n_frames(cell) -> None:
    result = normalize_cell(cell)
    np.testing.assert_array_equal(result, np.diag([4, 4, 4])[None, :, :])


def test_cell_init(expected_cell) -> None:
    result = Cell(array=np.array([[4, 0, 0], [0, 4, 0], [0, 0, 4]]), n_frames=10)
    np.testing.assert_array_equal(result.array, expected_cell.array)


def test_cell_repr() -> None:
    cell = Cell(array=np.array([[4, 0, 0], [0, 4, 0], [0, 0, 4]]), n_frames=1)
    expected = "Cell([[[4. 0. 0.]\n       [0. 4. 0.]\n       [0. 0. 4.]]])"
    assert cell.__repr__() == expected


def test_cell__array__dtype() -> None:
    cell = Cell(
        array=np.array([[4, 0, 0], [0, 4, 0], [0, 0, 4]]), n_frames=1, dtype=np.int64
    )
    result = cell.__array__(dtype=np.uint8)
    assert result.dtype == np.dtype(np.uint8)


def test_cell__array__copy_true() -> None:
    cell = Cell(
        array=np.array([[4, 0, 0], [0, 4, 0], [0, 0, 4]]), n_frames=1, dtype=np.int64
    )
    result = cell.__array__(copy=True)
    result[:] = 1
    assert not np.array_equal(result, cell._array)


def test_cell__array__copy_false() -> None:
    cell = Cell(
        array=np.array([[4, 0, 0], [0, 4, 0], [0, 0, 4]]), n_frames=1, dtype=np.int64
    )
    result = cell.__array__(copy=False)
    assert result is cell._array


def test_cell_len() -> None:
    cell = Cell(array=np.array([[4, 0, 0], [0, 4, 0], [0, 0, 4]]), n_frames=10)
    assert len(cell) == 10


def test_cell_getitem() -> None:
    cell = Cell(array=np.array([[2, 0, 0], [0, 3, 0], [0, 0, 4]]), n_frames=10)
    assert cell[0, 1, 1] == 3


def test_cell_array() -> None:
    cell = Cell(array=np.array([[2, 0, 0], [0, 3, 0], [0, 0, 4]]), n_frames=10)
    assert isinstance(cell.array, np.ndarray)


def test_cell_volume() -> None:
    cell = Cell(array=np.array([[4, 0, 0], [0, 4, 0], [0, 0, 4]]), n_frames=10)
    np.testing.assert_almost_equal(cell.volume, np.full((10,), 64))


def test_cell_lenghts() -> None:
    cell = Cell(array=np.array([[4, 0, 0], [0, 4, 0], [0, 0, 4]]), n_frames=10)
    np.testing.assert_almost_equal(cell.lengths, np.full((10, 3), 4))


@pytest.mark.xfail
def test_cell_angles() -> None:
    cell = Cell(array=np.array([[4, 0, 0], [0, 4, 0], [4, 0, 4]]), n_frames=2)
    np.testing.assert_almost_equal(
        cell.angles, np.array([[90.0, 90.0, 45.0], [90.0, 90.0, 45.0]])
    )


def test_cell_is_orthorhombic_true() -> None:
    cell = Cell(array=np.array([[4, 0, 0], [0, 4, 0], [0, 0, 4]]), n_frames=10)
    assert cell.is_orthorhombic()


def test_cell_is_orthorhombic_false() -> None:
    cell = Cell(array=np.array([[4, 0, 0], [0, 4, 0], [1, 0, 4]]), n_frames=10)
    assert not cell.is_orthorhombic()


def test_cell_is_symmetric_true() -> None:
    cell = Cell(array=np.array([[4, 0.5, 0], [0.5, 4, 0], [0, 0, 4]]), n_frames=10)
    assert cell.is_symmetric()


def test_cell_is_symmetric_false() -> None:
    cell = Cell(array=np.array([[4, 0.5, 0], [0, 4, 0], [0, 0, 4]]), n_frames=10)
    assert not cell.is_symmetric()


def test_cell_is_constant_true() -> None:
    cell = Cell(array=np.array([[4, 0, 0], [0, 4, 0], [0, 0, 4]]), n_frames=10)
    assert cell.is_constant()


def test_cell_is_constant_true_len_1() -> None:
    cell = Cell(array=np.array([[4, 0, 0], [0, 4, 0], [0, 0, 4]]), n_frames=1)
    assert cell.is_constant()


def test_cell_is_constant_false() -> None:
    cell = Cell(array=np.array([[4, 4, 4], [5, 5, 5]]), n_frames=2)
    assert not cell.is_constant()


@pytest.fixture
def expected_cell_dataarray() -> xr.DataArray:
    cell_arr = np.broadcast_to(np.array([[4, 0, 0], [0, 4, 0], [0, 0, 4]]), (10, 3, 3))
    return xr.DataArray(
        name="cell",
        data=cell_arr,
        dims=["time", "cell_vector", "xyz_dim"],
        coords={
            "time": np.arange(10),
            "cell_vector": list("ABC"),
            "xyz_dim": list("xyz"),
        },
    )


def test_cell_to_xarray_default_time_index(expected_cell_dataarray) -> None:
    cell = Cell(array=np.array([[4, 0, 0], [0, 4, 0], [0, 0, 4]]), n_frames=10)
    result = cell.to_xarray()
    xr.testing.assert_equal(result, expected_cell_dataarray)


def test_cell_to_xarray_valid_int_time_index(expected_cell_dataarray) -> None:
    cell = Cell(array=np.array([[4, 0, 0], [0, 4, 0], [0, 0, 4]]), n_frames=1)
    result = cell.to_xarray(time_index=10)
    xr.testing.assert_equal(result, expected_cell_dataarray)


@pytest.mark.parametrize("time_index", [0, -1])
def test_cell_to_xarray_invalid_int_time_index(
    time_index, expected_cell_dataarray
) -> None:
    with pytest.raises(ValueError):
        cell = Cell(array=np.array([[4, 0, 0], [0, 4, 0], [0, 0, 4]]), n_frames=1)
        cell.to_xarray(time_index=time_index)


def test_cell_to_xarray_len_time_index_greater_than_cell(
    expected_cell_dataarray,
) -> None:
    cell = Cell(array=np.array([[4, 0, 0], [0, 4, 0], [0, 0, 4]]), n_frames=1)
    result = cell.to_xarray(time_index=10)
    xr.testing.assert_equal(result, expected_cell_dataarray)


def test_cell_to_xarray_len_time_index_equal_to_cell(expected_cell_dataarray) -> None:
    cell = Cell(array=np.array([[4, 0, 0], [0, 4, 0], [0, 0, 4]]), n_frames=10)
    result = cell.to_xarray(time_index=10)
    xr.testing.assert_equal(result, expected_cell_dataarray)


def test_cell_to_xarray_len_time_index_less_than_cell() -> None:
    with pytest.raises(ValueError):
        cell = Cell(array=np.array([[4, 0, 0], [0, 4, 0], [0, 0, 4]]), n_frames=10)
        cell.to_xarray(time_index=5)


def test_cell_to_xarray_len_time_index_greater_than_2_frame_cell() -> None:
    with pytest.raises(ValueError):
        cell = Cell(array=np.array([[4, 0, 0], [0, 4, 0], [0, 0, 4]]), n_frames=2)
        cell.to_xarray(time_index=5)
