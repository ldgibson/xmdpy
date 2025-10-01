import dask
import dask.array as da
import numpy as np
import pytest

from xmdpy.analysis import (
    compute_distance_vectors,
    compute_distance_vectors_across_frames,
    compute_distances,
    compute_distances_across_frames,
    compute_radial_distribution,
    construct_bins,
    get_radial_weights,
    is_symmetric,
    lazy_take,
    wrap,
    wrap_trajectory,
)
from xmdpy.cell import Cell


@pytest.fixture
def wrappable_xyz() -> np.ndarray:
    return np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [8.0, 20.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 3.0, 10.0], [-1.0, 10.0, 6.0]],
        ]
    )


@pytest.fixture
def wrapped_xyz() -> np.ndarray:
    return np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [-2.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 3.0, 0.0], [-1.0, 0.0, -4.0]],
        ]
    )


@pytest.fixture
def distance_vectors() -> np.ndarray:
    return np.array(
        [
            [
                [[0.0, 0.0, 0.0], [-1.0, -1.0, 0.0], [2.0, 0.0, 0.0]],
                [[1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [3.0, 1.0, 0.0]],
                [[-2.0, 0.0, 0.0], [-3.0, -1.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [0.0, -3.0, 0.0], [1.0, 0.0, 4.0]],
                [[0.0, 3.0, 0.0], [0.0, 0.0, 0.0], [1.0, 3.0, 4.0]],
                [[-1.0, 0.0, -4.0], [-1.0, -3.0, -4.0], [0.0, 0.0, 0.0]],
            ],
        ]
    )


@pytest.fixture
def valid_cell() -> Cell:
    return Cell(10.0, n_frames=2)


@pytest.fixture
def non_orthorhombic_cell() -> Cell:
    return Cell([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [2.0, 0.0, 10.0]], n_frames=2)


def test_wrap_success(wrappable_xyz, valid_cell, wrapped_xyz) -> None:
    result = wrap(wrappable_xyz[0], valid_cell.lengths[0])
    np.testing.assert_array_almost_equal(result, wrapped_xyz[0])


def test_wrap_trajectory_success(wrappable_xyz, valid_cell, wrapped_xyz) -> None:
    result = wrap_trajectory(wrappable_xyz, valid_cell.lengths)
    np.testing.assert_array_almost_equal(result, wrapped_xyz)


def test_compute_distance_vectors_no_cell(wrapped_xyz, distance_vectors) -> None:
    xyz1 = wrapped_xyz[0]
    xyz2 = wrapped_xyz[0]

    result = compute_distance_vectors(xyz1, xyz2)
    np.testing.assert_array_almost_equal(result, distance_vectors[0])


def test_compute_distance_vectors_with_cell(
    wrappable_xyz, valid_cell, distance_vectors
) -> None:
    xyz1 = wrappable_xyz[0]
    xyz2 = wrappable_xyz[0]
    cell_lengths = valid_cell.lengths[0]

    result = compute_distance_vectors(xyz1, xyz2, cell_lengths)
    np.testing.assert_array_almost_equal(result, distance_vectors[0])


def test_compute_distances_no_cell(wrapped_xyz, distance_vectors) -> None:
    xyz1 = wrapped_xyz[0]
    xyz2 = wrapped_xyz[0]
    result = compute_distances(xyz1, xyz2)
    np.testing.assert_array_almost_equal(
        result, np.linalg.norm(distance_vectors, axis=-1)[0]
    )


def test_compute_distances_with_cell(
    wrappable_xyz, valid_cell, distance_vectors
) -> None:
    xyz1 = wrappable_xyz[0]
    xyz2 = wrappable_xyz[0]
    result = compute_distances(xyz1, xyz2, valid_cell.lengths[0])
    np.testing.assert_array_almost_equal(
        result, np.linalg.norm(distance_vectors, axis=-1)[0]
    )


def test_compute_distance_vectors_across_frames_no_cell(
    wrapped_xyz, distance_vectors
) -> None:
    xyz1 = wrapped_xyz
    xyz2 = wrapped_xyz
    result = compute_distance_vectors_across_frames(xyz1, xyz2)
    np.testing.assert_array_almost_equal(result, distance_vectors)


def test_compute_distance_vectors_across_frames_with_cell(
    wrappable_xyz, valid_cell, distance_vectors
) -> None:
    xyz1 = wrappable_xyz
    xyz2 = wrappable_xyz
    result = compute_distance_vectors_across_frames(xyz1, xyz2, valid_cell.lengths)
    np.testing.assert_array_almost_equal(result, distance_vectors)


def test_compute_distance_vectors_across_frames_different_lengths(
    wrappable_xyz,
) -> None:
    xyz1 = wrappable_xyz
    xyz2 = wrappable_xyz[[0]]
    with pytest.raises(ValueError):
        compute_distance_vectors_across_frames(xyz1, xyz2)


def test_compute_distances_across_frames_no_cell(wrapped_xyz, distance_vectors) -> None:
    xyz1 = wrapped_xyz
    xyz2 = wrapped_xyz
    result = compute_distances_across_frames(xyz1, xyz2)
    np.testing.assert_array_almost_equal(
        result, np.linalg.norm(distance_vectors, axis=-1)
    )


def test_compute_distances_across_frames_with_cell(
    wrappable_xyz, valid_cell, distance_vectors
) -> None:
    xyz1 = wrappable_xyz
    xyz2 = wrappable_xyz
    result = compute_distances_across_frames(xyz1, xyz2, valid_cell.lengths)
    np.testing.assert_array_almost_equal(
        result, np.linalg.norm(distance_vectors, axis=-1)
    )


def test_is_symmetric_true() -> None:
    assert is_symmetric(np.eye(3))


def test_is_symmetric_false() -> None:
    arr = np.eye(3, dtype="uint8")
    arr[1, 0] = 1
    assert not is_symmetric(arr)


def test_is_symmetric_false_ndims_greater_than_2() -> None:
    arr = np.full((3, 3, 3), 1.0)
    assert not is_symmetric(arr)


def test_is_symmetric_false_not_square() -> None:
    arr = np.full((4, 3), 1.0)
    assert not is_symmetric(arr)


@pytest.fixture
def values() -> np.ndarray[tuple[int], np.dtype[np.int64]]:
    return np.array([0, 2, 4, 6, 8])


@pytest.mark.parametrize("mode", ["clip", "wrap"])
def test_lazy_take_modes_numpy(values: np.ndarray, mode: str) -> None:
    indices = np.stack([np.arange(5), np.arange(5)[::-1]])

    result = lazy_take(values, indices, mode=mode)
    expected = np.take(values, indices, mode=mode)  # type: ignore

    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("mode", ["clip", "wrap"])
def test_lazy_take_modes_dask(values: np.ndarray, mode: str) -> None:
    indices = np.stack([np.arange(5), np.arange(5)[::-1]])

    dask_indices = da.from_array(indices, chunks=(2, 1))  # type: ignore

    result = lazy_take(values, dask_indices, mode=mode)
    expected = np.take(values, indices, mode=mode)  # type: ignore

    np.testing.assert_allclose(result.compute(), expected)  # type: ignore


def test_lazy_take_raise_mode(values: np.ndarray) -> None:
    indices = np.stack([np.arange(10), np.arange(10)[::-1]])

    with pytest.raises(IndexError):
        lazy_take(values, indices, mode="raise")


def test_lazy_take_raise_mode_dask(values: np.ndarray) -> None:
    indices = np.stack([np.arange(10), np.arange(10)[::-1]])

    dask_indices = da.from_array(indices, chunks=(2, 1))  # type: ignore

    with pytest.raises(IndexError):
        lazy_take(values, dask_indices, mode="raise").compute()  # type: ignore


def test_lazy_take_values_as_dask_array_raises_error(values: np.ndarray) -> None:
    indices = np.stack([np.arange(5), np.arange(5)[::-1]])

    dask_values = da.from_array(values, chunks=(1,))  # type: ignore

    with pytest.raises(TypeError):
        lazy_take(dask_values, indices, mode="clip")


def test_lazy_take_indices_as_dask_array(values: np.ndarray) -> None:
    indices = np.stack([np.arange(5), np.arange(5)[::-1]])

    dask_indices = da.from_array(indices, chunks=(2, 1))  # type: ignore

    result = lazy_take(values, dask_indices, mode="clip")
    expected = np.take(values, indices, mode="clip")  # type: ignore

    np.testing.assert_allclose(result.compute(), expected)  # type: ignore


def test_get_radial_weights_uniform_bins() -> None:
    r_target = np.array([[0.5, 1.5, 2.5], [2.5, 3.5, 4.5]])
    volume = np.ones((2, 1))
    n_pairs = 3
    n_frames = 2

    r_bins = np.linspace(0, 5, 6)

    result = get_radial_weights(
        distances=r_target,
        volume=volume,
        n_pairs=n_pairs,
        n_frames=n_frames,
        r_bins=r_bins,
    )

    dr = r_bins[1] - r_bins[0]
    expected = 1 / (4 * np.pi * r_target**2 * dr * n_pairs * n_frames)

    np.testing.assert_allclose(result, expected)


def test_get_radial_weights_use_bin_centers() -> None:
    r_target = np.array([[0.4, 1.7, 2.9999999], [2.5, 3.5, 4.5]])
    volume = np.ones((2, 1))
    n_pairs = 3
    n_frames = 2

    r_bins = np.linspace(0, 5, 6)

    result = get_radial_weights(
        distances=r_target,
        volume=volume,
        n_pairs=n_pairs,
        n_frames=n_frames,
        r_bins=r_bins,
    )

    r_binned = np.array([[0.5, 1.5, 2.5], [2.5, 3.5, 4.5]])
    dr = r_bins[1] - r_bins[0]
    expected = 1 / (4 * np.pi * r_binned**2 * dr * n_pairs * n_frames)

    np.testing.assert_allclose(result, expected)


def test_get_radial_weights_nonuniform_bins() -> None:
    r_target = np.array([[0.5, 1.5, 2.5], [2.5, 3.5, 4.5]])
    volume = np.ones((2, 1))
    n_pairs = 3
    n_frames = 2

    r_bins = np.array([0, 1, 2, 3, 5])

    result = get_radial_weights(
        distances=r_target,
        volume=volume,
        n_pairs=n_pairs,
        n_frames=n_frames,
        r_bins=r_bins,
    )

    r_binned = np.array([[0.5, 1.5, 2.5], [2.5, 4.0, 4.0]])
    dr = np.array([[1, 1, 1], [1, 2, 2]])
    expected = 1 / (4 * np.pi * r_binned**2 * dr * n_pairs * n_frames)

    np.testing.assert_allclose(result, expected)


def test_get_radial_weights_variable_volume() -> None:
    r_target = np.array([[0.5, 1.5, 2.5], [2.5, 3.5, 4.5]])
    volume = np.array([[1.0], [2.0]])
    n_pairs = 3
    n_frames = 2

    r_bins = np.linspace(0, 5, 6)

    result = get_radial_weights(
        distances=r_target,
        volume=volume,
        n_pairs=n_pairs,
        n_frames=n_frames,
        r_bins=r_bins,
    )

    dr = r_bins[1] - r_bins[0]
    expected = 1 / ((4 * np.pi * r_target**2 * dr) * (n_pairs / volume) * n_frames)

    np.testing.assert_allclose(result, expected)


def test_get_radial_weights_with_dask_returns_correct_dask_array() -> None:
    r_target = np.array([[0.5, 1.5, 2.5], [2.5, 3.5, 4.5]])
    r_target_dask = da.from_array(r_target, chunks=(2, 1))  # type: ignore
    volume_dask = da.ones((2, 1), chunks=(2, 1))
    n_pairs = 3
    n_frames = 2

    r_bins = np.linspace(0, 5, 6)

    result = get_radial_weights(
        distances=r_target_dask,
        volume=volume_dask,
        n_pairs=n_pairs,
        n_frames=n_frames,
        r_bins=r_bins,
    )

    dr = r_bins[1] - r_bins[0]
    expected = 1 / (4 * np.pi * r_target**2 * dr * n_pairs * n_frames)

    assert dask.is_dask_collection(result)
    np.testing.assert_allclose(result.compute(), expected)  # type: ignore


def test_construct_bins_integer_bins() -> None:
    result = construct_bins(5, range=(0, 5))
    expected = np.linspace(0, 5, num=6)
    np.testing.assert_allclose(result, expected)


def test_construct_bins_uniform_array_bins() -> None:
    array_bins = np.array([0, 1, 2, 3, 4, 5])
    result = construct_bins(array_bins, range=(0, 5))
    np.testing.assert_allclose(result, array_bins)


def test_construct_bins_uniform_list_bins() -> None:
    list_bins = [0, 1, 2, 3, 4, 5]
    result = construct_bins(list_bins, range=(0, 5))
    expected = np.array([0, 1, 2, 3, 4, 5])
    np.testing.assert_allclose(result, expected)


def test_construct_bins_nonuniform_array_bins() -> None:
    array_bins = np.array([0, 2, 3, 5])
    result = construct_bins(array_bins, range=(0, 5))
    np.testing.assert_allclose(result, array_bins)


@pytest.mark.parametrize("bin_range", [(0, 5), (0.5, 10.5), (1, 11.5)])
def test_construct_bins_range(bin_range: tuple[float, float]) -> None:
    n_bins = 5
    result = construct_bins(n_bins, range=bin_range)
    expected = np.linspace(*bin_range, num=n_bins + 1)
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("bin_range", [(0, 5), (0.5, 10.5), (1, 11.5)])
def test_construct_bins_out_of_range_warning(bin_range: tuple[float, float]) -> None:
    with pytest.warns(Warning):
        array_bins = np.array([0, 1, 2, 3, 4, 5, 6])
        result = construct_bins(array_bins, range=bin_range)
        np.testing.assert_allclose(result, array_bins)


def test_compute_radial_distribution_default_bins() -> None:
    volume = 10.0
    n_pairs = 3
    n_frames = 2
    n_bins = 50
    r_range = (0, 5)

    distances = np.array([[0.05, 1.05, 2.05], [2.55, 3.55, 4.55]])

    r, result = compute_radial_distribution(
        distances,
        volume=volume,
        n_pairs=n_pairs,
        n_frames=n_frames,
        r_range=r_range,
    )

    r_bins = np.linspace(*r_range, num=n_bins + 1)
    dr = r_bins[1] - r_bins[0]
    weights = 1 / ((4 * np.pi * distances**2 * dr) * (n_pairs / volume) * n_frames)
    expected_hist, expected_bins = np.histogram(
        distances, bins=r_bins, range=r_range, weights=weights
    )
    expected_bin_centers = (expected_bins[1:] + expected_bins[:-1]) / 2

    np.testing.assert_allclose(r, expected_bin_centers)
    np.testing.assert_allclose(result, expected_hist)


def test_compute_radial_distribution_default_r_range() -> None:
    volume = 10.0
    n_pairs = 3
    n_frames = 2
    n_bins = 5
    r_range = (0, 10)

    distances = np.array([[1.0, 3.0, 5.0], [5.0, 7.0, 9.0]])

    r, result = compute_radial_distribution(
        distances,
        volume=volume,
        n_pairs=n_pairs,
        n_frames=n_frames,
        bins=n_bins,
    )
    r_bins = np.linspace(*r_range, num=n_bins + 1)
    dr = r_bins[1] - r_bins[0]
    weights = 1 / (4 * np.pi * distances**2 * dr * (n_pairs / volume) * n_frames)
    expected_hist, expected_bins = np.histogram(
        distances, bins=n_bins, range=r_range, weights=weights
    )
    expected_bin_centers = (expected_bins[1:] + expected_bins[:-1]) / 2

    np.testing.assert_allclose(r, expected_bin_centers)
    np.testing.assert_allclose(result, expected_hist)


def test_compute_radial_distribution_scalar_volume() -> None:
    distances = np.array([[0.5, 1.5, 2.5], [2.5, 3.5, 4.5]])
    volume = 10.0
    n_pairs = 3
    n_frames = 2
    n_bins = 5
    r_range = (0, 5)

    r, result = compute_radial_distribution(
        distances,
        volume=volume,
        n_pairs=n_pairs,
        n_frames=n_frames,
        bins=n_bins,
        r_range=r_range,
    )
    weights = 1 / (4 * np.pi * distances**2 * 1 * (n_pairs / volume) * n_frames)
    expected_hist, expected_bins = np.histogram(
        distances, bins=n_bins, range=r_range, weights=weights
    )
    expected_bin_centers = (expected_bins[1:] + expected_bins[:-1]) / 2

    np.testing.assert_allclose(r, expected_bin_centers)
    np.testing.assert_allclose(result, expected_hist)


def test_compute_radial_distribution_variable_volume() -> None:
    distances = np.array([[0.5, 1.5, 2.5], [2.5, 3.5, 4.5]])
    volume = np.array([[10.0], [15.0]])
    n_pairs = 3
    n_frames = 2
    n_bins = 5
    r_range = (0, 5)

    r, result = compute_radial_distribution(
        distances,
        volume=volume,
        n_pairs=n_pairs,
        n_frames=n_frames,
        bins=n_bins,
        r_range=r_range,
    )
    weights = 1 / (4 * np.pi * distances**2 * 1 * (n_pairs / volume) * n_frames)
    expected_hist, expected_bins = np.histogram(
        distances, bins=n_bins, range=r_range, weights=weights
    )
    expected_bin_centers = (expected_bins[1:] + expected_bins[:-1]) / 2

    np.testing.assert_allclose(r, expected_bin_centers)
    np.testing.assert_allclose(result, expected_hist)


def test_compute_radial_distribution_flattened_distances() -> None:
    distances = np.array([[0.5, 1.5, 2.5], [2.5, 3.5, 4.5]])
    volume = 10.0
    n_pairs = 3
    n_frames = 2
    n_bins = 5
    r_range = (0, 5)

    r, result = compute_radial_distribution(
        distances.flatten(),
        volume=volume,
        n_pairs=n_pairs,
        n_frames=n_frames,
        bins=n_bins,
        r_range=r_range,
    )

    r_bins = np.linspace(*r_range, num=n_bins + 1)
    weights = 1 / (4 * np.pi * distances**2 * 1 * (n_pairs / volume) * n_frames)
    expected_hist, expected_bins = np.histogram(
        distances, bins=r_bins, range=r_range, weights=weights
    )
    expected_bin_centers = (expected_bins[1:] + expected_bins[:-1]) / 2

    np.testing.assert_allclose(r, expected_bin_centers)
    np.testing.assert_allclose(result, expected_hist)


def test_compute_radial_distribution_wrong_n_pairs_raises_value_error() -> None:
    distances = np.array([[0.5, 1.5, 2.5], [2.5, 3.5, 4.5]])
    volume = 10.0
    with pytest.raises(ValueError):
        _ = compute_radial_distribution(
            distances,
            volume=volume,
            n_pairs=10,
            n_frames=2,
            bins=5,
            r_range=(0, 5),
        )


def test_compute_radial_distribution_only_distances_as_dask_array_raises_error() -> (
    None
):
    distances = np.array([[0.5, 1.5, 2.5], [2.5, 3.5, 4.5]])
    volume = np.array([[10.0], [10.0]])
    dask_distances = da.from_array(distances, chunks=(2, 1))  # type: ignore
    n_pairs = 3
    n_frames = 2
    n_bins = 5
    r_range = (0, 5)

    with pytest.raises(TypeError):
        _ = compute_radial_distribution(
            dask_distances,
            volume=volume,
            n_pairs=n_pairs,
            n_frames=n_frames,
            bins=n_bins,
            r_range=r_range,
        )


def test_compute_radial_distribution_volume_as_dask_array_raises_error() -> None:
    distances = np.array([[0.5, 1.5, 2.5], [2.5, 3.5, 4.5]])
    volume = np.array([[10.0], [10.0]])
    dask_volume = da.from_array(volume, chunks=(2, 1))  # type: ignore
    n_pairs = 3
    n_frames = 2
    n_bins = 5
    r_range = (0, 5)

    with pytest.raises(TypeError):
        _ = compute_radial_distribution(
            distances,
            volume=dask_volume,
            n_pairs=n_pairs,
            n_frames=n_frames,
            bins=n_bins,
            r_range=r_range,
        )


def test_compute_radial_distribution_dask_arrays() -> None:
    distances = np.array([[0.5, 1.5, 2.5], [2.5, 3.5, 4.5]])
    volume = np.array([[10.0], [10.0]])
    dask_volume = da.from_array(volume, chunks=(2, 1))  # type: ignore
    dask_distances = da.from_array(distances, chunks=(2, 1))  # type: ignore
    n_pairs = 3
    n_frames = 2
    n_bins = 5
    r_range = (0, 5)

    r, result = compute_radial_distribution(
        dask_distances,
        volume=dask_volume,
        n_pairs=n_pairs,
        n_frames=n_frames,
        bins=n_bins,
        r_range=r_range,
    )
    weights = 1 / (4 * np.pi * distances**2 * 1 * (n_pairs / volume) * n_frames)
    expected_hist, expected_bins = np.histogram(
        distances, bins=n_bins, range=r_range, weights=weights
    )
    expected_bin_centers = (expected_bins[1:] + expected_bins[:-1]) / 2

    assert dask.is_dask_collection(result)
    np.testing.assert_allclose(r, expected_bin_centers)
    np.testing.assert_allclose(result.compute(), expected_hist)  # type: ignore
