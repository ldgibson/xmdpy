import numpy as np
import pytest

from xmdpy.analysis import (
    compute_distance_vectors,
    compute_pairwise_distance_vectors,
    compute_pairwise_distances,
    compute_pairwise_distances_in_frame,
    compute_radial_distribution,
    is_symmetric,
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


def test_compute_pairwise_distances_in_frame(wrapped_xyz, distance_vectors) -> None:
    xyz1 = wrapped_xyz[0]
    xyz2 = wrapped_xyz[0]

    result = compute_pairwise_distances_in_frame(xyz1, xyz2)
    np.testing.assert_array_almost_equal(result, distance_vectors[0])


def test_compute_pairwise_distance_vectors_no_cell(
    wrapped_xyz, distance_vectors
) -> None:
    xyz1 = wrapped_xyz
    xyz2 = wrapped_xyz
    result = compute_pairwise_distance_vectors(xyz1, xyz2)
    np.testing.assert_array_almost_equal(result, distance_vectors)


def test_compute_pairwise_distance_vectors_with_cell(
    wrappable_xyz, valid_cell, distance_vectors
) -> None:
    xyz1 = wrappable_xyz
    xyz2 = wrappable_xyz
    result = compute_pairwise_distance_vectors(xyz1, xyz2, valid_cell.lengths)
    np.testing.assert_array_almost_equal(result, distance_vectors)


def test_compute_pairwise_distance_vectors_different_lengths(wrappable_xyz) -> None:
    xyz1 = wrappable_xyz
    xyz2 = wrappable_xyz[[0]]
    with pytest.raises(ValueError):
        compute_pairwise_distance_vectors(xyz1, xyz2)


def test_compute_pairwise_distances_no_cell(wrapped_xyz, distance_vectors) -> None:
    xyz1 = wrapped_xyz
    xyz2 = wrapped_xyz
    result = compute_pairwise_distances(xyz1, xyz2)
    np.testing.assert_array_almost_equal(
        result, np.linalg.norm(distance_vectors, axis=-1)
    )


def test_compute_pairwise_distances_with_cell(
    wrappable_xyz, valid_cell, distance_vectors
) -> None:
    xyz1 = wrappable_xyz
    xyz2 = wrappable_xyz
    result = compute_pairwise_distances(xyz1, xyz2, valid_cell.lengths)
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


@pytest.mark.xfail
def test_compute_radial_distribution(distance_vectors) -> None:
    distances = np.linalg.norm(distance_vectors, axis=-1)
    volume = 10.0
    r, rdf = compute_radial_distribution(distances, volume, bins=5, r_range=(0, 5))
    assert False
