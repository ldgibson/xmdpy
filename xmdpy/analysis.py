from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from .core import Cell, NotOrthorhombicError

# type M = int
# type N = int

# type NDArray_NM = np.ndarray[tuple[int, N, M], np.dtype[np.float64]]


def wrap(xyz: npt.ArrayLike, cell: Cell) -> npt.NDArray:
    if not cell.is_orthorhombic():
        raise NotOrthorhombicError(cell)
    return xyz - cell.lengths * np.round(xyz / cell.lengths)


def compute_pairwise_distances_in_frame(
    xyz1: npt.ArrayLike,
    xyz2: npt.ArrayLike,
) -> npt.NDArray:
    n_atoms1 = xyz1.shape[0]
    n_atoms2 = xyz2.shape[0]

    distances = np.zeros((n_atoms1, n_atoms2, 3))

    for i in range(3):
        distances[:, :, i] = np.subtract.outer(xyz1[:, i], xyz2[:, i])

    return distances


def compute_pairwise_distance_vectors(
    xyz1: npt.ArrayLike,
    xyz2: npt.ArrayLike,
    cell: Cell | None = None,
) -> npt.NDArray:
    if xyz1.shape[0] != xyz2.shape[0]:
        raise ValueError("Number of frames in `xyz1` do not match `xyz2`.")

    n_frames = xyz1.shape[0]
    n_atoms1 = xyz1.shape[1]
    n_atoms2 = xyz2.shape[1]

    distance_vectors = np.zeros((n_frames, n_atoms1, n_atoms2, 3))

    for i, (_xyz1, _xyz2) in enumerate(zip(xyz1, xyz2)):
        distance_vectors[i] = compute_pairwise_distances_in_frame(_xyz1, _xyz2)

    if cell is not None:
        distance_vectors = wrap(distance_vectors, cell)

    return distance_vectors


def compute_pairwise_distances(
    xyz1: npt.ArrayLike,
    xyz2: npt.ArrayLike,
    cell: Cell | None = None,
) -> npt.NDArray:
    distance_vectors = compute_pairwise_distance_vectors(xyz1, xyz2, cell)
    return np.linalg.norm(distance_vectors, axis=-1)


def is_symmetric(arr, rtol=1e-05, atol=1e-08):
    if arr.ndim > 2:
        return False

    if arr.shape[0] != arr.shape[1]:
        return False

    return np.allclose(arr, arr.T, rtol=rtol, atol=atol)


def compute_radial_distribution(
    distances: npt.ArrayLike,
    volume: float | Sequence[float],
    bins: int = 50,
    r_range: tuple[int] = (0, 10),
):
    """Compute the radial distribution function for NVT or NPT simulations."""
    distances = np.asarray(distances)
    n_frames = distances.shape[0]

    if is_symmetric(distances[0]):
        i, j = np.triu_indices(distances.shape[1], k=1)
        distance_pairs = distances[:, i, j]
    else:
        distance_pairs = distances.reshape(n_frames, -1)
        # (15000, 60, 240) -> (15000, n_pairs)

    num_pairs = distance_pairs.shape[1]
    r_edges = np.linspace(*r_range, num=bins + 1)
    dr = r_edges[1] - r_edges[0]

    if hasattr(volume, "__len__"):
        volume = np.reshape(volume, shape=(n_frames, -1))

    # Compute weights of each distance
    ref_number_density = num_pairs / volume
    weights = 1 / (4 * np.pi * distance_pairs**2 * dr) / ref_number_density

    weighted_dist, bin_edges = np.histogram(
        distance_pairs, bins=bins, range=r_range, weights=weights
    )

    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

    # normalize distribution by number of frames
    rdf = weighted_dist / n_frames

    return bin_centers, rdf
