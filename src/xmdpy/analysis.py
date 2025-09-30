import warnings
from typing import Any

import dask
import dask.array as da
import numpy as np
import numpy.typing as npt

from xmdpy.types import FloatLike, TrajArray


def wrap(xyz: npt.NDArray[Any], cell_lengths: npt.NDArray[Any]) -> npt.NDArray[Any]:
    if xyz.ndim > 2:
        cell_lengths = np.expand_dims(cell_lengths, axis=tuple(range(xyz.ndim - 2)))

    return xyz - cell_lengths * np.round(xyz / cell_lengths)


def wrap_trajectory(
    xyz: np.ndarray[tuple[int, int, int]], cell_lengths: np.ndarray[tuple[int, int]]
) -> np.ndarray[tuple[int, int, int]]:
    return np.stack(
        [
            wrap(xyz_i, cell_lengths_i)
            for xyz_i, cell_lengths_i in zip(xyz, cell_lengths)
        ]
    )


def compute_distance_vectors(
    xyz1: np.ndarray[tuple[int, int]],
    xyz2: np.ndarray[tuple[int, int]],
    cell_lengths: np.ndarray[tuple[int]] | None = None,
) -> np.ndarray[tuple[int, int]]:
    distances = np.stack(
        [np.subtract.outer(xyz1[:, i], xyz2[:, i]) for i in range(3)], axis=-1
    )

    if cell_lengths is not None:
        distances = wrap(distances, cell_lengths)

    return distances


def compute_distances(
    xyz1: np.ndarray[tuple[int, int]],
    xyz2: np.ndarray[tuple[int, int]],
    cell_lengths: np.ndarray[tuple[int]] | None = None,
) -> np.ndarray[tuple[int]]:
    distance_vectors = compute_distance_vectors(xyz1, xyz2, cell_lengths)
    return np.linalg.norm(distance_vectors, axis=-1)


def compute_pairwise_distances_in_frame(
    xyz1: npt.NDArray[FloatLike],
    xyz2: npt.NDArray[FloatLike],
) -> np.ndarray[tuple[int, int, int], np.dtype[FloatLike]]:
    n_atoms1 = xyz1.shape[0]
    n_atoms2 = xyz2.shape[0]

    distances = np.zeros((n_atoms1, n_atoms2, 3))

    for i in range(3):
        distances[:, :, i] = np.subtract.outer(xyz1[:, i], xyz2[:, i])

    return distances


def compute_pairwise_distance_vectors(
    xyz1: TrajArray,
    xyz2: TrajArray,
    cell: np.ndarray[tuple[int, int]] | None = None,
) -> np.ndarray[tuple[int, int, int, int], np.dtype[FloatLike]]:
    if xyz1.shape[0] != xyz2.shape[0]:
        raise ValueError("Number of frames in `xyz1` do not match `xyz2`.")

    n_frames = xyz1.shape[0]
    n_atoms1 = xyz1.shape[1]
    n_atoms2 = xyz2.shape[1]

    distance_vectors = np.zeros((n_frames, n_atoms1, n_atoms2, 3))

    for i, (_xyz1, _xyz2) in enumerate(zip(xyz1, xyz2)):
        distance_vectors[i] = compute_pairwise_distances_in_frame(_xyz1, _xyz2)

        if cell is not None:
            distance_vectors[i] = wrap(distance_vectors[i], cell[i])

    return distance_vectors


def compute_pairwise_distances(
    xyz1: TrajArray,
    xyz2: TrajArray,
    cell: np.ndarray[tuple[int, int]] | None = None,
) -> np.ndarray[tuple[int, int, int], np.dtype[FloatLike]]:
    distance_vectors = compute_pairwise_distance_vectors(xyz1, xyz2, cell)
    return np.linalg.norm(distance_vectors, axis=-1)


def is_symmetric(
    arr: npt.NDArray[Any], rtol: float = 1e-05, atol: float = 1e-08
) -> bool:
    if arr.ndim > 2:
        return False

    if arr.shape[0] != arr.shape[1]:
        return False

    return np.allclose(arr, arr.T, rtol=rtol, atol=atol)


def lazy_take(
    values: npt.NDArray[FloatLike], indices: npt.NDArray[FloatLike], mode: str = "clip"
) -> npt.NDArray[FloatLike]:
    if dask.is_dask_collection(values) or dask.is_dask_collection(indices):
        return da.map_blocks(
            np.take,
            values,
            indices,
            mode=mode,
            dtype=values.dtype,  # type: ignore
        )
    else:
        return np.take(values, indices, mode=mode)  # type: ignore


def get_radial_weights(
    distances: np.ndarray[tuple[int, int], np.dtype[FloatLike]],
    n_pairs: int,
    n_frames: int,
    volume: np.ndarray[tuple[int, ...], np.dtype[FloatLike]],
    r_bins: np.ndarray[tuple[int], np.dtype[FloatLike]],
) -> np.ndarray[tuple[int, int], np.dtype[FloatLike]]:
    """Compute weights for each distance value"""
    dr = np.diff(r_bins)
    r_mid = (r_bins[1:] + r_bins[:-1]) / 2

    # Overall number density per frame
    ref_number_density = n_pairs / volume.reshape(n_frames, 1)

    ### For each frame, get closest shell volume for each distance value
    bin_idx = np.digitize(distances, r_bins)

    # If distances is a dask array, then lazy_take uses da.map_blocks(),
    # otherwise, it uses np.take()
    nearest_r_bin = lazy_take(r_mid, bin_idx - 1, mode="clip")
    nearest_dr = lazy_take(dr, bin_idx - 1, mode="clip")
    shell_volumes = 4 * np.pi * nearest_r_bin**2 * nearest_dr

    return 1 / (shell_volumes * ref_number_density * float(n_frames))


def construct_bins(
    bins: int | npt.ArrayLike, range: tuple[float, float]
) -> np.ndarray[tuple[int], np.dtype[FloatLike]]:
    if isinstance(bins, int):
        bins = np.linspace(*range, num=bins + 1)
    else:
        bins = np.asarray(bins)

        if np.any(bins < range[0]) or np.any(bins > range[1]):
            warnings.warn(f"bins extend beyond provided range: {range}")
    return bins


def compute_radial_distribution(
    distances: np.ndarray[tuple[int, int, int]],
    n_pairs: int,
    n_frames: int,
    volume: float | npt.ArrayLike,
    bins: int | npt.ArrayLike = 50,
    r_range: tuple[float, float] = (0, 10),
) -> tuple[
    np.ndarray[tuple[int], np.dtype[FloatLike]],
    np.ndarray[tuple[int], np.dtype[FloatLike]],
]:
    """Compute the radial distribution function for NVT or NPT simulations."""
    if not dask.is_dask_collection(distances):
        distances = np.asarray(distances)

    if not dask.is_dask_collection(volume):
        volume = np.asarray(volume)

    try:
        pair_distances_per_frame = distances.reshape(n_frames, n_pairs)

    except ValueError as e:
        raise ValueError(
            f"cannot reshape distances to ({n_frames=}, {n_pairs=}).\n{repr(e)}"
        )

    r_bins = construct_bins(bins, r_range)

    weights = get_radial_weights(
        pair_distances_per_frame,
        n_pairs,
        n_frames,
        volume,  # type: ignore
        r_bins,
    )

    rdf, _ = np.histogram(
        pair_distances_per_frame, bins=r_bins, range=r_range, weights=weights
    )

    # Value of each bin center, has same length as rdf
    r = (r_bins[1:] + r_bins[:-1]) / 2

    return r, rdf
