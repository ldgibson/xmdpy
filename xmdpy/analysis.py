import numpy as np


def pairwise_distances_in_frame(xyz1, xyz2, cell=None):
    n_atoms1 = xyz1.shape[0]
    n_atoms2 = xyz2.shape[0]

    distances = np.zeros((n_atoms1, n_atoms2, 3))

    for i in range(3):
        distances[:, :, i] = np.subtract.outer(xyz1[:, i], xyz2[:, i])

    if cell is not None:
        distances -= cell * np.round(distances / cell)

    return distances


def pairwise_distances(xyz1, xyz2, cell=None, vector=False):
    if xyz1.shape[0] != xyz2.shape[0]:
        raise ValueError("number of frames in `xyz1` do not match `xyz2`")

    n_frames = xyz1.shape[0]
    n_atoms1 = xyz1.shape[1]
    n_atoms2 = xyz2.shape[1]

    distances = np.zeros((n_frames, n_atoms1, n_atoms2, 3))

    for i, (_xyz1, _xyz2) in enumerate(zip(xyz1, xyz2)):
        distances[i] = pairwise_distances_in_frame(_xyz1, _xyz2, cell)

    if not vector:
        distances = np.linalg.norm(distances, axis=-1)
    return distances
