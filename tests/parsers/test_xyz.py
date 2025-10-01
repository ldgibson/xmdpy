import io
import os

import numpy as np
import pytest

from xmdpy.parsers.xyz import get_xyz_dims_and_details, read_xyz_frames


@pytest.fixture
def fake_filesystem(fs):
    MOCK_XYZ_FRAME = (
        "3\ni = {0}\nO 0.0 0.0 {0:.1f}\nH 0.9 0.7 {0:.1f}\nH -0.9 0.7 {0:.1f}\n"
    )
    MOCK_XYZ_TRAJ = "".join([MOCK_XYZ_FRAME.format(i) for i in range(10)])

    fs.create_file(
        "./tests/data/mock_traj.xyz",
        contents=MOCK_XYZ_TRAJ,
    )

    assert os.path.exists("./tests/data/mock_traj.xyz")
    yield fs


@pytest.fixture
def traj_bytes() -> bytes:
    MOCK_XYZ_FRAME = (
        "3\ni = {0}\nO 0.0 0.0 {0:.1f}\nH 0.9 0.7 {0:.1f}\nH -0.9 0.7 {0:.1f}\n"
    )
    MOCK_XYZ_TRAJ = "".join([MOCK_XYZ_FRAME.format(i) for i in range(10)])
    return MOCK_XYZ_TRAJ.encode()


@pytest.fixture
def expected() -> np.ndarray:
    return np.stack(
        [np.array([[0.0, 0.0, i], [0.9, 0.7, i], [-0.9, 0.7, i]]) for i in range(10)]
    )


def test_get_xyz_dims_and_details(fake_filesystem) -> None:
    n_frames, atoms, cell, variable_cell = get_xyz_dims_and_details(
        "./tests/data/mock_traj.xyz",
    )
    assert n_frames == 10
    assert atoms == ["O", "H", "H"]
    assert cell is None
    assert not variable_cell


@pytest.mark.parametrize(
    "frames", [np.arange(10), np.arange(0, 10, 2), np.arange(1, 8, 3)]
)
@pytest.mark.parametrize("atoms", [np.arange(3), np.array([0, 1]), np.array([2])])
@pytest.mark.parametrize("xyz_dim", [np.arange(3), np.array([0, 1]), np.array([2])])
def test_read_xyz_frames_slice_frames_atoms_xyzdim(
    frames: np.ndarray,
    atoms: np.ndarray,
    xyz_dim: np.ndarray,
    traj_bytes: bytes,
    expected: np.ndarray,
) -> None:
    with io.BytesIO(traj_bytes) as f:
        result = read_xyz_frames(
            f,
            indexes=(frames, atoms, xyz_dim),
            total_atoms=3,
            dtype=np.float64,
        )

    np.testing.assert_allclose(result, expected[frames][:, atoms][:, :, xyz_dim])


def test_read_xyz_frames_frames_not_array_raises_error(
    traj_bytes: bytes,
) -> None:
    frames = np.arange(10).tolist()
    atoms = np.arange(3)
    xyz_dim = np.arange(3)

    with pytest.raises(TypeError):
        with io.BytesIO(traj_bytes) as f:
            _ = read_xyz_frames(
                f,
                indexes=(frames, atoms, xyz_dim),  # type: ignore
                total_atoms=3,
                dtype=np.float64,
            )


def test_read_xyz_frames_atoms_not_array_raises_error(
    traj_bytes: bytes,
) -> None:
    frames = np.arange(10)
    atoms = np.arange(3).tolist()
    xyz_dim = np.arange(3)

    with pytest.raises(TypeError):
        with io.BytesIO(traj_bytes) as f:
            _ = read_xyz_frames(
                f,
                indexes=(frames, atoms, xyz_dim),  # type: ignore
                total_atoms=3,
                dtype=np.float64,
            )


def test_read_xyz_frames_xyzdim_not_array_raises_error(
    traj_bytes: bytes,
) -> None:
    frames = np.arange(10)
    atoms = np.arange(3)
    xyz_dim = np.arange(3).tolist()

    with pytest.raises(TypeError):
        with io.BytesIO(traj_bytes) as f:
            _ = read_xyz_frames(
                f,
                indexes=(frames, atoms, xyz_dim),  # type: ignore
                total_atoms=3,
                dtype=np.float64,
            )
