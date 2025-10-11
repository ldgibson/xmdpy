import os

import numpy as np
import pytest

from xmdpy.backends.xyz import get_xyz_dims_and_details, read_xyz_frames


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
def filename() -> str:
    return "./tests/data/mock_traj.xyz"


@pytest.fixture
def expected() -> np.ndarray:
    return np.stack(
        [np.array([[0.0, 0.0, i], [0.9, 0.7, i], [-0.9, 0.7, i]]) for i in range(10)]
    )


def test_get_xyz_dims_and_details(fake_filesystem, filename: str) -> None:
    n_frames, atoms = get_xyz_dims_and_details(filename)
    assert n_frames == 10
    assert atoms == ["O", "H", "H"]


@pytest.mark.parametrize(
    "frames", [np.arange(10), np.arange(0, 10, 2), np.arange(1, 8, 3)]
)
@pytest.mark.parametrize("atoms", [np.arange(3), np.array([0, 1]), np.array([2])])
@pytest.mark.parametrize("xyz_dim", [np.arange(3), np.array([0, 1]), np.array([2])])
def test_read_xyz_frames_slice_frames_atoms_xyzdim(
    fake_filesystem,
    filename: str,
    frames: np.ndarray,
    atoms: np.ndarray,
    xyz_dim: np.ndarray,
    expected: np.ndarray,
) -> None:
    result = read_xyz_frames(
        frames,
        atoms,
        xyz_dim,
        filename,
        total_atoms=3,
        dtype=np.float64,
    )

    np.testing.assert_allclose(result, expected[frames][:, atoms][:, :, xyz_dim])


def test_read_xyz_frames_frames_not_array_raises_error(
    fake_filesystem,
    filename: str,
) -> None:
    frames = np.arange(10).tolist()
    atoms = np.arange(3)
    xyz_dim = np.arange(3)

    with pytest.raises(TypeError):
        _ = read_xyz_frames(
            frames,  # type: ignore
            atoms,
            xyz_dim,
            filename,
            total_atoms=3,
            dtype=np.float64,
        )


def test_read_xyz_frames_atoms_not_array_raises_error(
    fake_filesystem,
    filename: str,
) -> None:
    frames = np.arange(10)
    atoms = np.arange(3).tolist()
    xyz_dim = np.arange(3)

    with pytest.raises(TypeError):
        _ = read_xyz_frames(
            frames,
            atoms,  # type: ignore
            xyz_dim,
            filename,
            total_atoms=3,
            dtype=np.float64,
        )


def test_read_xyz_frames_xyzdim_not_array_raises_error(
    fake_filesystem,
    filename: str,
) -> None:
    frames = np.arange(10)
    atoms = np.arange(3)
    xyz_dim = np.arange(3).tolist()

    with pytest.raises(TypeError):
        _ = read_xyz_frames(
            frames,
            atoms,
            xyz_dim,  # type: ignore
            filename,
            total_atoms=3,
            dtype=np.float64,
        )
