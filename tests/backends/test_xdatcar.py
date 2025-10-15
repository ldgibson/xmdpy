import os

import dask
import numpy as np
import pytest
from numpy.testing import assert_allclose

from xmdpy.backends.xdatcar import get_xdatcar_dims_and_details, read_xdatcar_frames


@pytest.fixture
def fake_filesystem(fs):
    dask_real_path = os.path.dirname(dask.__file__)
    fs.add_real_directory(dask_real_path)

    MOCK_XDATCAR_HEADER = (
        "COMMENT\n1.0\n10.0 0.0 0.0\n0.0 10.0 0.0\n0.0 0.0 10.0\nO H\n1 2\n"
    )
    MOCK_XDATCAR_FRAME = (
        "Direct configuration num= {0}\n"
        "0.00 0.00 {1:.2f}\n"
        "0.09 0.07 {1:.2f}\n"
        "-0.09 0.07 {1:.2f}\n"
    )

    MOCK_XDATCAR_TRAJ = MOCK_XDATCAR_HEADER + "".join(
        [MOCK_XDATCAR_FRAME.format(i, i / 10.0) for i in range(10)]
    )

    MOCK_XDATCAR_SELECTIVE_DYNAMICS_FRAME = (
        "Selective dynamics\n"
        "Direct configuration num= {0}\n"
        "0.00 0.00 {1:.2f} T T T\n"
        "0.09 0.07 {1:.2f} T T T\n"
        "-0.09 0.07 {1:.2f} T T T\n"
    )

    MOCK_XDATCAR_SELECTIVE_DYNAMICS_TRAJ = MOCK_XDATCAR_HEADER + "".join(
        [MOCK_XDATCAR_SELECTIVE_DYNAMICS_FRAME.format(i, i / 10.0) for i in range(10)]
    )

    MOCK_XDATCAR_VARIABLE_CELL_TRAJ = "".join(
        [
            MOCK_XDATCAR_HEADER + MOCK_XDATCAR_FRAME.format(i, i / 10.0)
            for i in range(10)
        ]
    )

    MOCK_CARTESIAN_XDATCAR_FRAME = (
        "Cartesian configuration num= {0}\n"
        "0.0 0.0 {1:.2f}\n"
        "0.9 0.7 {1:.2f}\n"
        "-0.9 0.7 {1:.2f}\n"
    )

    MOCK_XDATCAR_CARTESIAN_TRAJ = MOCK_XDATCAR_HEADER + "".join(
        [MOCK_CARTESIAN_XDATCAR_FRAME.format(i, i) for i in range(10)]
    )

    fs.create_file(
        "./tests/data/mock_XDATCAR",
        contents=MOCK_XDATCAR_TRAJ,
    )
    fs.create_file(
        "./tests/data/mock_cartesian_XDATCAR",
        contents=MOCK_XDATCAR_CARTESIAN_TRAJ,
    )
    fs.create_file(
        "./tests/data/mock_selective_dynamics_XDATCAR",
        contents=MOCK_XDATCAR_SELECTIVE_DYNAMICS_TRAJ,
    )
    fs.create_file(
        "./tests/data/mock_variable_cell_XDATCAR",
        contents=MOCK_XDATCAR_VARIABLE_CELL_TRAJ,
    )
    assert os.path.exists("./tests/data/mock_XDATCAR")
    assert os.path.exists("./tests/data/mock_cartesian_XDATCAR")
    assert os.path.exists("./tests/data/mock_variable_cell_XDATCAR")
    assert os.path.exists("./tests/data/mock_selective_dynamics_XDATCAR")
    yield fs


@pytest.fixture
def traj_bytes() -> bytes:
    MOCK_XDATCAR_HEADER = (
        "COMMENT\n1.0\n10.0 0.0 0.0\n0.0 10.0 0.0\n0.0 0.0 10.0\nO H\n1 2\n"
    )
    MOCK_XDATCAR_FRAME = (
        "Direct configuration num= {0}\n"
        "0.00 0.00 {1:.2f}\n"
        "0.09 0.07 {1:.2f}\n"
        "-0.09 0.07 {1:.2f}\n"
    )

    MOCK_XDATCAR_TRAJ = MOCK_XDATCAR_HEADER + "".join(
        [MOCK_XDATCAR_FRAME.format(i, i / 10.0) for i in range(10)]
    )
    return MOCK_XDATCAR_TRAJ.encode()


@pytest.fixture
def traj_bytes_cartesian() -> bytes:
    MOCK_XDATCAR_HEADER = (
        "COMMENT\n1.0\n10.0 0.0 0.0\n0.0 10.0 0.0\n0.0 0.0 10.0\nO H\n1 2\n"
    )
    MOCK_XDATCAR_FRAME = (
        "Cartesian configuration num= {0}\n"
        "0.0 0.0 {1:.2f}\n"
        "0.9 0.7 {1:.2f}\n"
        "-0.9 0.7 {1:.2f}\n"
    )

    MOCK_XDATCAR_TRAJ = MOCK_XDATCAR_HEADER + "".join(
        [MOCK_XDATCAR_FRAME.format(i, i) for i in range(10)]
    )
    return MOCK_XDATCAR_TRAJ.encode()


@pytest.fixture
def traj_bytes_selective_dynamics() -> bytes:
    MOCK_XDATCAR_HEADER = (
        "COMMENT\n1.0\n10.0 0.0 0.0\n0.0 10.0 0.0\n0.0 0.0 10.0\nO H\n1 2\n"
    )
    MOCK_XDATCAR_FRAME = (
        "Selective dynamics\n"
        "Direct configuration num= {0}\n"
        "0.00 0.00 {1:.2f} T T T\n"
        "0.09 0.07 {1:.2f} T T T\n"
        "-0.09 0.07 {1:.2f} T T T\n"
    )

    MOCK_XDATCAR_TRAJ = MOCK_XDATCAR_HEADER + "".join(
        [MOCK_XDATCAR_FRAME.format(i, i / 10.0) for i in range(10)]
    )
    return MOCK_XDATCAR_TRAJ.encode()


@pytest.fixture
def direct_xdatcar() -> str:
    return "./tests/data/mock_XDATCAR"


@pytest.fixture
def cartesian_xdatcar() -> str:
    return "./tests/data/mock_cartesian_XDATCAR"


@pytest.fixture
def selective_dynamics_xdatcar() -> str:
    return "./tests/data/mock_selective_dynamics_XDATCAR"


@pytest.fixture
def variable_cell_xdatcar() -> str:
    return "./tests/data/mock_variable_cell_XDATCAR"


@pytest.fixture
def cell() -> np.ndarray:
    return np.eye(3, dtype="float64") * 10.0


@pytest.fixture
def expected() -> tuple[np.ndarray, np.ndarray]:
    xyz_data = np.stack(
        [np.array([[0.0, 0.0, i], [0.9, 0.7, i], [-0.9, 0.7, i]]) for i in range(10)]
    ).astype("float64")

    cell_data = np.broadcast_to(np.eye(3, dtype="float64") * 10.0, (10, 3, 3))

    return xyz_data, cell_data


def test_get_xdatcar_dims_and_details_constant_cell(fake_filesystem, expected) -> None:
    n_frames, atoms, cell, variable_cell = get_xdatcar_dims_and_details(
        "./tests/data/mock_XDATCAR",
    )
    assert n_frames == 10
    assert atoms == ["O", "H", "H"]
    assert_allclose(cell, expected[1][0])
    assert not variable_cell


def test_get_xdatcar_dims_and_details_variable_cell(fake_filesystem, expected) -> None:
    n_frames, atoms, cell, variable_cell = get_xdatcar_dims_and_details(
        "./tests/data/mock_variable_cell_XDATCAR",
    )
    assert n_frames == 10
    assert atoms == ["O", "H", "H"]
    assert_allclose(cell, expected[1][0])
    assert variable_cell


@pytest.mark.parametrize(
    "frames", [np.arange(10), np.arange(0, 10, 2), np.arange(1, 8, 3)]
)
@pytest.mark.parametrize("atoms", [np.arange(3), np.array([0, 1]), np.array([2])])
@pytest.mark.parametrize("xyz_dim", [np.arange(3), np.array([0, 1]), np.array([2])])
def test_read_xdatcar_frames_slice_frames_atoms_xyzdim(
    fake_filesystem,
    frames: np.ndarray,
    atoms: np.ndarray,
    xyz_dim: np.ndarray,
    direct_xdatcar: str,
    cell: np.ndarray,
    expected: np.ndarray,
) -> None:
    result = read_xdatcar_frames(
        frames,
        atoms,
        xyz_dim,
        direct_xdatcar,
        total_atoms=3,
        cell=cell,
        dtype=np.float64,
    )

    np.testing.assert_allclose(result, expected[0][frames][:, atoms][:, :, xyz_dim])


@pytest.mark.parametrize(
    "frames", [np.arange(10), np.arange(0, 10, 2), np.arange(1, 8, 3)]
)
@pytest.mark.parametrize("atoms", [np.arange(3), np.array([0, 1]), np.array([2])])
@pytest.mark.parametrize("xyz_dim", [np.arange(3), np.array([0, 1]), np.array([2])])
def test_read_xdatcar_frames_slice_frames_atoms_xyzdim_cartesian(
    fake_filesystem,
    frames: np.ndarray,
    atoms: np.ndarray,
    xyz_dim: np.ndarray,
    cell: np.ndarray,
    cartesian_xdatcar: str,
    expected: np.ndarray,
) -> None:
    result = read_xdatcar_frames(
        frames,
        atoms,
        xyz_dim,
        cartesian_xdatcar,
        total_atoms=3,
        cell=cell,
        dtype=np.float64,
        direct=False,
    )

    np.testing.assert_allclose(result, expected[0][frames][:, atoms][:, :, xyz_dim])


@pytest.mark.parametrize(
    "frames", [np.arange(10), np.arange(0, 10, 2), np.arange(1, 8, 3)]
)
@pytest.mark.parametrize("atoms", [np.arange(3), np.array([0, 1]), np.array([2])])
@pytest.mark.parametrize("xyz_dim", [np.arange(3), np.array([0, 1]), np.array([2])])
def test_read_xdatcar_frames_slice_frames_atoms_xyzdim_selective_dynamics(
    fake_filesystem,
    frames: np.ndarray,
    atoms: np.ndarray,
    xyz_dim: np.ndarray,
    cell: np.ndarray,
    selective_dynamics_xdatcar: str,
    expected: np.ndarray,
) -> None:
    result = read_xdatcar_frames(
        frames,
        atoms,
        xyz_dim,
        selective_dynamics_xdatcar,
        total_atoms=3,
        cell=cell,
        dtype=np.float64,
        selective_dynamics=True,
    )

    np.testing.assert_allclose(result, expected[0][frames][:, atoms][:, :, xyz_dim])


def test_read_xdatcar_frames_frames_not_array_raises_error(
    fake_filesystem,
    direct_xdatcar: str,
    cell: np.ndarray,
) -> None:
    frames = np.arange(10).tolist()
    atoms = np.arange(3)
    xyz_dim = np.arange(3)

    with pytest.raises(TypeError):
        _ = read_xdatcar_frames(
            frames,  # type: ignore
            atoms,
            xyz_dim,
            direct_xdatcar,
            total_atoms=3,
            cell=cell,
            dtype=np.float64,
        )


def test_read_xdatcar_frames_atoms_not_array_raises_error(
    fake_filesystem,
    direct_xdatcar: str,
    cell: np.ndarray,
) -> None:
    frames = np.arange(10)
    atoms = np.arange(3).tolist()
    xyz_dim = np.arange(3)

    with pytest.raises(TypeError):
        _ = read_xdatcar_frames(
            frames,
            atoms,  # type: ignore
            xyz_dim,
            direct_xdatcar,
            total_atoms=3,
            cell=cell,
            dtype=np.float64,
        )


def test_read_xdatcar_frames_xyzdim_not_array_raises_error(
    fake_filesystem,
    direct_xdatcar: str,
    cell: np.ndarray,
) -> None:
    frames = np.arange(10)
    atoms = np.arange(3)
    xyz_dim = np.arange(3).tolist()

    with pytest.raises(TypeError):
        _ = read_xdatcar_frames(
            frames,
            atoms,
            xyz_dim,  # type: ignore
            direct_xdatcar,
            total_atoms=3,
            cell=cell,
            dtype=np.float64,
        )
