import io
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

    fs.create_file(
        "./tests/data/mock_XDATCAR",
        contents=MOCK_XDATCAR_TRAJ,
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
    frames: np.ndarray,
    atoms: np.ndarray,
    xyz_dim: np.ndarray,
    traj_bytes: bytes,
    expected: np.ndarray,
) -> None:
    with io.BytesIO(traj_bytes) as f:
        result = read_xdatcar_frames(
            f,
            indexes=(frames, atoms, xyz_dim),
            total_atoms=3,
            dtype=np.float64,
        )

    np.testing.assert_allclose(result, expected[0][frames][:, atoms][:, :, xyz_dim])


@pytest.mark.parametrize(
    "frames", [np.arange(10), np.arange(0, 10, 2), np.arange(1, 8, 3)]
)
@pytest.mark.parametrize("atoms", [np.arange(3), np.array([0, 1]), np.array([2])])
@pytest.mark.parametrize("xyz_dim", [np.arange(3), np.array([0, 1]), np.array([2])])
def test_read_xdatcar_frames_slice_frames_atoms_xyzdim_cartesian(
    frames: np.ndarray,
    atoms: np.ndarray,
    xyz_dim: np.ndarray,
    traj_bytes_cartesian: bytes,
    expected: np.ndarray,
) -> None:
    with io.BytesIO(traj_bytes_cartesian) as f:
        result = read_xdatcar_frames(
            f,
            indexes=(frames, atoms, xyz_dim),
            total_atoms=3,
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
    frames: np.ndarray,
    atoms: np.ndarray,
    xyz_dim: np.ndarray,
    traj_bytes_selective_dynamics: bytes,
    expected: np.ndarray,
) -> None:
    with io.BytesIO(traj_bytes_selective_dynamics) as f:
        result = read_xdatcar_frames(
            f,
            indexes=(frames, atoms, xyz_dim),
            total_atoms=3,
            dtype=np.float64,
            selective_dynamics=True,
        )

    np.testing.assert_allclose(result, expected[0][frames][:, atoms][:, :, xyz_dim])


def test_read_xdatcar_frames_frames_not_array_raises_error(
    traj_bytes: bytes,
) -> None:
    frames = np.arange(10).tolist()
    atoms = np.arange(3)
    xyz_dim = np.arange(3)

    with pytest.raises(TypeError):
        with io.BytesIO(traj_bytes) as f:
            _ = read_xdatcar_frames(
                f,
                indexes=(frames, atoms, xyz_dim),  # type: ignore
                total_atoms=3,
                dtype=np.float64,
            )


def test_read_xdatcar_frames_atoms_not_array_raises_error(
    traj_bytes: bytes,
) -> None:
    frames = np.arange(10)
    atoms = np.arange(3).tolist()
    xyz_dim = np.arange(3)

    with pytest.raises(TypeError):
        with io.BytesIO(traj_bytes) as f:
            _ = read_xdatcar_frames(
                f,
                indexes=(frames, atoms, xyz_dim),  # type: ignore
                total_atoms=3,
                dtype=np.float64,
            )


def test_read_xdatcar_frames_xyzdim_not_array_raises_error(
    traj_bytes: bytes,
) -> None:
    frames = np.arange(10)
    atoms = np.arange(3)
    xyz_dim = np.arange(3).tolist()

    with pytest.raises(TypeError):
        with io.BytesIO(traj_bytes) as f:
            _ = read_xdatcar_frames(
                f,
                indexes=(frames, atoms, xyz_dim),  # type: ignore
                total_atoms=3,
                dtype=np.float64,
            )
