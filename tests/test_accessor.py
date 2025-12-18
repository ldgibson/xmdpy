import numpy as np
import pytest
import xarray as xr
from xarray.testing import assert_allclose

from xmdpy.accessor import TrajectoryAccessor


@pytest.fixture()
def mock_traj_dataset() -> xr.Dataset:
    xyz_data = np.array(
        [
            [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [-1.5, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [-1.5, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [-1.5, 0.0, 0.0]],
        ]
    )
    cell_data = np.broadcast_to(np.eye(3, dtype="float64") * 10.0, (5, 3, 3))

    return xr.Dataset(
        data_vars={
            "xyz": (["time", "atom_id", "xyz_dim"], xyz_data),
            "cell": (["time", "cell_vector", "xyz_dim"], cell_data),
        },
        coords={
            "time": ("time", np.arange(5)),
            "atom_id": ("atom_id", np.arange(3)),
            "atoms": ("atom_id", list("COO")),
            "xyz_dim": ("xyz_dim", list("xyz")),
            "cell_vector": ("cell_vector", list("ABC")),
        },
    ).set_xindex("atoms")


@pytest.fixture()
def expected_rdf_result() -> xr.DataArray:
    r = np.array([0.3, 0.9, 1.5, 2.1, 2.7])
    rdf = np.array([0.0, 32.74793068, 35.36776513, 6.01492604, 0.0])
    return xr.DataArray(rdf, coords={"r": r}, dims="r")


@pytest.fixture
def mock_accessor(mock_traj_dataset: xr.Dataset) -> TrajectoryAccessor:
    return TrajectoryAccessor(mock_traj_dataset)


@pytest.mark.xfail
def test_contains_atoms_selection() -> None:
    ...
    # (atoms: str | Sequence[str] | int | Sequence[int] | slice, obj: Dataset | DataArray) -> bool:


@pytest.mark.xfail
def test_initialization_without_cell() -> None: ...


@pytest.mark.xfail
def test_initialization_with_cell() -> None: ...


@pytest.mark.xfail
def test_set_cell() -> None: ...


@pytest.mark.xfail
def test_atom_sel() -> None: ...


@pytest.mark.xfail
def test_get_atom_selections() -> None: ...


@pytest.mark.xfail
def test_get_distances() -> None: ...


def test_compute_rdf(mock_accessor, expected_rdf_result) -> None:
    result = mock_accessor.compute_rdf("C", "O", bins=5, r_range=(0, 3))
    assert_allclose(result, expected_rdf_result)
