import pytest
import xarray as xr

from xmdpy.accessor import TrajectoryAccessor


@pytest.fixture
def mock_accessor() -> TrajectoryAccessor:
    dataset = xr.Dataset()
    return TrajectoryAccessor(dataset)


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


@pytest.mark.xfail
def test_compute_rdf(mock_accessor) -> None: ...
