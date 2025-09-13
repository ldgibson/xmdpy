from xmdpy.cell import normalize_cell
import numpy as np

import pytest


@pytest.mark.parametrize(
    "cell",
    [
        4,
        [4, 4, 4],
        [[4, 4, 4] for _ in range(10)],
        np.array([4, 4, 4]),
        np.array([[4, 4, 4] for _ in range(10)]),
        np.array([[4, 0, 0], [0, 4, 0], [0, 0, 4]]),
    ],
)
def test_normalize_cell_arg_cell(cell):
    result = normalize_cell(cell, n_frames=10, dtype=np.int64)

    expected = np.broadcast_to(np.diag([4, 4, 4]).astype(np.int64), (10, 3, 3))

    np.testing.assert_array_equal(result, expected)
