"""Test evaluation utils."""
import numpy as np
import pytest

from evaluation.utils import metrics


# Examples copied from
# https://stackoverflow.com/questions/20305272/dihedral-
# torsion-angle-from-four-points-in-cartesian-coordinates-in-python
@pytest.mark.parametrize(
    ("a", "b", "c", "d", "expected"),
    [
        (
            np.array([24.969, 13.428, 30.692]),
            np.array([24.044, 12.661, 29.808]),
            np.array([22.785, 13.482, 29.543]),
            np.array([21.951, 13.670, 30.431]),
            -71.21515,
        ),
        (
            np.array([24.969, 13.428, 30.692]),
            np.array([24.044, 12.661, 29.808]),
            np.array([23.672, 11.328, 30.466]),
            np.array([22.881, 10.326, 29.620]),
            -171.94319,
        ),
        (
            np.array([24.044, 12.661, 29.808]),
            np.array([23.672, 11.328, 30.466]),
            np.array([22.881, 10.326, 29.620]),
            np.array([23.691, 9.935, 28.389]),
            60.82226,
        ),
        (
            np.array([24.044, 12.661, 29.808]),
            np.array([23.672, 11.328, 30.466]),
            np.array([22.881, 10.326, 29.620]),
            np.array([22.557, 9.096, 30.459]),
            -177.63641,
        ),
    ],
)
def test_dihedrals(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    expected: float,
) -> None:
    """Check the functionality of dihedral angle calculation.

    Args:
        a: 3D coordinates of point a.
        b: 3D coordinates of point b.
        c: 3D coordinates of point c.
        d: 3D coordinates of point d.
        expected: expected dihedral angle in degree.
    """
    dihedral_angle = metrics.dihedrals(
        a=a,
        b=b,
        c=c,
        d=d,
    )

    assert abs(np.rad2deg(dihedral_angle) - expected) < 1e-4
