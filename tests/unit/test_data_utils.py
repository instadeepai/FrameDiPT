"""Test data processing utils."""
import numpy as np
import pytest
from pytest_mock import MockFixture

from framedipt.data import utils as data_utils


@pytest.mark.parametrize(
    ("index", "expected"),
    [(0, "A"), (25, "Z"), (26 + 1, "AB"), (26 * 26, "ZA"), (26 * 27, "AAA")],
)
def test_map_to_new_str_name(
    index: int,
    expected: str,
) -> None:
    """Check the functionality of map index to string name.

    Args:
        index: integer of index.
        expected: expected string name.
    """
    name = data_utils.map_to_new_str_name(index=index)

    assert name == expected


@pytest.mark.parametrize(
    ("chain_idx", "res_mask", "length", "start_idx", "expected_mask"),
    [
        (
            np.array(
                [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2]
            ),  # three chains of length 6, 5 and 4
            np.full((15,), True),
            3,
            1,
            np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1]),
        ),
        (
            np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),  # two chains of length 6, 5
            np.full((11,), True),
            2,
            0,
            np.array([1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0]),
        ),
        (
            np.array([0, 0, 0, 0, 0, 0]),  # one chain of length 6
            np.full((6,), True),
            4,
            1,
            np.array([0, 1, 1, 1, 1, 0]),
        ),
    ],
)
def test_create_redacted_regions(
    chain_idx: np.ndarray,
    res_mask: np.ndarray,
    length: int,
    start_idx: int,
    expected_mask: np.ndarray,
    mocker: MockFixture,
) -> None:
    """Test create_redacted_regions function.

    Args:
        chain_idx: chain indices, shape [N_res].
        res_mask: residue mask, shape [N_res,].
        length: Length of the redacted region.
        start_idx: The start index of the redacted region.
        expected_mask: Expected diffused mask.
        mocker: Fixture which allows to fix the returned value of a random function.
    """
    redact_min_len = 2
    redact_max_len = 5
    mocker_rng = mocker.patch("numpy.random.default_rng")
    mocker_rng.return_value.integers.side_effect = [
        length,
        start_idx,  # start_idx for first chain
        length,
        start_idx,  # start_idx for the second chain
        length,
        start_idx,  # start_idx for the third chain
    ]
    diff_mask = data_utils.create_redacted_regions(
        chain_idx=chain_idx,
        res_mask=res_mask,
        rng=mocker_rng(),
        redact_min_len=redact_min_len,
        redact_max_len=redact_max_len,
    )
    np.testing.assert_array_equal(diff_mask, expected_mask)
