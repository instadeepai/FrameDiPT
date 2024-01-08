"""Test experiment utils."""
import numpy as np
import pytest

from experiments import utils as exp_utils


@pytest.mark.parametrize(
    ("diffused_mask", "chain_index", "expected"),
    [
        (
            np.array([0, 1, 1, 0]),
            np.array([2, 2, 2, 2]),
            ([0], [1], [2]),
        ),
        (
            np.array([0, 0, 0, 0, 1, 1, 1]),
            np.array([0, 0, 0, 0, 2, 2, 2]),
            ([1], [0], [2]),
        ),
        (
            np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 0]),
            np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2]),
            ([0, 1, 2], [2, 0, 0], [3, 2, 1]),
        ),
        (
            np.array([0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0]),
            np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2]),
            ([0, 1, 1, 2], [2, 0, 2, 0], [3, 0, 3, 1]),
        ),
        (
            np.array([0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1]),
            np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3]),
            ([0, 0, 1, 3, 3], [1, 3, 2, 0, 3], [1, 3, 3, 1, 4]),
        ),
    ],
)
def test_get_diffused_region_per_chain(
    diffused_mask: np.ndarray,
    chain_index: np.ndarray,
    expected: tuple[list[int], list[int], list[int]],
) -> None:
    """Check the functionality of get_chain_diffused_region.

    Args:
        chain_index: array of chain indices of the protein structure, shape [N_res,].
        min_idx: minimum index of the diffused region.
        max_idx: maximum index of the diffused region.
        expected: expected results.
    """

    chains, start_indexes, end_indexes = exp_utils.get_diffused_region_per_chain(
        diffused_mask=diffused_mask,
        chain_index=chain_index,
    )

    assert chains == expected[0]
    assert start_indexes == expected[1]
    assert end_indexes == expected[2]
