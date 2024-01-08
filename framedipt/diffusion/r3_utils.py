"""Utils for r3 diffusion."""
from __future__ import annotations

import torch

from framedipt.data.utils import maybe_move_to_torch
from framedipt.tools.custom_type import TensorNDArray


def gaussian_log_prob(
    mu: TensorNDArray,
    std: TensorNDArray,
    x: TensorNDArray,
    diffuse_mask: TensorNDArray | None,
) -> float:
    """Calculate the log probability of a point in a Gaussian distribution.

    Args:
        mu: mean of the Gaussian distribution.
        std: standard deviation of the Gaussian distribution.
        x: data point for which the log probability is calculated.
        diffuse_mask: mask with 1 indicating the residue should be included.

    Returns:
        log_p: log probability of the given point in the Gaussian. l
    """
    # Move to torch if necessary.

    mu, std, x = (maybe_move_to_torch(x_i) for x_i in [mu, std, x])

    dist = torch.distributions.Normal(loc=mu, scale=std)
    log_p = dist.log_prob(x)

    if diffuse_mask is not None:
        mask_tensor = maybe_move_to_torch(diffuse_mask).bool()
        # Sum the log probabilities in the masked region
        log_p = torch.masked_select(log_p, mask_tensor[..., None]).cpu().numpy()

    # Sum the log probabilities.
    log_p = log_p.sum()

    return log_p
