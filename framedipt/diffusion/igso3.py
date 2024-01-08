"""Parts of this file were developed in conjunction with the DiffDock authors.
https://github.com/gcorso/DiffDock
"""
from __future__ import annotations

import os

import numpy as np
import numpy.typing as npt
import torch

from framedipt.data import utils as data_utils
from framedipt.diffusion import so3_utils
from framedipt.tools.custom_type import NDArrayFloat
from framedipt.tools.log import get_logger

logger = get_logger()


def f_igso3(omega: torch.Tensor, t: float, truncation_level: int = 500) -> torch.Tensor:
    """Truncated sum of IGSO(3) distribution.

    This function approximates the power series in equation 5 of
    "DENOISING DIFFUSION PROBABILISTIC MODELS ON SO(3) FOR ROTATIONAL
    ALIGNMENT"
    Leach et al. 2022

    This expression diverges from the expression in Leach in that here, sigma =
    sqrt(2) * eps, if eps_leach were the scale parameter of the IGSO(3).

    With this reparameterization, IGSO(3) agrees with the Brownian motion on
    SO(3) with t=sigma^2 when defined for the canonical inner product on SO3,
    <u, v>_SO3 = Trace(u v^T)/2

    Args:
        omega: i.e. the angle of rotation associated with rotation matrix
        t: variance parameter of IGSO(3), maps onto time in Brownian motion
        truncation_level: How many terms from infinite sum to use.
    """

    ls = torch.arange(truncation_level)  # of shape [1, truncation_level]
    approx = (
        (2 * ls + 1)
        * torch.exp(-ls * (ls + 1) * t / 2)
        * torch.sin(omega.unsqueeze(-1) * (ls + 1 / 2))
        / torch.sin(omega.unsqueeze(-1) / 2)
    ).sum(dim=-1)
    return approx


def d_logf_d_omega(
    omega: torch.Tensor | np.ndarray, t: float, truncation_level: int = 500
) -> torch.Tensor:
    omega = torch.tensor(omega, requires_grad=True)
    log_f = torch.log(f_igso3(omega, t, truncation_level))
    out = torch.autograd.grad(log_f.sum(), omega)[0]
    return out


# IGSO3 density with respect to the volume form on SO(3)
def igso3_density(
    rot_t: torch.Tensor, t: float, truncation_level: int = 500
) -> torch.Tensor:
    return f_igso3(so3_utils.omega(rot_t), t, truncation_level)


def igso3_density_angle(
    omega: np.ndarray, t: float, truncation_level: int = 500
) -> np.ndarray:
    return (
        f_igso3(torch.tensor(omega), t, truncation_level).numpy()
        * (1 - np.cos(omega))
        / np.pi
    )


# grad_R log IGSO3(R; I_3, t)
def igso3_score(
    rot: torch.Tensor, t: float, truncation_level: int = 500
) -> torch.Tensor:
    omega = so3_utils.omega(rot)
    unit_vector = torch.einsum(
        "...ij,...jk->...ik", rot, so3_utils.log_matrix(rot)
    ) / omega.unsqueeze(-1).unsqueeze(-2)
    return unit_vector * d_logf_d_omega(omega, t, truncation_level).unsqueeze(
        -1
    ).unsqueeze(-2)


def calculate_igso3(
    *,
    num_ts: int = 1000,
    num_omegas: int = 1000,
    min_t: float = 0.01,
    max_t: float = 4,
    truncation_level: int = 500,
) -> dict[str, np.ndarray]:
    """calculate_igso3 pre-computes numerical approximations to the IGSO3 cdfs
    and score norms and expected squared score norms.

    Args:
        num_ts: number of different ts for which to compute igso3
            quantities.
        num_omegas: number of point in the discretization in the angle of
            rotation.
        min_t, max_t: the upper and lower ranges for the angle of
            rotation on which to consider the IGSO3 distribution.  This cannot
            be too low or it will create numerical instability.
    """
    # Discretize omegas for calculating CDFs. Skip omega=0.
    discrete_omegas = np.linspace(0, np.pi, num_omegas + 1)[1:]

    # Exponential separation of sigmas.
    discrete_ts = 10 ** np.linspace(np.log10(min_t), np.log10(max_t), num_ts)

    # Compute the pdf and cdf values for the marginal distribution of the angle
    # of rotation (which is needed for sampling)
    pdf_vals = np.asarray(
        [igso3_density_angle(discrete_omegas, t, truncation_level) for t in discrete_ts]
    )
    pdf_vol_form_vals = np.asarray(
        [
            f_igso3(torch.tensor(discrete_omegas), t, truncation_level).numpy()
            for t in discrete_ts
        ]
    )
    cdf_vals = np.asarray([pdf.cumsum() / num_omegas * np.pi for pdf in pdf_vals])

    # Compute the norms of the scores.  This are used to scale the rotation axis when
    # computing the score as a vector.
    d_logf_d_omega_val = np.asarray(
        [
            d_logf_d_omega(discrete_omegas, t, truncation_level).numpy()
            for t in discrete_ts
        ]
    )

    return {
        "cdf": cdf_vals,  # CDF for angle of rotation -- for sampling
        "pdf_angle": pdf_vals,  # PDF for angle of rotation
        "pdf": pdf_vol_form_vals,  # PDF for w.r.t. volume form
        "d_logf_d_omega": d_logf_d_omega_val,
        "discrete_omegas": discrete_omegas,
        "discrete_ts": discrete_ts,
    }


class IGSO3:
    def __init__(
        self,
        min_t: float = 0.02,
        max_t: float = 4,
        truncation_level: int = 500,
        num_ts: int = 500,
        num_omegas: int = 1000,
        cache_dir: str = "./",
        recompute: bool = False,
    ) -> None:
        self.min_t = min_t
        self.max_t = max_t
        self.num_ts = num_ts
        self.num_omegas = num_omegas

        # Precompute and cache IGSO3 values.
        cache_dir = os.path.join(
            cache_dir, f"numT_{num_ts}_numOmega_{num_omegas}_minT_{min_t}_maxT_{max_t}"
        )

        # If cache directory doesn't exist, create it
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir)
        cdf_cache = os.path.join(cache_dir, "cdf_vals.npy")
        pdf_cache = os.path.join(cache_dir, "pdf_vals.npy")
        pdf_angle_cache = os.path.join(cache_dir, "pdf_angle_vals.npy")
        d_logf_d_omega_cache = os.path.join(cache_dir, "d_logf_d_omega_cache.npy")
        discrete_ts_cache = os.path.join(cache_dir, "discrete_ts.npy")
        discrete_omegas_cache = os.path.join(cache_dir, "discrete_omegas.npy")
        cache_fns = [
            cdf_cache,
            pdf_cache,
            pdf_angle_cache,
            d_logf_d_omega_cache,
            discrete_ts_cache,
            discrete_omegas_cache,
        ]

        # Compute and cache if cache does not exist
        if recompute or any(not os.path.exists(cache_fn) for cache_fn in cache_fns):
            logger.info("Computing and caching IGSO3.")
            igso3_vals = calculate_igso3(
                num_ts=num_ts,
                num_omegas=num_omegas,
                min_t=min_t,
                max_t=max_t,
                truncation_level=truncation_level,
            )
            np.save(cdf_cache, igso3_vals["cdf"])
            np.save(pdf_cache, igso3_vals["pdf"])
            np.save(pdf_angle_cache, igso3_vals["pdf_angle"])
            np.save(d_logf_d_omega_cache, igso3_vals["d_logf_d_omega"])
            np.save(discrete_ts_cache, igso3_vals["discrete_ts"])
            np.save(discrete_omegas_cache, igso3_vals["discrete_omegas"])

        self._cdf = np.load(cdf_cache)
        self._pdf = np.load(pdf_cache)
        self._pdf_angle = np.load(pdf_angle_cache)
        self._d_logf_d_omega = np.load(d_logf_d_omega_cache)
        self._discrete_ts = np.load(discrete_ts_cache)
        self._discrete_omegas = np.load(discrete_omegas_cache)

        self._argmin_omega_for_d_logf_d_omega = self._discrete_omegas[
            np.argmin(self._d_logf_d_omega, axis=1)
        ]

    def t_idx(self, t: NDArrayFloat) -> npt.NDArray[np.int64] | np.int64:
        """Calculates the index for discretized t during IGSO(3) initialization."""
        digitized_indices: npt.NDArray[np.int64] | np.int64 = np.digitize(
            t, self._discrete_ts
        )
        return digitized_indices - 1

    def argmin_omega_for_d_logf_d_omega(self, t: NDArrayFloat) -> NDArrayFloat:
        return self._argmin_omega_for_d_logf_d_omega[self.t_idx(t)]

    def pdf_wrt_uniform(self, rot: torch.Tensor, t: NDArrayFloat) -> np.ndarray:
        omegas = so3_utils.omega(rot)
        return np.interp(omegas, self._discrete_omegas, self._pdf[self.t_idx(t)])

    def sample_angle(self, t: NDArrayFloat, n_samples: int = 1) -> np.ndarray:
        """Uses the inverse cdf to sample an angle of rotation from IGSO(3).

        Args:
            t: time of Brownian motion
            n_samples: number of samples to draw.

        Returns: [n_samples] angles of rotation.
        """
        x = np.random.rand(n_samples)
        return np.interp(x, self._cdf[self.t_idx(t)], self._discrete_omegas)

    def sample(self, t: NDArrayFloat, n_samples: int = 1) -> torch.Tensor:
        """Generates rotation vector(s) from IGSO(3).

        Args:
            t: continuous time in [0, 1].
            n_sample: number of samples to generate.

        Returns:
            [n_samples, 3, 3] rotation matrix sampled from IGSO(3) as torch
                tensor
        """
        x = np.random.randn(n_samples, 3)
        x /= np.linalg.norm(x, axis=-1, keepdims=True)
        rot_vecs = x * self.sample_angle(t, n_samples=n_samples)[:, None]
        rot_vecs = torch.tensor(rot_vecs)
        return so3_utils.exp_matrix(
            so3_utils.skew_symmetric_matrix_from_axis_angle(rot_vecs)
        )

    def d_logf_d_omega(self, omega: np.ndarray, t: NDArrayFloat) -> np.ndarray:
        return np.interp(
            omega, self._discrete_omegas, self._d_logf_d_omega[self.t_idx(t)]
        )

    def score(
        self, rot: torch.Tensor, t: torch.Tensor, eps: float = 1e-6
    ) -> torch.Tensor:
        """Computes the score of IGSO(3) density as a rotation vector.

        Same as score function but uses pytorch and performs a look-up.

        Args:
            rot: [..., 3, 3] array of rotation matrices in SO(3)
            t: time for Brownian motion
            eps: for stability when dividing by something small

        Returns:
            [..., 3, 3] score vector in the direction of the sampled vector with
            magnitude given by _d_logf_d_omega.
        """
        omega = so3_utils.omega(rot)
        t_idx = self.t_idx(data_utils.move_to_np(t))
        omega_idx = torch.bucketize(
            omega, torch.tensor(self._discrete_omegas[:-1]).to(rot.device)
        )
        if isinstance(t_idx, np.int64):
            d_logf_d_omega_value = self._d_logf_d_omega[t_idx, omega_idx]
        else:
            d_logf_d_omega_value = self._d_logf_d_omega[t_idx[:, None], omega_idx]

        # Unit vector in tangent space
        direction = torch.einsum(
            "...jk,...kl->...jl",
            rot,
            so3_utils.log_matrix(rot) / (omega[..., None, None] + eps),
        )

        return direction * torch.tensor(
            d_logf_d_omega_value[..., None, None], dtype=direction.dtype
        )
