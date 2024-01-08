"""SO(3) diffusion methods."""
from __future__ import annotations

import logging
import os

import numpy as np
import numpy.typing as npt
import torch
from omegaconf import DictConfig

from framedipt.data import transforms
from framedipt.data import utils as data_utils
from framedipt.diffusion.r3_utils import gaussian_log_prob
from framedipt.tools.custom_type import NDArrayFloat, TensorNDArray, TensorNDArrayFloat


def igso3_expansion(
    omega: TensorNDArray,
    eps: TensorNDArray,
    truncation_level: int = 1000,
    use_torch: bool = False,
) -> TensorNDArrayFloat:
    """Truncated sum of IGSO(3) distribution.

    This function approximates the power series in equation 5 of
    "DENOISING DIFFUSION PROBABILISTIC MODELS ON SO(3) FOR ROTATIONAL
    ALIGNMENT"
    Leach et al. 2022

    This expression diverges from the expression in Leach in that here, eps =
    sqrt(2) * eps_leach, if eps_leach were the scale parameter of the IGSO(3).

    With this reparameterization, IGSO(3) agrees with the Brownian motion on
    SO(3) with t=eps^2.

    Args:
        omega: rotation of Euler vector (i.e. the angle of rotation)
        eps: std of IGSO(3).
        truncation_level: Truncation level
        use_torch: set true to use torch tensors, otherwise use numpy arrays.

    Raises:
        ValueError if
            omega is not tensor when use_torch=True.
            number of dimensions of omega and eps do not match.
            first dimension of omega and eps do not match.
    """
    lib = torch if use_torch else np
    levels = lib.arange(truncation_level)
    if use_torch:
        if not isinstance(omega, torch.Tensor):
            raise TypeError("omega should be Tensor in case of use_torch.")
        levels = levels.to(omega.device)
    # In case of batch operation, omega is of shape [Batch, N_res].
    if len(omega.shape) == 2:
        if len(omega.shape) != len(eps.shape):
            raise ValueError("number of dimensions of omega and eps should match.")
        if omega.shape[0] != eps.shape[0]:
            raise ValueError("first dimension of omega and eps should match.")
        # Used during predicted score calculation.
        levels = levels[None, None]  # [1, 1, L]
        omega = omega[..., None]  # [Batch, N_res, 1]
        eps = eps[..., None]
    elif len(omega.shape) == 1:
        # Used during cache computation.
        levels = levels[None]  # [1, L]
        omega = omega[..., None]  # [Batch, 1]
    else:
        raise ValueError("Omega must be 1D or 2D.")
    p = (
        (2 * levels + 1)
        * lib.exp(-levels * (levels + 1) * eps**2 / 2)
        * lib.sin(omega * (levels + 1 / 2))
        / lib.sin(omega / 2)
    )
    return p.sum(axis=-1)


def density(expansion: float, omega: float, marginal: bool = True) -> float:
    """IGSO(3) density.

    Args:
        expansion: truncated approximation of the power series in the IGSO(3)
        density.
        omega: length of an Euler vector (i.e. angle of rotation)
        marginal: set true to give marginal density over the angle of rotation,
            otherwise include normalization to give density on SO(3) or a
            rotation with angle omega.
    """
    if marginal:
        # if marginal, density over [0, pi], else over SO(3)
        return expansion * (1 - np.cos(omega)) / np.pi

    # the constant factor doesn't affect any actual calculations though
    return expansion / 8 / np.pi**2


def align_rotation_vectors(inputs: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Align the inputs with the target rotation vector.

    Args:
        inputs: rotations vector, shape (N_res, 3).
        targets: rotations vector, shape (N_res, 3).

    Returns:
        inputs_new: updated rotation vector, shape (N_res, 3).
    """
    inputs_angle = np.linalg.norm(inputs, axis=-1, keepdims=True)
    inputs_axis = inputs / inputs_angle
    target_axis = targets / np.linalg.norm(targets, axis=-1, keepdims=True)
    dot_product = np.einsum("...i,...i->...", target_axis, inputs_axis)
    sign = np.sign(dot_product)[..., np.newaxis]
    inputs_newaxis = inputs_axis * sign
    inputs_potential_angle = 2 * np.pi - inputs_angle
    inputs_newangle = np.where(sign > 0, inputs_angle, inputs_potential_angle)
    inputs_new = inputs_newaxis * inputs_newangle

    return inputs_new


def score(
    exp: TensorNDArrayFloat,
    omega: TensorNDArray,
    eps: TensorNDArray,
    truncation_level: int = 1000,
    use_torch: bool = False,
) -> TensorNDArrayFloat:  # score of density over SO(3)
    """score uses the quotient rule to compute the scaling factor for the score
    of the IGSO(3) density.

    This function is used within the Diffuser class to when computing the score
    as an element of the tangent space of SO(3).

    This uses the quotient rule of calculus, and take the derivative of the
    log:
        d hi(x)/lo(x) = (lo(x) d hi(x)/dx - hi(x) d lo(x)/dx) / lo(x)^2
    and
        d log expansion(x) / dx = (d expansion(x)/ dx) / expansion(x)

    Args:
        exp: truncated expansion of the power series in the IGSO(3) density
        omega: length of an Euler vector (i.e. angle of rotation)
        eps: scale parameter for IGSO(3) -- as in expansion() this scaling
            differ from that in Leach by a factor of sqrt(2).
        truncation_level: truncation level
        use_torch: set true to use torch tensors, otherwise use numpy arrays.

    Returns:
        The d/d omega log IGSO3(omega; eps)/(1-cos(omega))

     Raises:
        ValueError if
            omega is not tensor when use_torch=True.
            omega does not have the same shape as exp.
            number of dimensions of omega and eps do not match.
            first dimension of omega and eps do not match.
    """
    lib = torch if use_torch else np
    levels = lib.arange(truncation_level)
    if use_torch:
        if not isinstance(omega, torch.Tensor):
            raise TypeError("omega should be Tensor in case of use_torch.")
        levels = levels.to(omega.device)
    levels = levels[None]
    # In case of batch operation, omega is of shape [Batch, N_res].
    if len(omega.shape) == 2:
        if isinstance(exp, (np.ndarray, torch.Tensor)) and omega.shape != exp.shape:
            raise ValueError("omega should have the same shape as exp.")
        if len(omega.shape) != len(eps.shape):
            raise ValueError("number of dimensions of omega and eps should match.")
        if omega.shape[0] != eps.shape[0]:
            raise ValueError("first dimension of omega and eps should match.")
        levels = levels[None]
    elif len(omega.shape) > 2:
        raise ValueError("Omega must be 1D or 2D.")

    omega = omega[..., None]
    eps = eps[..., None]
    hi = lib.sin(omega * (levels + 1 / 2))
    dhi = (levels + 1 / 2) * lib.cos(omega * (levels + 1 / 2))
    lo = lib.sin(omega / 2)
    dlo = 1 / 2 * lib.cos(omega / 2)
    dsigma = (
        (2 * levels + 1)
        * lib.exp(-levels * (levels + 1) * eps**2 / 2)
        * (lo * dhi - hi * dlo)
        / lo**2
    )
    dsigma = dsigma.sum(axis=-1)
    return dsigma / (exp + 1e-4)


class SO3Diffuser:
    def __init__(self, so3_conf: DictConfig) -> None:
        self.schedule = so3_conf.schedule

        self.min_sigma = so3_conf.min_sigma
        self.max_sigma = so3_conf.max_sigma

        self.num_sigma = so3_conf.num_sigma
        self.use_cached_score = so3_conf.use_cached_score
        self._log = logging.getLogger(__name__)

        # Discretize omegas for calculating CDFs. Skip omega=0.
        self.discrete_omega = np.linspace(0, np.pi, so3_conf.num_omega + 1)[1:]

        def replace_period(x: float) -> str:
            """Replace `.` in a float by `_` and return the string.

            For example, 1.1 will be "1_1".

            Args:
                x: input float.

            Returns:
                 String of the input float, replaced `.` by `_`.
            """
            return str(x).replace(".", "_")

        # Precompute IGSO3 values.
        cache_dir = os.path.join(
            so3_conf.cache_dir,
            f"eps_{so3_conf.num_sigma}_"
            f"omega_{so3_conf.num_omega}_"
            f"min_sigma_{replace_period(so3_conf.min_sigma)}_"
            f"max_sigma_{replace_period(so3_conf.max_sigma)}_"
            f"schedule_{so3_conf.schedule}",
        )

        # If cache directory doesn't exist, create it
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir)
        pdf_cache = os.path.join(cache_dir, "pdf_vals.npy")
        cdf_cache = os.path.join(cache_dir, "cdf_vals.npy")
        score_norms_cache = os.path.join(cache_dir, "score_norms.npy")

        if (
            os.path.exists(pdf_cache)
            and os.path.exists(cdf_cache)
            and os.path.exists(score_norms_cache)
        ):
            self._log.info(f"Using cached IGSO3 in {cache_dir}")
            self._pdf = np.load(pdf_cache)
            self._cdf = np.load(cdf_cache)
            self._score_norms = np.load(score_norms_cache)
        else:
            self._log.info(f"Computing IGSO3. Saving in {cache_dir}")
            # compute the expansion of the power series
            exp_vals = np.asarray(
                [
                    igso3_expansion(self.discrete_omega, sigma)
                    for sigma in self.discrete_sigma
                ]
            )
            # Compute the pdf and cdf values for the marginal distribution of the angle
            # of rotation (which is needed for sampling)
            self._pdf = np.asarray(
                [density(x, self.discrete_omega, marginal=True) for x in exp_vals]
            )
            self._cdf = np.asarray(
                [pdf.cumsum() / so3_conf.num_omega * np.pi for pdf in self._pdf]
            )

            # Compute the norms of the scores.
            # These are used to scale the rotation axis when
            # computing the score as a vector.
            self._score_norms = np.asarray(
                [
                    score(exp_vals[i], self.discrete_omega, x)
                    for i, x in enumerate(self.discrete_sigma)
                ]
            )

            # Cache the precomputed values
            np.save(pdf_cache, self._pdf)
            np.save(cdf_cache, self._cdf)
            np.save(score_norms_cache, self._score_norms)

        self._score_scaling = np.sqrt(
            np.abs(
                np.sum(self._score_norms**2 * self._pdf, axis=-1)
                / np.sum(self._pdf, axis=-1)
            )
        ) / np.sqrt(3)
        np.random.seed(so3_conf.seed)

    @property
    def discrete_sigma(self) -> np.ndarray:
        return self.sigma(np.linspace(0.0, 1.0, self.num_sigma))

    def sigma_idx(self, sigma: NDArrayFloat) -> npt.NDArray[np.int64] | np.int64:
        """Calculates the index for discretized sigma during IGSO(3) initialization."""
        digitized_indices: npt.NDArray[np.int64] | np.int64 = np.digitize(
            sigma, self.discrete_sigma
        )
        return digitized_indices - 1

    def sigma(self, t: NDArrayFloat) -> NDArrayFloat:
        """Extract \\sigma(t) corresponding to chosen sigma schedule."""
        if np.any(t < 0) or np.any(t > 1):
            raise ValueError(f"Invalid t={t}")
        if self.schedule == "logarithmic":
            return np.log(t * np.exp(self.max_sigma) + (1 - t) * np.exp(self.min_sigma))

        raise ValueError(f"Unrecognize schedule {self.schedule}")

    def diffusion_coef(self, t: NDArrayFloat) -> NDArrayFloat:
        """Compute diffusion coefficient (g_t)."""
        if self.schedule == "logarithmic":
            g_t = np.sqrt(
                2
                * (np.exp(self.max_sigma) - np.exp(self.min_sigma))
                * self.sigma(t)
                / np.exp(self.sigma(t))
            )
        else:
            raise ValueError(f"Unrecognize schedule {self.schedule}")
        return g_t

    def t_to_idx(self, t: NDArrayFloat) -> npt.NDArray[np.int64] | np.int64:
        """Helper function to go from time t to corresponding sigma_idx."""
        return self.sigma_idx(self.sigma(t))

    def sample_igso3(self, t: float, n_samples: int = 1) -> np.ndarray:
        """Uses the inverse cdf to sample an angle of rotation from IGSO(3).

        Args:
            t: continuous time in [0, 1].
            n_samples: number of samples to draw.

        Returns:
            [n_samples] angles of rotation.
        """
        x = np.random.rand(n_samples)
        return np.interp(
            x,
            self._cdf[self.t_to_idx(t)],
            self.discrete_omega,
        )

    def sample(self, t: float, n_samples: int = 1) -> np.ndarray:
        """Generates rotation vector(s) from IGSO(3).

        Args:
            t: continuous time in [0, 1].
            n_samples: number of samples to generate.

        Returns:
            [n_samples, 3] axis-angle rotation vectors sampled from IGSO(3).
        """
        x = np.random.randn(n_samples, 3)
        x /= np.linalg.norm(x, axis=-1, keepdims=True)
        return x * self.sample_igso3(t, n_samples=n_samples)[:, None]

    def sample_ref(self, n_samples: int = 1) -> np.ndarray:
        return self.sample(1.0, n_samples=n_samples)

    def score(self, vec: np.ndarray, t: float, eps: float = 1e-6) -> np.ndarray:
        """Computes the score of IGSO(3) density as a rotation vector.

        Args:
            vec: [..., 3] array of axis-angle rotation vectors.
            t: continuous time in [0, 1].

        Returns:
            [..., 3] score vector in the direction of the sampled vector with
            magnitude given by _score_norms.
        """
        torch_score = self.torch_score(torch.tensor(vec), torch.tensor(t)[None], eps)
        return torch_score.numpy()

    def torch_score(
        self, vec: torch.Tensor, t: torch.Tensor, eps: float = 1e-6
    ) -> torch.Tensor:
        """Computes the score of IGSO(3) density as a rotation vector.

        Same as score function but uses pytorch and performs a look-up.

        Args:
            vec: [..., 3] array of axis-angle rotation vectors.
            t: continuous time in [0, 1].

        Returns:
            [..., 3] score vector in the direction of the sampled vector with
            magnitude given by _score_norms.
        """
        omega = torch.linalg.norm(vec, dim=-1) + eps
        if self.use_cached_score:
            score_norms_t = self._score_norms[self.t_to_idx(data_utils.move_to_np(t))]
            score_norms_t = torch.tensor(score_norms_t).to(vec.device)
            omega_idx = torch.bucketize(
                omega, torch.tensor(self.discrete_omega[:-1]).to(vec.device)
            )
            omega_idx = omega_idx[None, :] if omega_idx.dim() == 1 else omega_idx
            omega_scores_t = torch.gather(score_norms_t, 1, omega_idx)
        else:
            sigma = self.discrete_sigma[self.t_to_idx(data_utils.move_to_np(t))]
            sigma = torch.tensor(sigma).to(vec.device)
            omega_vals = igso3_expansion(omega, sigma[:, None], use_torch=True)
            omega_scores_t = score(omega_vals, omega, sigma[:, None], use_torch=True)
        return omega_scores_t[..., None] * vec / omega[..., None]

    def score_scaling(self, t: NDArrayFloat) -> NDArrayFloat:
        """Calculates scaling used for scores during training."""
        return self._score_scaling[self.t_to_idx(t)]

    def forward(  # pylint: disable=unused-argument
        self,
        x_t_1: np.ndarray,
        t_1: float,
        dt: float,
        diffuse_mask: np.ndarray | None = None,
        chain_indices: np.ndarray | None = None,  # noqa: ARG002
        noise_scale: float = 1.0,
    ) -> np.ndarray:
        """Samples marginal p(x(t) | x(t-1)), i.e. one-step noising.

        Args:
            x_t_1: [..., n, 3] initial rotations at time t-1 in Angstroms.
            t_1: continuous time in [0, 1].
            dt: time gap between two steps.
            diffuse_mask: Array of diffusion mask with 1 indicating
                the residue is diffused, shape (N).
            noise_scale: noise scale to use in forward step.

        Returns:
            x_t: [..., n, 3] rotations at time t in Angstroms.
        """
        g_t = self.diffusion_coef(t_1)
        z = noise_scale * np.random.normal(size=x_t_1.shape)
        perturb = g_t * np.sqrt(dt) * z

        if diffuse_mask is not None:
            perturb *= diffuse_mask[..., None]
        n_samples = np.cumprod(x_t_1.shape[:-1])[-1]

        # Right multiply.
        x_t = transforms.compose_rotvec(
            x_t_1.reshape(n_samples, 3), perturb.reshape(n_samples, 3)
        ).reshape(x_t_1.shape)

        return x_t

    def forward_marginal(
        self, rot_0: np.ndarray, t: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Samples from the forward diffusion process at time index t.

        Args:
            rot_0: [..., 3] initial rotations.
            t: continuous time in [0, 1].

        Returns:
            rot_t: [..., 3] noised rotation vectors.
            rot_score: [..., 3] score of rot_t as a rotation vector.
        """
        n_samples = np.cumprod(rot_0.shape[:-1])[-1]
        sampled_rots = self.sample(t, n_samples=n_samples)
        rot_score = self.score(sampled_rots, t).reshape(rot_0.shape)

        # Right multiply.
        rot_t = transforms.compose_rotvec(rot_0, sampled_rots).reshape(rot_0.shape)
        return rot_t, rot_score

    def log_prob_forward(  # pylint: disable=unused-argument
        self,
        rot_t: np.ndarray,
        rot_t_1: np.ndarray,
        t_1: float,
        dt: float,
        diffuse_mask: np.ndarray | None = None,
        chain_indices: np.ndarray | None = None,  # noqa: ARG002
    ) -> float:
        """Compute the log probability of q(x(t) | x_(t-1)).

        Args:
            rot_t: rotations at time t, shape (N_res, 3).
            rot_t_1: rotations at time t-1, shape (N_res, 3).
            t_1: continuous time at t-1 in [0, 1].
            dt: continuous step size in [0, 1].
            diffuse_mask: Array of diffusion mask with 1 indicating
                the residue is diffused, shape (N).

        Returns:
            log_p: log probability of q(x(t) | x_(t-1)).
        """
        g_t = self.diffusion_coef(t_1)
        std = g_t * np.sqrt(dt)
        rot_t_new = align_rotation_vectors(rot_t, rot_t_1)
        log_p = gaussian_log_prob(
            mu=rot_t_1, std=std, x=rot_t_new, diffuse_mask=diffuse_mask
        )

        return log_p

    def log_prob_backward(  # pylint: disable=unused-argument
        self,
        rot_t: np.ndarray,
        rot_t_1: np.ndarray,
        score_t: np.ndarray,
        t: float,
        dt: float,
        diffuse_mask: np.ndarray | None = None,
        chain_indices: np.ndarray | None = None,  # noqa: ARG002
    ) -> float:
        """Compute the log probability of p(x(t-1) | (x(t))).

        Args:
            rot_t: rotations at time t, shape (N_res, 3).
            rot_t_1: rotations at time t-1, shape (N_res, 3).
            score_t: rotation score at time t, shape (N_res, 3).
            t: continuous time in [0, 1].
            dt: continuous step size in [0, 1].
            diffuse_mask: Array of diffusion mask with 1 indicating
                the residue is diffused, shape (N).

        Returns:
            log_p: log probability of q(x(t) | x_(t-1)).
        """
        mu, std = self.distribution(
            rot_t=rot_t, t=t, score_t=score_t, dt=dt, diffuse_mask=diffuse_mask
        )
        rot_t_1_new = align_rotation_vectors(rot_t_1, mu)
        log_p = gaussian_log_prob(
            mu=mu, std=std, x=rot_t_1_new, diffuse_mask=diffuse_mask
        )

        return log_p

    def distribution(  # pylint: disable=unused-argument
        self,
        rot_t: np.ndarray,
        score_t: np.ndarray,
        t: float,
        dt: float,
        diffuse_mask: np.ndarray | None = None,
        chain_indices: np.ndarray | None = None,  # noqa: ARG002
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get Gaussian distribution from x_t.

        Args:
            rot_t: rotations at time t, shape (N_res, 3).
            score_t: rotation score at time t, shape (N_res, 3).
            t: continuous time in [0, 1].
            dt: continuous step size in [0, 1].
            diffuse_mask: optional mask.

        Returns:
             Mean and variance of the Gaussian distribution.
        """
        g_t = self.diffusion_coef(t)
        drift = (g_t**2) * score_t * dt
        std = g_t * np.sqrt(dt)

        if diffuse_mask is not None:
            drift *= diffuse_mask[..., None]

        # Reshape so arrays are (N, 3)
        n_samples = np.cumprod(rot_t.shape[:-1])[-1]

        # Right multiply.
        mu = transforms.compose_rotvec(
            rot_t.reshape(n_samples, 3), drift.reshape(n_samples, 3)
        ).reshape(rot_t.shape)

        return mu, std

    def reverse(
        self,
        rot_t: np.ndarray,
        score_t: np.ndarray,
        t: float,
        dt: float,
        diffuse_mask: np.ndarray | None = None,
        noise_scale: float = 1.0,
    ) -> np.ndarray:
        """Simulates the reverse SDE for 1 step using the Geodesic random walk.

        Args:
            rot_t: [..., 3] current rotations at time t.
            score_t: [..., 3] rotation score at time t.
            t: continuous time in [0, 1].
            dt: continuous step size in [0, 1].
            diffuse_mask: True indicates which residues to diffuse.

        Returns:
            [..., 3] rotation vector at next step.
        """
        g_t = self.diffusion_coef(t)
        z = noise_scale * np.random.normal(size=score_t.shape)
        perturb = (g_t**2) * score_t * dt + g_t * np.sqrt(dt) * z

        if diffuse_mask is not None:
            perturb *= diffuse_mask[..., None]
        n_samples = np.cumprod(rot_t.shape[:-1])[-1]

        # Right multiply.
        rot_t_1 = transforms.compose_rotvec(
            rot_t.reshape(n_samples, 3), perturb.reshape(n_samples, 3)
        ).reshape(rot_t.shape)
        return rot_t_1
