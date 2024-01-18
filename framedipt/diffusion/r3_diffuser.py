"""R^3 diffusion methods."""
from __future__ import annotations

import numpy as np
import torch
from omegaconf import DictConfig

from framedipt.diffusion.r3_utils import gaussian_log_prob
from framedipt.tools.custom_type import NDArrayFloat, TensorNDArray, TensorNDArrayFloat


class R3Diffuser:
    """VP-SDE diffuser class for translations."""

    def __init__(self, r3_conf: DictConfig) -> None:
        """Init function.

        Args:
            r3_conf: config for R3Diffuser.
        """
        self._r3_conf = r3_conf
        self.min_b = r3_conf.min_b
        self.max_b = r3_conf.max_b
        np.random.seed(r3_conf.seed)

    def _scale(self, x: TensorNDArrayFloat) -> TensorNDArrayFloat:
        """Scale translation.

        Args:
            x: tensor/array of translation, shape [..., 3].

        Returns:
            scaled translation, shape [..., 3].
        """
        return x * self._r3_conf.coordinate_scaling

    def _unscale(self, x: TensorNDArrayFloat) -> TensorNDArrayFloat:
        """Unscale translation.

        Args:
            x: scaled translation, shape [..., 3].

        Returns:
            unscaled translation, shape [..., 3].
        """
        return x / self._r3_conf.coordinate_scaling

    def b_t(self, t: NDArrayFloat) -> NDArrayFloat:
        """Get beta value (Gaussian distribution variance) at timestep t.

        Args:
            t: array of timesteps.

        Returns:
            array of beta values.

        Raises:
            ValueError: if any timestep is not within 0 and 1.
        """
        if np.any(t < 0) or np.any(t > 1):
            raise ValueError(f"Invalid t={t}")
        return self.min_b + t * (self.max_b - self.min_b)

    def diffusion_coef(self, t: NDArrayFloat) -> NDArrayFloat:
        """Time-dependent diffusion coefficient.

        Args:
            t: array of timesteps.

        Returns:
            diffusion coefficient i.e. sqrt(beta_t).
        """
        return np.sqrt(self.b_t(t))

    def drift_coef(self, x: np.ndarray, t: NDArrayFloat) -> np.ndarray:
        """Time-dependent drift coefficient.

        Args:
            x: array of translations.
            t: array of timesteps.

        Returns:
            drift coefficient i.e. -1/2 * beta_t * x.
        """
        return -1 / 2 * self.b_t(t) * x

    def marginal_b_t(self, t: TensorNDArrayFloat) -> TensorNDArrayFloat:
        """Marginal beta_t.

        Args:
            t: tensor/array/scaler of timestep.

        Returns:
            tensor/array/scalar of marginal beta_t.
        """
        return t * self.min_b + (1 / 2) * (t**2) * (self.max_b - self.min_b)

    def calc_trans_0(
        self,
        score_t: TensorNDArray,
        x_t: TensorNDArray,
        t: TensorNDArray,
        use_torch: bool = True,
    ) -> TensorNDArray:
        """Calculate original translation x_0 from x_t and score.

        Args:
            score_t: score at time t.
            x_t: translation at time t.
            t: continuous time in [0, 1].
            use_torch: whether to use torch.

        Returns:
            translation x_0.
        """
        beta_t: TensorNDArray = self.marginal_b_t(t)
        beta_t = beta_t[..., None, None]
        exp_fn = torch.exp if use_torch else np.exp
        cond_var = 1 - exp_fn(-beta_t)
        return (score_t * cond_var + x_t) / exp_fn(-1 / 2 * beta_t)

    def forward(  # pylint: disable=unused-argument
        self,
        x_t_1: np.ndarray,
        t_1: float,
        dt: float,
        diffuse_mask: np.ndarray | None = None,
        chain_indices: np.ndarray | None = None,  # noqa: ARG002
        center: bool = True,
        noise_scale: float = 1.0,
    ) -> np.ndarray:
        """Samples marginal p(x(t) | x(t-1)), i.e. one-step noising.

        Args:
            x_t_1: [..., n, 3] initial positions at time t-1 in Angstroms.
            t_1: continuous time at t-1 in [0, 1].
            dt: gap between adjacent time steps.
            diffuse_mask: optional mask, True=diffuse this.
            chain_indices: per-residue chain indices, unused.
            center: whether to center the data.
            noise_scale: noise scale to use in forward step.

        Returns:
            x_t: [..., n, 3] positions at time t in Angstroms.
        """
        x_t_1 = self._scale(x_t_1)
        g_t = self.diffusion_coef(t_1)
        f_t = self.drift_coef(x_t_1, t_1)
        z = noise_scale * np.random.normal(size=x_t_1.shape)
        perturb = f_t * dt + g_t * np.sqrt(dt) * z

        if diffuse_mask is not None:
            perturb *= diffuse_mask[..., None]
        else:
            diffuse_mask = np.ones(x_t_1.shape[:-1])
        x_t = x_t_1 + perturb
        if center:
            com = np.sum(x_t, axis=-2) / np.sum(diffuse_mask, axis=-1)[..., None]
            x_t -= com[..., None, :]
        x_t = self._unscale(x_t)
        return x_t

    def distribution(  # pylint: disable=unused-argument
        self,
        x_t: np.ndarray,
        score_t: np.ndarray,
        t: float,
        dt: float,
        diffuse_mask: np.ndarray | None = None,
        chain_indices: np.ndarray | None = None,  # noqa: ARG002
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the mean and std of the distribution p(x(t-1) | x(t), t).

        Args:
            x_t: translation at time t.
            score_t: score at time t.
            t: continuous time in [0, 1].
            dt: gap between adjacent time steps.
            diffuse_mask: optional mask, True in diffusion region.

        Returns:
            Mean and standard deviation of p(x(t-1) | x(t), t).
        """
        x_t = self._scale(x_t)
        g_t = self.diffusion_coef(t)
        f_t = self.drift_coef(x_t, t)
        std = g_t * np.sqrt(dt)
        mu = x_t - (f_t - g_t**2 * score_t) * dt
        if diffuse_mask is not None:
            mu *= diffuse_mask[..., None]
        return mu, std

    def log_prob_forward(  # pylint: disable=unused-argument
        self,
        x_t: np.ndarray,
        x_t_1: np.ndarray,
        t_1: float,
        dt: float,
        diffuse_mask: np.ndarray | None,
        chain_indices: np.ndarray | None = None,  # noqa: ARG002
    ) -> float:
        """Compute the log probability of q(x(t) | x(t-1)).

        Args:
            x_t: [..., n, 3] initial positions at time t in Angstroms.
            x_t_1: [..., n, 3] initial positions at time t-1 in Angstroms.
            t_1: continuous time at t-1 in [0, 1].
            diffuse_mask: optional diffusion mask.

        Returns:
            Scalar log probability of q(x(t) | x(t-1)).
        """
        x_t_1 = self._scale(x_t_1)
        g_t = self.diffusion_coef(t_1)
        f_t = self.drift_coef(x_t_1, t_1)
        std = g_t * np.sqrt(dt)
        mu = x_t_1 + f_t * dt

        if diffuse_mask is not None:
            mu *= diffuse_mask[..., None]

        x_t = self._scale(x_t)
        log_p = gaussian_log_prob(mu=mu, std=std, x=x_t, diffuse_mask=diffuse_mask)

        return log_p

    def log_prob_backward(  # pylint: disable=unused-argument
        self,
        x_t: np.ndarray,
        x_t_1: np.ndarray,
        score_t: np.ndarray,
        t: float,
        dt: float,
        diffuse_mask: np.ndarray | None,
        chain_indices: np.ndarray | None = None,  # noqa: ARG002
    ) -> float:
        """Compute the log probability of p(x(t-1) | (x_t)).

        Args:
            x_t: [..., n, 3] initial positions at time t in Angstroms.
            x_t_1: [..., n, 3] initial positions at time t-1 in Angstroms.
            score_t: score at time t with shape (N_res, 3).
            t: continuous time in [0, 1].
            dt: continuous step size in [0, 1].
            diffuse_mask: diffusion mask with 1 indicating
                the residue is diffused, shape [N].

        Returns:
            log_prob: log probability of p(x(t-1) | (x_t)).
        """
        if diffuse_mask is not None:
            diffuse_mask = diffuse_mask.astype(bool)

        mu, std = self.distribution(
            x_t=x_t, score_t=score_t, t=t, diffuse_mask=diffuse_mask, dt=dt
        )
        x_t_1 = self._scale(x_t_1)
        log_p = gaussian_log_prob(mu=mu, std=std, x=x_t_1, diffuse_mask=diffuse_mask)

        return log_p

    def forward_marginal(  # pylint: disable=unused-argument
        self,
        x_0: np.ndarray,
        t: float,
        diffuse_mask: np.ndarray | None = None,
        chain_indices: np.ndarray | None = None,  # noqa: ARG002
    ) -> tuple[np.ndarray, np.ndarray]:
        """Samples marginal p(x(t) | x(0)).

        Args:
            x_0: [..., n, 3] initial positions in Angstroms.
            t: continuous time in [0, 1].
            diffuse_mask: [..., n] equals 1 in regions to diffuse
            chain_indices: [..., n] chain indexes for each residue.

        Returns:
            x_t: [..., n, 3] positions at time t in Angstroms.
            score_t: [..., n, 3] score at time t in scaled Angstroms.
        """
        x_0_scaled = self._scale(x_0)
        x_t_scaled = np.random.normal(
            loc=np.exp(-1 / 2 * self.marginal_b_t(t)) * x_0_scaled,
            scale=np.sqrt(1 - np.exp(-self.marginal_b_t(t))),
        )
        score_t = self.score(x_t_scaled, x_0_scaled, t, scale=False)  # already scaled.
        x_t = self._unscale(x_t_scaled)
        # Apply mask
        if diffuse_mask is not None:
            x_t = diffuse_mask[..., None] * x_t + (1 - diffuse_mask[..., None]) * x_0
            score_t = diffuse_mask[..., None] * score_t
        return x_t, score_t

    def sample_stationary_distribution(  # pylint: disable=unused-argument
        self,
        x_reference: np.ndarray,
        diffuse_mask: np.ndarray | None,
        chain_indices: np.ndarray | None,  # noqa: ARG002
    ) -> np.ndarray:
        """Samples stationary distribution p(x(T)).

        The stationary distribution is the distribution the SDE converges to in the
            limit. It contains no information about x_0.

        Args:
            x_reference: [..., n,3] array of reference positions, values marked 0 in the
                diffuse mask will be kept in the output, values marked 1 will be
                replaced by p(x(T)).
            diffuse_mask: [..., n] equals 1 in regions to diffuse.
            chain_indices: [..., n] chain indexes for each residue, unused in r3.

        Returns:
            x_out: sample from stationary distribution.
        """
        x_reference_scaled = self._scale(x_reference)
        # We're going to handle the diffuse mask slightly differently here.
        # If our inpaint regions contain NaNs (because we don't know their values),
        # we don't want them to propagate, and 0 * NaN = NaN, so we can't just
        # multiply by the mask.
        if diffuse_mask is not None:
            bool_mask = diffuse_mask.astype(bool)
        else:
            bool_mask = np.ones(x_reference.shape[:-1], dtype=np.bool_)
        loc = np.zeros_like(x_reference[bool_mask])
        scale = np.ones_like(x_reference[bool_mask])
        inpaint_region = np.random.normal(loc=loc, scale=scale)
        x_out_scaled = x_reference_scaled.copy()
        x_out_scaled[bool_mask] = inpaint_region

        x_out = self._unscale(x_out_scaled)
        return x_out

    def score_scaling(self, t: float) -> float:
        """Compute score scaling.

        Args:
            t: continuous time in [0, 1].

        Returns:
            score scaling.
        """
        return 1 / np.sqrt(self.conditional_var(t))

    def reverse(  # pylint: disable=unused-argument
        self,
        x_t: np.ndarray,
        score_t: np.ndarray,
        t: float,
        dt: float,
        diffuse_mask: np.ndarray | None = None,
        chain_indices: np.ndarray | None = None,  # noqa: ARG002
        center: bool = True,
        noise_scale: float = 1.0,
    ) -> np.ndarray:
        """Simulates the reverse SDE for 1 step.

        Args:
            x_t: [..., 3] current positions at time t in angstroms.
            score_t: [..., 3] rotation score at time t.
            t: continuous time in [0, 1].
            dt: continuous step size in [0, 1].
            diffuse_mask: True indicates which residues to diffuse.
            chain_indices: which chain each residue belongs to.
            center: whether to center the data.
            noise_scale: noise scale to use in reverse step.

        Returns:
            [..., 3] positions at next step t-1.
        """
        x_t = self._scale(x_t)
        g_t = self.diffusion_coef(t)
        f_t = self.drift_coef(x_t, t)
        z = noise_scale * np.random.normal(size=score_t.shape)
        perturb = (f_t - g_t**2 * score_t) * dt + g_t * np.sqrt(dt) * z

        if diffuse_mask is not None:
            perturb *= diffuse_mask[..., None]
        else:
            diffuse_mask = np.ones(x_t.shape[:-1])
        x_t_1 = x_t - perturb
        if center:
            com = np.sum(x_t_1, axis=-2) / np.sum(diffuse_mask, axis=-1)[..., None]
            x_t_1 -= com[..., None, :]
        x_t_1 = self._unscale(x_t_1)
        return x_t_1

    def conditional_var(
        self, t: TensorNDArrayFloat, use_torch: bool = False
    ) -> torch.Tensor | np.ndarray:
        """Conditional variance of p(x_t|x_0).

        Var[x_t|x_0] = conditional_var(t)*I

        Args:
            t: continuous time in [0, 1].
            use_torch: whether to use torch.

        Returns:
            tensor or array of conditional variance.
        """
        if use_torch:
            marginal_bt = (
                self.marginal_b_t(t)
                if isinstance(t, torch.Tensor)
                else torch.tensor(self.marginal_b_t(t))
            )
            return 1 - torch.exp(torch.tensor(-marginal_bt))
        return 1 - np.exp(-self.marginal_b_t(t))

    def score(  # pylint: disable=unused-argument
        self,
        x_t: TensorNDArray,
        x_0: TensorNDArray,
        t: TensorNDArrayFloat,
        diffuse_mask: np.ndarray | None = None,  # noqa: ARG002
        chain_indices: np.ndarray | None = None,  # noqa: ARG002
        use_torch: bool = False,
        scale: bool = False,
    ) -> TensorNDArray:
        """Get score \\Delta log p(x_t | x_0).

        Args:
            x_t: translation at time t.
            x_0: translation at time 0, i.e. non-noised data.
            t: continuous time in [0, 1].
            diffuse_mask: Optional array of what residues to diffuse
            chain_indices: which chain each residue belongs to.
            use_torch: whether to use torch.
            scale: whether to scale translation.

        Returns:
            translation score in SDE.
        """
        exp_fn = torch.exp if use_torch else np.exp
        if scale:
            x_t = self._scale(x_t)
            x_0 = self._scale(x_0)
        return -(
            x_t - exp_fn(-1 / 2 * self.marginal_b_t(t)) * x_0
        ) / self.conditional_var(t, use_torch=use_torch)
