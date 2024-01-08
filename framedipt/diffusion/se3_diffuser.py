"""SE(3) diffusion methods."""
from __future__ import annotations

import logging

import numpy as np
import torch
from omegaconf import DictConfig
from scipy.spatial.transform import Rotation

from framedipt.data import transforms
from framedipt.diffusion import r3_diffuser, so3_diffuser
from openfold.utils import rigid_utils


def _extract_trans_rots(rigid: rigid_utils.Rigid) -> tuple[np.ndarray, np.ndarray]:
    rot = rigid.get_rots().get_rot_mats().cpu().numpy()
    rot_shape = rot.shape
    num_rots = np.cumprod(rot_shape[:-2])[-1]
    rot = rot.reshape((num_rots, 3, 3))
    rot = Rotation.from_matrix(rot).as_rotvec().reshape(rot_shape[:-2] + (3,))
    tran = rigid.get_trans().cpu().numpy()
    return tran, rot


def _assemble_rigid(rotvec: np.ndarray, trans: np.ndarray) -> rigid_utils.Rigid:
    rotvec_shape = rotvec.shape
    num_rotvecs = np.cumprod(rotvec_shape[:-1])[-1]
    rotvec = rotvec.reshape((num_rotvecs, 3))
    rotmat = (
        Rotation.from_rotvec(rotvec).as_matrix().reshape(rotvec_shape[:-1] + (3, 3))
    )
    return rigid_utils.Rigid(
        rots=rigid_utils.Rotation(rot_mats=torch.Tensor(rotmat)),
        trans=torch.tensor(trans),
    )


class SE3Diffuser:
    def __init__(self, se3_conf: DictConfig) -> None:
        self._log = logging.getLogger(__name__)
        self._se3_conf = se3_conf

        self._diffuse_rot = se3_conf.diffuse_rot
        self._so3_diffuser = so3_diffuser.SO3Diffuser(self._se3_conf.so3)

        self._diffuse_trans = se3_conf.diffuse_trans
        self._r3_diffuser = r3_diffuser.R3Diffuser(self._se3_conf.r3)

    def forward(
        self,
        rigids_t_1: rigid_utils.Rigid,
        t_1: float,
        dt: float,
        diffuse_mask: np.ndarray | None = None,
        chain_indices: np.ndarray | None = None,
    ) -> rigid_utils.Rigid:
        """Samples marginal p(x(t) | x(t-1)), i.e. one-step forward noising.

        Args:
            rigids_t_1: rigid-frame representation at time t-1 in Angstrom.
            t_1: continuous time t-1 in [0, 1].
            dt: time gap between two steps.
            diffuse_mask: Array of diffusion mask with 1 indicating
                the residue is diffused, shape (N).

        Returns:
            Rigid-frame representation at time t.
        """
        # Unpack trans_t_1 and rot_t_1, each with shape (N_res, 3).
        trans_t_1, rot_t_1 = _extract_trans_rots(rigids_t_1)

        trans_t = self._r3_diffuser.forward(
            x_t_1=trans_t_1,
            t_1=t_1,
            dt=dt,
            diffuse_mask=diffuse_mask,
            chain_indices=chain_indices,
            center=False,
        )
        rot_t = self._so3_diffuser.forward(
            x_t_1=rot_t_1,
            t_1=t_1,
            dt=dt,
            diffuse_mask=diffuse_mask,
            chain_indices=chain_indices,
        )

        if diffuse_mask is not None:
            rot_t = self._apply_mask(rot_t, rot_t_1, diffuse_mask[..., None])
            trans_t = self._apply_mask(trans_t, trans_t_1, diffuse_mask[..., None])

        rigids = _assemble_rigid(rot_t, trans_t)

        return rigids

    def log_prob_forward(
        self,
        rigids_t: rigid_utils.Rigid,
        rigids_t_1: rigid_utils.Rigid,
        t_1: float,
        dt: float,
        diffuse_mask: np.ndarray | None = None,
        chain_indices: np.ndarray | None = None,
    ) -> float:
        """Log probability of the forward distribution q(x(t) | q(x(t-1)).

        Args:
            rigids_t: rigid-frame representation at time t in Angstrom.
            rigids_t_1: rigid-frame representation at time t-1 in Angstrom.
            t_1: continuous time at t-1 in [0, 1].
            dt: continuous step size in [0, 1].Q
            diffuse_mask: Array of diffusion mask with 1 indicating
                the residue is diffused, shape (N).

        Returns:
            log_p: log probability of q(x(t) | q(x(t-1)) evaluated at x_t.
        """
        trans_t, rot_t = _extract_trans_rots(rigids_t)
        trans_t_1, rot_t_1 = _extract_trans_rots(rigids_t_1)

        trans_log_p = self._r3_diffuser.log_prob_forward(
            x_t=trans_t,
            x_t_1=trans_t_1,
            t_1=t_1,
            dt=dt,
            diffuse_mask=diffuse_mask,
            chain_indices=chain_indices,
        )

        rot_log_p = self._so3_diffuser.log_prob_forward(
            rot_t=rot_t,
            rot_t_1=rot_t_1,
            t_1=t_1,
            dt=dt,
            diffuse_mask=diffuse_mask,
            chain_indices=chain_indices,
        )

        log_p = trans_log_p + rot_log_p

        return log_p

    def log_prob_backward(
        self,
        rigids_t: rigid_utils.Rigid,
        rigids_t_1: rigid_utils.Rigid,
        trans_score_t: np.ndarray,
        rot_score_t: np.ndarray,
        t: float,
        dt: float,
        diffuse_mask: np.ndarray | None,
        chain_indices: np.ndarray | None = None,
    ) -> float:
        """Log probability of the backward distribution p(x(t-1) | x(t)).

        Args:
            rigids_t: rigid-frame representation at time t.
            rigids_t_1: rigid-frame representation at time t-1.
            trans_score_t: translation score at time t, shape (N_res, 3).
            rot_score_t: rotation score at time t, shape (N_res, 3).
            t: continuous time in [0, 1].
            dt: continuous step size in [0, 1].
            diffuse_mask: Array of diffusion mask with 1 indicating
                the residue is diffused, shape (N).

        Returns:
            log_prob: log probability of the distribution p(x(t-1) | x(t)).
        """
        trans_t, rot_t = _extract_trans_rots(rigids_t)
        trans_t_1, rot_t_1 = _extract_trans_rots(rigids_t_1)

        trans_log_p = self._r3_diffuser.log_prob_backward(
            x_t=trans_t,
            x_t_1=trans_t_1,
            score_t=trans_score_t,
            dt=dt,
            t=t,
            diffuse_mask=diffuse_mask,
            chain_indices=chain_indices,
        )

        rot_log_p = self._so3_diffuser.log_prob_backward(
            rot_t=rot_t,
            rot_t_1=rot_t_1,
            score_t=rot_score_t,
            t=t,
            dt=dt,
            diffuse_mask=diffuse_mask,
            chain_indices=chain_indices,
        )

        # Overall backwards log prob
        log_p = trans_log_p + rot_log_p

        return log_p

    def forward_marginal(
        self,
        rigids_0: rigid_utils.Rigid,
        t: float,
        diffuse_mask: np.ndarray | None = None,
        chain_indices: np.ndarray | None = None,
        as_tensor_7: bool = True,
    ) -> dict[str, rigid_utils.Rigid | torch.Tensor | np.ndarray | float]:
        """
        Args:
            rigids_0: [..., N] openfold Rigid objects
            t: continuous time in [0, 1].
            chain_index: Array of chain indexes indicating each residue
                belongs to which chain.
            diffuse_mask: Array of diffusion mask with 1 indicating
                the residue is diffused.
            as_tensor_7: Whether to represent rigid frames as tensor of shape 7.

        Returns:
            rigids_t: [..., N] noised rigid. [..., N, 7] if as_tensor_7 is true.
            trans_score: [..., N, 3] translation score
            rot_score: [..., N, 3] rotation score
            trans_score_norm: [...] translation score norm
            rot_score_norm: [...] rotation score norm
        """
        trans_0, rot_0 = _extract_trans_rots(rigids_0)

        if not self._diffuse_rot:
            rot_t, rot_score, rot_score_scaling = (
                rot_0,
                np.zeros_like(rot_0),
                np.ones_like(t),
            )
        else:
            rot_t, rot_score = self._so3_diffuser.forward_marginal(rot_0, t)
            rot_score_scaling = self._so3_diffuser.score_scaling(t)

        if not self._diffuse_trans:
            trans_t, trans_score, trans_score_scaling = (
                trans_0,
                np.zeros_like(trans_0),
                np.ones_like(t),
            )
        else:
            trans_t, trans_score = self._r3_diffuser.forward_marginal(
                trans_0, t, diffuse_mask=diffuse_mask, chain_indices=chain_indices
            )

            trans_score_scaling = self._r3_diffuser.score_scaling(t)

        if diffuse_mask is not None:
            rot_t = self._apply_mask(rot_t, rot_0, diffuse_mask[..., None])
            rot_score = self._apply_mask(
                rot_score, np.zeros_like(rot_score), diffuse_mask[..., None]
            )
        rigids_t = _assemble_rigid(rot_t, trans_t)
        if as_tensor_7:
            rigids_t = rigids_t.to_tensor_7()
        return {
            "rigids_t": rigids_t,
            "trans_score": trans_score,
            "rot_score": rot_score,
            "trans_score_scaling": trans_score_scaling,
            "rot_score_scaling": rot_score_scaling,
        }

    def calc_trans_0(
        self, trans_score: np.ndarray, trans_t: np.ndarray, t: np.ndarray
    ) -> np.ndarray:
        return self._r3_diffuser.calc_trans_0(trans_score, trans_t, t)

    def calc_trans_score(
        self,
        trans_t: np.ndarray,
        trans_0: np.ndarray,
        t: float,
        use_torch: bool = False,
        scale: bool = True,
    ) -> np.ndarray:
        return self._r3_diffuser.score(
            trans_t, trans_0, t, use_torch=use_torch, scale=scale
        )

    def calc_rot_score(
        self,
        rots_t: rigid_utils.Rotation,
        rots_0: rigid_utils.Rotation,
        t: torch.Tensor,
    ) -> torch.Tensor:
        rots_0_inv = rots_0.invert()
        quats_0_inv = rots_0_inv.get_quats()
        quats_t = rots_t.get_quats()
        quats_0t = rigid_utils.quat_multiply(quats_0_inv, quats_t)
        rotvec_0t = transforms.quat_to_rotvec(quats_0t)
        return self._so3_diffuser.torch_score(rotvec_0t, t)

    def _apply_mask(
        self, x_diff: np.ndarray, x_fixed: np.ndarray, diff_mask: np.ndarray
    ) -> np.ndarray:
        return diff_mask * x_diff + (1 - diff_mask) * x_fixed

    def trans_parameters(
        self,
        trans_t: np.ndarray,
        score_t: np.ndarray,
        t: float,
        dt: float,
        diffuse_mask: np.ndarray,
        chain_indices: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._r3_diffuser.distribution(
            trans_t, score_t, t, dt, diffuse_mask, chain_indices=chain_indices
        )

    def score(
        self,
        rigid_0: rigid_utils.Rigid,
        rigid_t: rigid_utils.Rigid,
        t: float,
        diffuse_mask: np.ndarray | None = None,
        chain_indices: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        tran_0, rot_0 = _extract_trans_rots(rigid_0)
        tran_t, rot_t = _extract_trans_rots(rigid_t)

        if not self._diffuse_rot:
            rot_score = np.zeros_like(rot_0)
        else:
            rot_score = self._so3_diffuser.score(rot_t, t)

        if not self._diffuse_trans:
            trans_score = np.zeros_like(tran_0)
        else:
            trans_score = self._r3_diffuser.score(
                tran_t,
                tran_0,
                t,
                diffuse_mask=diffuse_mask,
                chain_indices=chain_indices,
            )

        return trans_score, rot_score

    def score_scaling(self, t: float) -> tuple[float, float]:
        rot_score_scaling = self._so3_diffuser.score_scaling(t)
        trans_score_scaling = self._r3_diffuser.score_scaling(t)
        return rot_score_scaling, trans_score_scaling

    def reverse(
        self,
        rigid_t: rigid_utils.Rigid,
        rot_score: np.ndarray,
        trans_score: np.ndarray,
        t: float,
        dt: float,
        diffuse_mask: np.ndarray | None = None,
        chain_indices: torch.Tensor | None = None,
        center: bool = True,
        noise_scale: float = 1.0,
    ) -> rigid_utils.Rigid:
        """Reverse sampling function from (t) to (t-1).

        Args:
            rigid_t: [..., N] protein rigid objects at time t.
            rot_score: [..., N, 3] rotation score.
            trans_score: [..., N, 3] translation score.
            t: continuous time in [0, 1].
            dt: continuous step size in [0, 1].
            mask: [..., N] which residues to update.
            center: true to set center of mass to zero after step

        Returns:
            rigid_t_1: [..., N] protein rigid objects at time t-1.
        """
        trans_t, rot_t = _extract_trans_rots(rigid_t)
        if not self._diffuse_rot:
            rot_t_1 = rot_t
        else:
            rot_t_1 = self._so3_diffuser.reverse(
                rot_t=rot_t,
                score_t=rot_score,
                t=t,
                dt=dt,
                noise_scale=noise_scale,
            )
        if not self._diffuse_trans:
            trans_t_1 = trans_t
        else:
            trans_t_1 = self._r3_diffuser.reverse(
                x_t=trans_t,
                score_t=trans_score,
                t=t,
                dt=dt,
                diffuse_mask=diffuse_mask,
                chain_indices=chain_indices,
                noise_scale=noise_scale,
                center=center,
            )

        if diffuse_mask is not None:
            trans_t_1 = self._apply_mask(trans_t_1, trans_t, diffuse_mask[..., None])
            rot_t_1 = self._apply_mask(rot_t_1, rot_t, diffuse_mask[..., None])

        return _assemble_rigid(rot_t_1, trans_t_1)

    def sample_ref_for_rotation(
        self,
        n_samples: int,
        rot_impute: np.ndarray,
    ) -> np.ndarray:
        """Sample reference for rotation.

        Args:
            n_samples: number of samples.
            rot_impute: original rotation, applied in case of
                not diffusing rotation.

        Return:
            Sample of rotation from reference distribution
                or original rotation.
        """
        if self._diffuse_rot:
            rot_ref = self._so3_diffuser.sample_ref(n_samples=n_samples)
        else:
            rot_ref = rot_impute

        return rot_ref

    def sample_ref_for_translation(
        self,
        trans_impute: np.ndarray,
        chain_index: np.ndarray | None = None,
        diffuse_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Sample reference for translation.

        Args:
            trans_impute: original translation, applied in case of
                not diffusing translation.
            chain_index: Array of chain indexes indicating each residue
                belongs to which chain.
            diffuse_mask: Array of diffusion mask with 1 indicating
                the residue is diffused.

        Return:
            Sample of translation from reference distribution
                or original translation.
        """
        if self._diffuse_trans:
            trans_ref = self._r3_diffuser.sample_stationary_distribution(
                trans_impute, diffuse_mask=diffuse_mask, chain_indices=chain_index
            )
        else:
            trans_ref = trans_impute

        return trans_ref

    def sample_ref(
        self,
        n_samples: int,
        chain_index: np.ndarray | None = None,
        impute: rigid_utils.Rigid | None = None,
        diffuse_mask: np.ndarray | None = None,
        as_tensor_7: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Samples rigids from reference distribution.

        Args:
            n_samples: Number of samples.
            impute: Rigid objects to use as imputation values if either
                translations or rotations are not diffused.
            chain_index: Array of chain indexes indicating each residue
                belongs to which chain.
            diffuse_mask: Array of diffusion mask with 1 indicating
                the residue is diffused.
            as_tensor_7: Whether to represent rigid frames as tensor of shape 7.

        Returns:
            Dictionary of {"rigid_t": sampled reference rigid}.

        Raises:
            ValueError if impute does not have the shape of (n_samples, ...).
            ValueError if doing masked diffusion (inpainting) and impute is not given.
            ValueError if not diffusing rotations or not diffusing translations and
                impute is not given.
        """
        if impute is None:
            # If either rotation or translation aren't being diffused,
            # then we need impute
            if not self._diffuse_rot:
                raise ValueError(
                    "Must provide impute values as we're not diffusing rotations!"
                )
            if not self._diffuse_trans:
                raise ValueError(
                    "Must provide impute values as we're not diffusing translations!"
                )
            # If there's a mask, then we need impute values to work with
            if diffuse_mask is not None:
                raise ValueError("Must provide imputation values for unmasked regions!")
            # We have no impute and no mask, therefore we're doing full diffusion,
            # So we can make some dummy values for translation and rotation
            # knowing that they won't be used.
            dummy_impute = rigid_utils.Rigid.identity((n_samples,))
            trans_impute, rot_impute = _extract_trans_rots(dummy_impute)
        else:
            if impute.shape[0] != n_samples:
                raise ValueError(
                    f"impute should have shape ({n_samples}, ...), got {impute.shape}."
                )
            trans_impute, rot_impute = _extract_trans_rots(impute)
            trans_impute = trans_impute.reshape((n_samples, 3))
            rot_impute = rot_impute.reshape((n_samples, 3))

        rot_ref = self.sample_ref_for_rotation(
            n_samples=n_samples,
            rot_impute=rot_impute,
        )

        trans_ref = self.sample_ref_for_translation(
            trans_impute=trans_impute,
            chain_index=chain_index,
            diffuse_mask=diffuse_mask,
        )

        if diffuse_mask is not None:
            rot_ref = self._apply_mask(rot_ref, rot_impute, diffuse_mask[..., None])

        rigids_t = _assemble_rigid(rot_ref, trans_ref)
        if as_tensor_7:
            rigids_t = rigids_t.to_tensor_7()
        return {"rigids_t": rigids_t}
