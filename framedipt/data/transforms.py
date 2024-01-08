"""Module of rigid transform utils."""
# pylint: disable=E1120,C0103,W0612
from __future__ import annotations

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from framedipt.diffusion import so3_utils
from framedipt.protein import residue_constants
from framedipt.tools.log import get_logger
from openfold.utils import rigid_utils

logger = get_logger()


def rigid_frames_from_atom_14(atom_14: torch.Tensor) -> rigid_utils.Rigid:
    n_atoms = atom_14[:, 0]
    ca_atoms = atom_14[:, 1]
    c_atoms = atom_14[:, 2]
    return rigid_utils.Rigid.from_3_points(n_atoms, ca_atoms, c_atoms)


def rigid_frames_from_all_atom(all_atom_pos: np.ndarray) -> rigid_utils.Rigid:
    rigid_atom_pos = []
    for atom in ["N", "CA", "C"]:
        atom_idx = residue_constants.atom_order[atom]
        atom_pos = all_atom_pos[..., atom_idx, :]
        rigid_atom_pos.append(atom_pos)
    return rigid_utils.Rigid.from_3_points(*rigid_atom_pos)


def compose_rotvec(r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
    """Compose two rotation euler vectors."""
    R1 = rotvec_to_matrix(r1)
    R2 = rotvec_to_matrix(r2)
    cR = np.einsum("...ij,...jk->...ik", R1, R2)
    return matrix_to_rotvec(cR)


def rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    return Rotation.from_rotvec(rotvec).as_matrix()


def matrix_to_rotvec(mat: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(mat).as_rotvec()


def rotvec_to_quat(rotvec: np.ndarray) -> np.ndarray:
    return Rotation.from_rotvec(rotvec).as_quat()


def quat_to_rotvec(quat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # w > 0 to ensure 0 <= angle <= pi
    flip = (quat[..., :1] < 0).float()
    quat = (-1 * quat) * flip + (1 - flip) * quat

    angle = 2 * torch.atan2(torch.linalg.norm(quat[..., 1:], dim=-1), quat[..., 0])

    angle2 = angle * angle
    small_angle_scales = 2 + angle2 / 12 + 7 * angle2 * angle2 / 2880
    large_angle_scales = angle / torch.sin(angle / 2 + eps)

    small_angles = (angle <= 1e-3).float()
    rot_vec_scale = (
        small_angle_scales * small_angles + (1 - small_angles) * large_angle_scales
    )
    rot_vec = rot_vec_scale[..., None] * quat[..., 1:]
    return rot_vec


def quat_to_rotmat(quat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    rot_vec = quat_to_rotvec(quat, eps)
    return so3_utils.rot_mat_from_axis_angle_by_exp_map(rot_vec)


def rigid_transform_3D(
    A: np.ndarray, B: np.ndarray, verbose: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    # Transforms A to look like B
    # https://github.com/nghiaho12/rigid_transform_3D
    assert A.shape == B.shape
    A = A.T
    B = B.T

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    # if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    reflection_detected = False
    if np.linalg.det(R) < 0:
        if verbose:
            logger.info("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2, :] *= -1
        R = Vt.T @ U.T
        reflection_detected = True

    t = -R @ centroid_A + centroid_B
    optimal_A = R @ A + t

    return optimal_A.T, R, t, reflection_detected
