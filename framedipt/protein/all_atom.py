"""Utilities for calculating all atom representations."""
from __future__ import annotations

import torch

from framedipt.protein import residue_constants
from openfold.data import data_transforms
from openfold.utils import feats, rigid_utils

# Residue Constants from OpenFold/AlphaFold2.
IDEALIZED_POS37 = torch.tensor(residue_constants.restype_atom37_rigid_group_positions)
IDEALIZED_POS37_MASK = torch.any(IDEALIZED_POS37, dim=-1)
IDEALIZED_POS = torch.tensor(residue_constants.restype_atom14_rigid_group_positions)
DEFAULT_FRAMES = torch.tensor(residue_constants.restype_rigid_group_default_frame)
ATOM_MASK = torch.tensor(residue_constants.restype_atom14_mask)
GROUP_IDX = torch.tensor(residue_constants.restype_atom14_to_rigid_group)


def torsion_angles_to_frames(
    r: rigid_utils.Rigid,
    alpha: torch.Tensor,
    aatype: torch.Tensor,
) -> rigid_utils.Rigid:
    """Conversion method of torsion angles to frames provided the backbone.

    Args:
        r: Backbone rigid groups.
        alpha: Torsion angles.
        aatype: residue types.

    Returns:
        All 8 frames corresponding to each torsion frame.

    """
    # [*, N, 8, 4, 4]
    default_4x4 = DEFAULT_FRAMES.to(r.device)[aatype, ...]

    # [*, N, 8] transformations, i.e.
    #   One [*, N, 8, 3, 3] rotation matrix and
    #   One [*, N, 8, 3]    translation matrix
    default_r = r.from_tensor_4x4(default_4x4)

    bb_rot = alpha.new_zeros((*((1,) * len(alpha.shape[:-1])), 2))
    bb_rot[..., 1] = 1

    # [*, N, 8, 2]
    alpha = torch.cat([bb_rot.expand(*alpha.shape[:-2], -1, -1), alpha], dim=-2)

    # [*, N, 8, 3, 3]
    # Produces rotation matrices of the form:
    # [
    #   [1, 0  , 0  ],
    #   [0, a_2,-a_1],
    #   [0, a_1, a_2]
    # ]
    # This follows the original code rather than the supplement, which uses
    # different indices.

    all_rots = alpha.new_zeros(default_r.get_rots().get_rot_mats().shape)
    all_rots[..., 0, 0] = 1
    all_rots[..., 1, 1] = alpha[..., 1]
    all_rots[..., 1, 2] = -alpha[..., 0]
    all_rots[..., 2, 1:] = alpha

    all_rots_rigid = rigid_utils.Rigid(rigid_utils.Rotation(rot_mats=all_rots), None)

    all_frames = default_r.compose(all_rots_rigid)

    chi2_frame_to_frame = all_frames[..., 5]
    chi3_frame_to_frame = all_frames[..., 6]
    chi4_frame_to_frame = all_frames[..., 7]

    chi1_frame_to_bb = all_frames[..., 4]
    chi2_frame_to_bb = chi1_frame_to_bb.compose(chi2_frame_to_frame)
    chi3_frame_to_bb = chi2_frame_to_bb.compose(chi3_frame_to_frame)
    chi4_frame_to_bb = chi3_frame_to_bb.compose(chi4_frame_to_frame)

    all_frames_to_bb = rigid_utils.Rigid.cat(
        [
            all_frames[..., :5],
            chi2_frame_to_bb.unsqueeze(-1),
            chi3_frame_to_bb.unsqueeze(-1),
            chi4_frame_to_bb.unsqueeze(-1),
        ],
        dim=-1,
    )

    all_frames_to_global = r[..., None].compose(all_frames_to_bb)

    return all_frames_to_global


def prot_to_torsion_angles(
    aatype: torch.Tensor, atom37: torch.Tensor, atom37_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate torsion angle features from protein features."""
    prot_feats = {
        "aatype": aatype,
        "all_atom_positions": atom37,
        "all_atom_mask": atom37_mask,
    }
    torsion_angles_feats = data_transforms.atom37_to_torsion_angles()(prot_feats)
    torsion_angles = torsion_angles_feats["torsion_angles_sin_cos"]
    torsion_mask = torsion_angles_feats["torsion_angles_mask"]
    return torsion_angles, torsion_mask


def frames_to_atom14_pos(
    r: rigid_utils.Rigid,
    aatype: torch.Tensor,
) -> torch.Tensor:
    """Convert frames to their idealized all atom representation.

    Args:
        r: All rigid groups. [..., N, 8, 3]
        aatype: Residue types. [..., N]

    Returns:

    """
    # [*, N, 14]
    group_mask = GROUP_IDX.to(r.device)[aatype, ...]

    # [*, N, 14, 8]
    group_mask = torch.nn.functional.one_hot(
        group_mask,
        num_classes=DEFAULT_FRAMES.shape[-3],
    ).to(r.device)

    # [*, N, 14, 8]
    t_atoms_to_global = r[..., None, :] * group_mask

    # [*, N, 14]
    t_atoms_to_global = t_atoms_to_global.map_tensor_fn(lambda x: torch.sum(x, dim=-1))

    # [*, N, 14, 1]
    frame_atom_mask = ATOM_MASK.to(r.device)[aatype, ...].unsqueeze(-1)

    # [*, N, 14, 3]
    frame_null_pos = IDEALIZED_POS.to(r.device)[aatype, ...]
    pred_positions = t_atoms_to_global.apply(frame_null_pos)
    pred_positions = pred_positions * frame_atom_mask

    return pred_positions


def compute_backbone(
    bb_rigids: rigid_utils.Rigid,
    psi_torsions: torch.Tensor,
    aatype: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tile_dims = [1 for _ in range(len(bb_rigids.shape))] + [7, 1]
    torsion_angles = torch.tile(
        psi_torsions[..., None, :],
        tile_dims,
    ).to(bb_rigids.device)

    default_aatype = torch.zeros(bb_rigids.shape).long().to(bb_rigids.device)
    if aatype is None:
        aatype = default_aatype

    aatype = aatype.to(bb_rigids.device)
    aatype = torch.where(aatype == 20, default_aatype, aatype)

    all_frames = feats.torsion_angles_to_frames(
        bb_rigids, torsion_angles, aatype, DEFAULT_FRAMES.to(bb_rigids.device)
    )
    atom14_pos = frames_to_atom14_pos(all_frames, aatype)
    atom37_bb_pos = torch.zeros((*bb_rigids.shape, 37, 3))
    # atom14 bb order = ['N', 'CA', 'C', 'O', 'CB']
    # atom37 bb order = ['N', 'CA', 'C', 'CB', 'O']
    atom37_bb_pos[..., :3, :] = atom14_pos[..., :3, :]
    atom37_bb_pos[..., 3, :] = atom14_pos[..., 4, :]
    atom37_bb_pos[..., 4, :] = atom14_pos[..., 3, :]
    atom37_mask = torch.any(atom37_bb_pos, dim=-1)
    return atom37_bb_pos, atom37_mask, aatype, atom14_pos


def calculate_neighbor_angles(v_ac: torch.Tensor, v_ab: torch.Tensor) -> torch.Tensor:
    """Calculate angles between atoms c <- a -> b.

    Parameters
    ----------
        v_ac: Tensor, shape = (N,3)
            Vector from atom a to c.
        v_ab: Tensor, shape = (N,3)
            Vector from atom a to b.

    Returns
    -------
        angle_cab: Tensor, shape = (N,)
            Angle between atoms c <- a -> b.
    """
    # cos(alpha) = (u * v) / (|u|*|v|)
    x = torch.sum(v_ac * v_ab, dim=1)  # shape = (N,)
    # sin(alpha) = |u x v| / (|u|*|v|)
    y = torch.cross(v_ac, v_ab).norm(dim=-1)  # shape = (N,)
    # avoid that for y == (0,0,0) the gradient wrt. y becomes NaN
    y = torch.max(y, torch.tensor(1e-9))
    angle = torch.atan2(y, x)
    return angle


def vector_projection(v_ab: torch.Tensor, v_n: torch.Tensor) -> torch.Tensor:
    """
    Project the vector v_ab onto a plane with normal vector v_n.

    Parameters
    ----------
        v_ab: Tensor, shape = (N,3)
            Vector from atom a to b.
        v_n: Tensor, shape = (N,3)
            Normal vector of a plane onto which to project v_ab.

    Returns
    -------
        v_ab_proj: Tensor, shape = (N,3)
            Projected vector (orthogonal to v_n).
    """
    a_x_b = torch.sum(v_ab * v_n, dim=-1)
    b_x_b = torch.sum(v_n * v_n, dim=-1)
    return v_ab - (a_x_b / b_x_b)[:, None] * v_n
