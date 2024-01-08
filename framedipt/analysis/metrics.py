"""Module of analysis metrics."""
from __future__ import annotations

import pathlib

import mdtraj as md
import numpy as np
import tree
from tmtools import tm_align

from framedipt.analysis import utils as analysis_utils
from framedipt.data import transforms
from framedipt.data import utils as data_utils
from openfold.np import residue_constants
from openfold.np.relax import amber_minimize

CA_IDX = residue_constants.atom_order["CA"]

INTER_VIOLATION_METRICS = [
    "bonds_c_n_loss_mean",
    "angles_ca_c_n_loss_mean",
    "clashes_mean_loss",
]

SHAPE_METRICS = [
    "coil_percent",
    "helix_percent",
    "strand_percent",
    "radius_of_gyration",
]

CA_VIOLATION_METRICS = [
    "ca_ca_bond_dev",
    "ca_ca_valid_percent",
    "ca_steric_clash_percent",
    "num_ca_steric_clashes",
]

EVAL_METRICS = [
    "tm_score",
]

ALL_METRICS = (
    INTER_VIOLATION_METRICS + SHAPE_METRICS + CA_VIOLATION_METRICS + EVAL_METRICS
)


def calc_tm_score(
    pos_1: np.ndarray, pos_2: np.ndarray, seq_1: str, seq_2: str
) -> tuple[float, float]:
    """Calculate TM score for two protein structures.

    Args:
        pos_1: array of residue coordinates of protein 1, shape [N_res, 3].
        pos_2: array of residue coordinates of protein 2, shape [N_res, 3].
        seq_1: AA sequence of protein 1.
        seq_2: AA sequence of protein 2.

    Returns:
        Tuple of TM scores by aligning w.r.t. protein 1 and protein 2.
            Notes: TM score is not symmetric.
    """
    tm_results = tm_align(pos_1, pos_2, seq_1, seq_2)
    return tm_results.tm_norm_chain1, tm_results.tm_norm_chain2


def calc_mdtraj_metrics(pdb_path: pathlib.Path) -> dict[str, float]:
    """Calculate protein structure plausibility metrics using mdtraj.

    Args:
        pdb_path: path to the pdb file.

    Returns:
        Dictionary of the following metrics:
            - non_coil_percent: percentage of non-coils in the structure.
            - coil_percent: percentage of coils in the structure.
            - helix_percent: percentage of alpha helix in the structure.
            - strand_percent: percentage of beta strand in the structure.
            - radius_of_gyration: the radius of gyration (Rg),
                which is the distribution of atoms of a protein around its axis.
    """
    traj = md.load(pdb_path)
    pdb_ss = md.compute_dssp(traj, simplified=True)
    pdb_coil_percent = np.mean(pdb_ss == "C").item()
    pdb_helix_percent = np.mean(pdb_ss == "H").item()
    pdb_strand_percent = np.mean(pdb_ss == "E").item()
    pdb_ss_percent = pdb_helix_percent + pdb_strand_percent
    pdb_rg = md.compute_rg(traj)[0]
    return {
        "non_coil_percent": pdb_ss_percent,
        "coil_percent": pdb_coil_percent,
        "helix_percent": pdb_helix_percent,
        "strand_percent": pdb_strand_percent,
        "radius_of_gyration": pdb_rg,
    }


def calc_aligned_rmsd(pos_1: np.ndarray, pos_2: np.ndarray) -> float:
    """Calculate aligned RMSD between two proteins.

    Args:
        pos_1: array of atom positions of protein 1, shape [N_atoms, 3].
        pos_2: array of atom positions of protein 2, shape [N_atoms, 3].

    Returns:
        RMSD between two aligned proteins.
    """
    aligned_pos_1 = transforms.rigid_transform_3D(pos_1, pos_2)[0]
    return np.mean(np.linalg.norm(aligned_pos_1 - pos_2, axis=-1)).item()


def protein_metrics(
    *,
    pdb_path: pathlib.Path,
    atom37_pos: np.ndarray,
    gt_atom37_pos: np.ndarray,
    gt_aatype: np.ndarray,
    diffuse_mask: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Calculate metrics for evaluating protein predictions.

    Args:
        pdb_path: path to the pdb file.
        atom37_pos: array of predicted atom positions, shape [N_res, 37, 3].
        gt_atom37_pos: array of ground truth atom positions, shape [N_res, 37, 3].
        gt_aatype: array of ground truth AA types, shape [N_res,].
        diffuse_mask: array of diffused mask, shape [N_res,].

    Returns:
        Dictionary of the following metrics:
            - ca_ca_bond_dev: mean of absolute deviation of CA-CA bond distances
                w.r.t. residue_constants.ca_ca.
            - ca_ca_valid_percent: percentage of valid CA-CA bond distances.
            - ca_steric_clash_percent: percentage of CA clashes.
            - num_ca_steric_clashes: number of CA clashes.
            - tm_score: TM score between the prediction and the ground truth.
            - bonds_c_n_loss_mean: mean C-N bond violation loss.
            - angles_ca_c_n_loss_mean: mean CA-C-N bond angle loss.
            - clashes_mean_loss: mean of non-bonded atom clash loss.
            - non_coil_percent: percentage of non-coils in the structure.
            - coil_percent: percentage of coils in the structure.
            - helix_percent: percentage of alpha helix in the structure.
            - strand_percent: percentage of beta strand in the structure.
            - radius_of_gyration: the radius of gyration (Rg),
                which is the distribution of atoms of a protein around its axis.
    """
    # SS percentage
    mdtraj_metrics = calc_mdtraj_metrics(pdb_path)
    atom37_mask = np.any(atom37_pos, axis=-1)
    atom37_diffuse_mask = diffuse_mask[..., None] * atom37_mask
    prot = analysis_utils.create_full_prot(atom37_pos, atom37_diffuse_mask)
    violation_metrics = amber_minimize.get_violation_metrics(prot)
    struct_violations = violation_metrics["structural_violations"]
    inter_violations = struct_violations["between_residues"]

    # Geometry
    bb_mask = np.any(atom37_mask, axis=-1)
    ca_pos = atom37_pos[..., CA_IDX, :][bb_mask.astype(bool)]
    ca_ca_bond_dev, ca_ca_valid_percent = ca_ca_distance(ca_pos)
    num_ca_steric_clashes, ca_steric_clash_percent = ca_ca_clashes(ca_pos)

    # Eval
    bb_diffuse_mask = (diffuse_mask * bb_mask).astype(bool)
    unpad_gt_scaffold_pos = gt_atom37_pos[..., CA_IDX, :][bb_diffuse_mask]
    unpad_pred_scaffold_pos = atom37_pos[..., CA_IDX, :][bb_diffuse_mask]
    seq = data_utils.aatype_to_seq(gt_aatype[bb_diffuse_mask])
    _, tm_score = calc_tm_score(
        unpad_pred_scaffold_pos, unpad_gt_scaffold_pos, seq, seq
    )

    metrics_dict = {
        "ca_ca_bond_dev": ca_ca_bond_dev,
        "ca_ca_valid_percent": ca_ca_valid_percent,
        "ca_steric_clash_percent": ca_steric_clash_percent,
        "num_ca_steric_clashes": num_ca_steric_clashes,
        "tm_score": tm_score,
        **mdtraj_metrics,
    }
    for k in INTER_VIOLATION_METRICS:
        metrics_dict[k] = inter_violations[k]
    metrics_dict = tree.map_structure(lambda x: np.mean(x).item(), metrics_dict)
    return metrics_dict


def ca_ca_distance(ca_pos: np.ndarray, tol: float = 0.1) -> tuple[float, float]:
    """Get carbon-alpha bond distance deviation and valid bond percentage.

    Args:
        ca_pos: array of carbon-alpha positions, shape [N_res, 3].
        tol: tolerance for valid distance, default to 0.1.
            The valid distances should be smaller than residue_constants.ca_ca + tol.

    Returns:
        ca_ca_dev: mean of absolute deviation of CA-CA bond distances
            w.r.t. residue_constants.ca_ca.
        ca_ca_valid: percentage of valid CA-CA bond distances.
    """
    ca_bond_dists = np.linalg.norm(ca_pos - np.roll(ca_pos, 1, axis=0), axis=-1)[1:]
    ca_ca_dev = np.mean(np.abs(ca_bond_dists - residue_constants.ca_ca)).item()
    ca_ca_valid = np.mean(ca_bond_dists < (residue_constants.ca_ca + tol)).item()
    return ca_ca_dev, ca_ca_valid


def ca_ca_clashes(ca_pos: np.ndarray, tol: float = 1.5) -> tuple[float, float]:
    """Get number and percentage of carbon-alpha clashes.

    Args:
        ca_pos: array of carbon-alpha positions, shape [N_res, 3].
        tol: tolerance for CA clashes, default to 1.5.

    Returns:
        Tuple of (number of clashes, percentage of clashes).
    """
    ca_ca_dists2d = np.linalg.norm(ca_pos[:, None, :] - ca_pos[None, :, :], axis=-1)
    inter_dists = ca_ca_dists2d[np.where(np.triu(ca_ca_dists2d, k=0) > 0)]
    clashes = inter_dists < tol
    clashes_sum: np.ndarray = np.sum(clashes)
    return clashes_sum.item(), np.mean(clashes).item()
