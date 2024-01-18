"""Evaluation metrics."""
from __future__ import annotations

import dataclasses
import pathlib
from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache
from typing import Any, TypeVar

import Bio
import mdtraj as md
import numpy as np
import numpy.typing as npt
import pandas as pd
from Bio.PDB import SASA, Model, Residue

from evaluation.utils.constants import EVAL_METRICS, TCR_CHAINS
from framedipt.data import parsers
from framedipt.data import utils as data_utils
from framedipt.data.utils import read_pdb
from framedipt.protein import residue_constants
from framedipt.tools.custom_type import NDArrayFloat


def backbone_rmsd(
    model_1: Model.Model,
    model_2: Model.Model,
    chains: list[str] | str,
    model_1_diffusion_region: list[tuple[int, int]] | tuple[int, int],
    model_2_diffusion_region: list[tuple[int, int]] | tuple[int, int],
) -> float:
    """Compute backbone RMSD between two models in the diffused region.

    Args:
        model_1: input model of PDB structure 1.
        model_2: input model of PDB structure 2.
        chains: chain id or list of chain ids to get backbone atoms.
        model_1_diffusion_region: tuple of start and end indexes
            or list of tuples indicating diffusion region in model_1.
        model_2_diffusion_region: tuple of start and end indexes
            or list of tuples indicating diffusion region in model_2.

    Returns:
        Backbone RMSD.
    """
    if isinstance(chains, str):
        chains = [chains]
    if isinstance(model_1_diffusion_region, tuple):
        model_1_diffusion_region = [model_1_diffusion_region]
    if isinstance(model_2_diffusion_region, tuple):
        model_2_diffusion_region = [model_2_diffusion_region]

    all_backbone_distances = []
    for chain_id, diffusion_region_1, diffusion_region_2 in zip(
        chains, model_1_diffusion_region, model_2_diffusion_region
    ):
        backbone_distances = get_backbone_delta(
            model_1, model_2, chain_id, diffusion_region_1, diffusion_region_2
        )
        all_backbone_distances.append(backbone_distances)

    all_backbone_distances = np.concatenate(all_backbone_distances)
    all_backbone_distances = all_backbone_distances.reshape((-1, 3))
    rmsd = np.sqrt(
        np.sum(np.square(all_backbone_distances)) / all_backbone_distances.shape[0]
    ).item()

    return rmsd


def chain_backbone_rmsd(
    model_1: Model.Model,
    model_2: Model.Model,
    chains: list[str] | str,
    model_1_diffusion_region: list[tuple[int, int]] | tuple[int, int],
    model_2_diffusion_region: list[tuple[int, int]] | tuple[int, int],
) -> dict[str, float]:
    """Compute backbone RMSD between two models in the diffused region.

    Args:
        model_1: input model of PDB structure 1.
        model_2: input model of PDB structure 2.
        chains: chain id or list of chain ids to get backbone atoms.
        starts: start index or list of start indexes
            of the chains to get backbone atoms.
        ends: end index or list of end indexes
            of the chains to get backbone atoms.

    Returns:
        Backbone RMSD.
    """
    rmsds = {}
    if isinstance(chains, str):
        chains = [chains]
    if isinstance(model_1_diffusion_region, tuple):
        model_1_diffusion_region = [model_1_diffusion_region]
    if isinstance(model_2_diffusion_region, tuple):
        model_2_diffusion_region = [model_2_diffusion_region]
    for tcr_chain, chain_id, model_1_diff, model_2_diff in zip(
        TCR_CHAINS, chains, model_1_diffusion_region, model_2_diffusion_region
    ):
        distances = get_backbone_delta(
            model_1, model_2, chain_id, model_1_diff, model_2_diff
        )
        distances = distances.reshape((-1, 3))
        rmsd = np.sqrt(np.sum(np.square(distances)) / distances.shape[0]).item()
        rmsds[tcr_chain] = rmsd
    return rmsds


def get_backbone_delta(
    model_1: Model.Model,
    model_2: Model.Model,
    chain_id: str,
    model_1_region: tuple[int, int],
    model_2_region: tuple[int, int],
) -> np.ndarray:
    """Compute the atom-wise offset in the specified regions between two Models.

    Args:
        model_1: input model of PDB model 1. (typically ground truth)
        model_2: input model of PDB model 2. (typically ground truth)
        chain_id: which chain to get backbone atoms from.
        model_1_region: diffused region tuple (start, end)
            representing the start and end residue indices
        model_2_region: diffused region tuple (start, end)
            representing the start and end residue indices.
    """
    backbone_coords_1 = get_backbone_atom_coords(
        model=model_1,
        chain_id=chain_id,
        start=model_1_region[0],
        end=model_1_region[1],
    )
    backbone_coords_2 = get_backbone_atom_coords(
        model=model_2,
        chain_id=chain_id,
        start=model_2_region[0],
        end=model_2_region[1],
    )

    distances: np.ndarray = backbone_coords_1 - backbone_coords_2
    return distances


def residue_backbone_rmsd(
    model_1: Model.Model,
    model_2: Model.Model,
    chains: list[str] | str,
    model_1_diffusion_region: list[tuple[int, int]] | tuple[int, int],
    model_2_diffusion_region: list[tuple[int, int]] | tuple[int, int],
) -> dict[str, dict[int, float]]:
    """Compute backbone RMSD between two models in the diffused region.

    Args:
        model_1: input model of PDB structure 1.
        model_2: input model of PDB structure 2.
        chains: chain id or list of chain ids to get backbone atoms.
        model_1_diffusion_region: (list of) diffused region tuples (start, end)
            representing the start and end residue indices
        model_2_diffusion_region: (list of) diffused region tuples (start, end)
            representing the start and end residue indices

    Returns:
        Backbone RMSD Per Residue.
    """
    rmsds = {}
    if isinstance(chains, str):
        chains = [chains]
    if isinstance(model_1_diffusion_region, tuple):
        model_1_diffusion_region = [model_1_diffusion_region]
    if isinstance(model_2_diffusion_region, tuple):
        model_2_diffusion_region = [model_2_diffusion_region]
    for tcr_chain, chain_id, model_1_diff, model_2_diff in zip(
        TCR_CHAINS, chains, model_1_diffusion_region, model_2_diffusion_region
    ):
        distances = get_backbone_delta(
            model_1, model_2, chain_id, model_1_diff, model_2_diff
        )
        rmsd = list(np.sqrt(np.mean(np.sum(np.square(distances), axis=-1), axis=-1)))
        rmsds[tcr_chain] = convert_to_eval_idx(rmsd)
    return rmsds


def full_atom_rmsd(
    model_1: Model.Model,
    model_2: Model.Model,
    chains: list[str] | str,
    model_1_diffusion_region: list[tuple[int, int]] | tuple[int, int],
    model_2_diffusion_region: list[tuple[int, int]] | tuple[int, int],
) -> float:
    """Calculate full atom RMSD between two aligned structures in the diffused region.

    Note: model_1 is assumed to be the ground truth model.

    Args:
        model_1: The first model (should be ground truth).
        model_2: The second model, aligned to the first.
        chains: chain id or list of chain ids to get backbone atoms.
        model_1_diffusion_region: tuple of start and end indexes
            or list of tuples indicating diffusion region in model_1.
        model_2_diffusion_region: tuple of start and end indexes
            or list of tuples indicating diffusion region in model_2.

    Returns:
        A list of per-residue RMSD values.

    Raises:
        ValueError if the structures have different numbers of residues
        or the sequence of residues in the diffused region are not
        the same.
    """
    if isinstance(chains, str):
        chains = [chains]
    if isinstance(model_1_diffusion_region, tuple):
        model_1_diffusion_region = [model_1_diffusion_region]
    if isinstance(model_2_diffusion_region, tuple):
        model_2_diffusion_region = [model_2_diffusion_region]

    chains_regions = zip(chains, model_1_diffusion_region, model_2_diffusion_region)

    atom1_coords = []
    atom2_coords = []
    for chain_id, diffusion_region_1, diffusion_region_2 in chains_regions:
        diffused_residues_1 = get_masked_residues(
            model_1, chain_id, start=diffusion_region_1[0], end=diffusion_region_1[1]
        )
        diffused_residues_2 = get_masked_residues(
            model_2, chain_id, start=diffusion_region_2[0], end=diffusion_region_2[1]
        )

        if len(diffused_residues_1) != len(diffused_residues_2):
            raise ValueError("The structures have different numbers of residues")

        for res1, res2 in zip(diffused_residues_1, diffused_residues_2):
            if res1.id[0] != res2.id[0]:
                raise ValueError("Res1 is not the same type as Res2.")

            atoms1 = [atom for atom in res1.get_atoms() if atom.element != "H"]
            atoms2 = [atom for atom in res2.get_atoms() if atom.element != "H"]

            # Sort by atom id.
            atoms1 = sorted(atoms1, key=lambda x: x.id)
            atoms2 = sorted(atoms2, key=lambda x: x.id)

            # Only calculate RMSD values for residues contained in the
            # ground truth structure (i.e. avoid missing atoms).
            i = 0
            j = 0
            while i < len(atoms1) and j < len(atoms2):
                atom1 = atoms1[i]
                atom2 = atoms2[j]
                if atom1.id == atom2.id:
                    atom1_coords.append(atom1.coord)
                    atom2_coords.append(atom2.coord)
                    i += 1
                j += 1

    atom1_coords = np.array(atom1_coords)
    atom2_coords = np.array(atom2_coords)

    rmsd = np.sqrt(
        np.sum(np.square(atom1_coords - atom2_coords)) / atom1_coords.shape[0]
    ).item()

    return rmsd


def get_masked_residues(
    model: Model.Model, chain_id: str, start: int, end: int
) -> list[Bio.PDB.Residue]:
    """Get residues from a specific chain and region of a structure.

    Note: non-standard and missing residues will be skipped.

    Args:
        model: The structure model from which to extract residues.
            Assumes it is the first model of the structure.
        chain_id: The ID of the chain from which to extract residues.
        start: The starting residue number of the region (inclusive).
        end: The ending residue number of the region (inclusive).

    Returns:
        A list of residues in the specified region.
    """
    chain = model[chain_id]
    residues = [residue for residue in chain if Bio.PDB.is_aa(residue)]

    return residues[start : end + 1]


def angle_error(deg1: NDArrayFloat, deg2: NDArrayFloat) -> NDArrayFloat:
    """Compute absolute angle error.

    Args:
        deg1: angle or array of angles 1.
        deg2: angle or array of angles 2.

    Returns:
        Positive angle errors.
    """
    return np.minimum(
        np.minimum(np.abs(deg1 - deg2), np.abs(deg1 + 360 - deg2)),
        np.abs(deg1 - 360 - deg2),
    )


def angle_error_with_sign(deg1: NDArrayFloat, deg2: NDArrayFloat) -> NDArrayFloat:
    """Compute signed angle error.

    It's not the absolute angle error, but could be positive or negative.

    Args:
        deg1: angle or array of angles 1.
        deg2: angle or array of angles 2.

    Returns:
        Positive or negative angle errors.
    """
    angle_diffs = np.stack([deg1 - deg2, deg1 + 360 - deg2, deg1 - 360 - deg2], axis=0)
    all_angle_errors = np.stack(
        [np.abs(deg1 - deg2), np.abs(deg1 + 360 - deg2), np.abs(deg1 - 360 - deg2)],
        axis=0,
    )
    arg_idxs = np.argmin(all_angle_errors, axis=0)
    if len(angle_diffs.shape) == 1:
        return angle_diffs[arg_idxs]
    res = []
    for i in range(angle_diffs.shape[1]):
        res.append(angle_diffs[arg_idxs[i], i])
    return np.array(res)


def residue_asa_error(
    model_1: Model.Model,
    model_2: Model.Model,
    chains: list[str] | str,
    model_1_diffusion_region: list[tuple[int, int]] | tuple[int, int],
    model_2_diffusion_region: list[tuple[int, int]] | tuple[int, int],
) -> dict[str, dict[int, float]]:
    """Compute per-residue absolute soluble error between models in the diffused region.

    Args:
        model_1: input model of PDB structure 1.
        model_2: input model of PDB structure 2.
        chains: chain id or list of chain ids to get backbone atoms.
        model_1_diffusion_region: (list of) diffused region tuples (start, end)
            representing the start and end residue indices
        model_2_diffusion_region: (list of) diffused region tuples (start, end)
            representing the start and end residue indices

    Returns:
        Absolute Soluble Area Error (difference between two vals) Per Residue.
    """
    starts, ends = (tuple(x) for x in zip(*model_1_diffusion_region))
    asa1 = {
        k: convert_to_eval_idx(v)
        for k, v in get_sasa(model_1, tuple(chains), starts, ends)[0].items()
    }
    starts, ends = (tuple(x) for x in zip(*model_2_diffusion_region))
    asa2 = {
        k: convert_to_eval_idx(v)
        for k, v in get_sasa(model_2, tuple(chains), starts, ends)[0].items()
    }

    asa_error = {}
    for chain, residues1 in asa1.items():
        residues2 = asa2[chain]

        errors = {k: (residues1[k] - residues2[k]) for k in residues1}
        asa_error[chain] = errors

    return asa_error


def residue_rsa_error(
    model_1: Model.Model,
    model_2: Model.Model,
    chains: list[str] | str,
    model_1_diffusion_region: list[tuple[int, int]] | tuple[int, int],
    model_2_diffusion_region: list[tuple[int, int]] | tuple[int, int],
) -> dict[str, dict[int, float]]:
    """Compute per-residue relative soluble error between models in the diffused region.

    Args:
        model_1: input model of PDB structure 1.
        model_2: input model of PDB structure 2.
        chains: chain id or list of chain ids to get backbone atoms.
        model_1_diffusion_region: (list of) diffused region tuples (start, end)
            representing the start and end residue indices
        model_2_diffusion_region: (list of) diffused region tuples (start, end)
            representing the start and end residue indices

    Returns:
        Relative Soluble Area Error (difference between two vals) per Residue.
    """
    starts, ends = (tuple(x) for x in zip(*model_1_diffusion_region))

    rsa1 = {
        k: convert_to_eval_idx(v)
        for k, v in get_sasa(model_1, tuple(chains), starts, ends)[1].items()
    }
    starts, ends = (tuple(x) for x in zip(*model_2_diffusion_region))
    rsa2 = {
        k: convert_to_eval_idx(v)
        for k, v in get_sasa(model_2, tuple(chains), starts, ends)[1].items()
    }

    rsa_error: dict[str, dict[int, float]] = {}
    for chain, residues1 in rsa1.items():
        residues2 = rsa2[chain]

        errors = {k: (residues1[k] - residues2[k]) for k in residues1}
        rsa_error[chain] = errors

    return rsa_error


def gt_asa(  # pylint: disable=unused-argument
    model_1: Model.Model,
    model_2: Model.Model,  # noqa: ARG001
    chains: list[str] | str,
    model_1_diffusion_region: list[tuple[int, int]] | tuple[int, int],
    model_2_diffusion_region: list[tuple[int, int]] | tuple[int, int],  # noqa: ARG001
) -> dict[str, dict[int, float]]:
    """Compute Absolute Soluble Area of model 1 in the diffused region.

    Args:
        model_1: input model of PDB structure 1.
        model_2: input model of PDB structure 2.
            Unused, kept to maintain function signature of metric.
        chains: chain id or list of chain ids to get backbone atoms.
        model_1_diffusion_region: (list of) diffused region tuples (start, end)
            representing the start and end residue indices.
        model_2_diffusion_region: (list of) diffused region tuples (start, end)
            representing the start and end residue indices. Unused, kept to maintain
            function signature of metric.

    Returns:
        Absolute Soluble Area per Residue.
    """
    starts, ends = (tuple(x) for x in zip(*model_1_diffusion_region))

    asa = {
        k: convert_to_eval_idx(v)
        for k, v in get_sasa(model_1, tuple(chains), starts, ends)[0].items()
    }
    return asa


def gt_rsa(  # pylint: disable=unused-argument
    model_1: Model.Model,
    model_2: Model.Model,  # noqa: ARG001
    chains: list[str] | str,
    model_1_diffusion_region: list[tuple[int, int]] | tuple[int, int],
    model_2_diffusion_region: list[tuple[int, int]] | tuple[int, int],  # noqa: ARG001
) -> dict[str, dict[int, float]]:
    """Compute Relative Soluble Area of model 1 in the diffused region.

    Args:
        model_1: input model of PDB structure 1.
        model_2: input model of PDB structure 2.
            Unused, kept to maintain function signature of metric.
        chains: chain id or list of chain ids to get backbone atoms.
        model_1_diffusion_region: (list of) diffused region tuples (start, end)
            representing the start and end residue indices.
        model_2_diffusion_region: (list of) diffused region tuples (start, end)
            representing the start and end residue indices. Unused, kept to maintain
            function signature of metric.

    Returns:
        Relative Soluble Area per Residue.
    """
    starts, ends = (tuple(x) for x in zip(*model_1_diffusion_region))

    return {
        k: convert_to_eval_idx(v)
        for k, v in get_sasa(model_1, tuple(chains), starts, ends)[1].items()
    }


def sample_asa(  # pylint: disable=unused-argument
    model_1: Model.Model,  # noqa: ARG001
    model_2: Model.Model,
    chains: list[str] | str,
    model_1_diffusion_region: list[tuple[int, int]] | tuple[int, int],  # noqa: ARG001
    model_2_diffusion_region: list[tuple[int, int]] | tuple[int, int],
) -> dict[str, dict[int, float]]:
    """Compute Absolute Soluble Area of model 2 in the diffused region.

    Args:
        model_1: input model of PDB structure 1.
            Unused, kept to maintain function signature of metric.
        model_2: input model of PDB structure 2.
        chains: chain id or list of chain ids to get backbone atoms.
        model_1_diffusion_region: (list of) diffused region tuples (start, end)
            representing the start and end residue indices. Unused, kept to maintain
            function signature of metric.
        model_2_diffusion_region: (list of) diffused region tuples (start, end)
            representing the start and end residue indices.

    Returns:
        Absolute Soluble Area per Residue.
    """
    starts, ends = (tuple(x) for x in zip(*model_2_diffusion_region))
    return {
        k: convert_to_eval_idx(v)
        for k, v in get_sasa(model_2, tuple(chains), starts, ends)[0].items()
    }


def sample_rsa(  # pylint: disable=unused-argument
    model_1: Model.Model,  # noqa: ARG001
    model_2: Model.Model,
    chains: list[str] | str,
    model_1_diffusion_region: list[tuple[int, int]] | tuple[int, int],  # noqa: ARG001
    model_2_diffusion_region: list[tuple[int, int]] | tuple[int, int],
) -> dict[str, dict[int, float]]:
    """Compute Relative Soluble Area of model 2 in the diffused region.

    Args:
        model_1: input model of PDB structure 1.
            Unused, kept to maintain function signature of metric.
        model_2: input model of PDB structure 2.
        chains: chain id or list of chain ids to get backbone atoms.
        model_1_diffusion_region: (list of) diffused region tuples (start, end)
            representing the start and end residue indices. Unused, kept to maintain
            function signature of metric.
        model_2_diffusion_region: (list of) diffused region tuples (start, end)
            representing the start and end residue indices.

    Returns:
        Relative Soluble Area per Residue.
    """
    starts, ends = (tuple(x) for x in zip(*model_2_diffusion_region))
    return {
        k: convert_to_eval_idx(v)
        for k, v in get_sasa(model_2, tuple(chains), starts, ends)[1].items()
    }


def asa_abs_error(
    model_1: Model.Model,
    model_2: Model.Model,
    chains: list[str] | str,
    model_1_diffusion_region: list[tuple[int, int]] | tuple[int, int],
    model_2_diffusion_region: list[tuple[int, int]] | tuple[int, int],
) -> dict[str, dict[int, float]]:
    """Compute per-residue the absolute value of asa error in the diffused regions.

    Args:
        model_1: input model of PDB structure 1.
        model_2: input model of PDB structure 2.
        chains: chain id or list of chain ids to get backbone atoms.
        model_1_diffusion_region: (list of) diffused region tuples (start, end)
            representing the start and end residue indices
        model_2_diffusion_region: (list of) diffused region tuples (start, end)
            representing the start and end residue indices

    Returns:
        Absolute asa_error (absolute difference between the two asa values) Per Residue.
    """
    errors = residue_asa_error(
        model_1, model_2, chains, model_1_diffusion_region, model_2_diffusion_region
    )
    abs_error = {k: {k2: abs(v2) for k2, v2 in v.items()} for k, v in errors.items()}
    return abs_error


def rsa_abs_error(
    model_1: Model.Model,
    model_2: Model.Model,
    chains: list[str] | str,
    model_1_diffusion_region: list[tuple[int, int]] | tuple[int, int],
    model_2_diffusion_region: list[tuple[int, int]] | tuple[int, int],
) -> dict[str, dict[int, float]]:
    """Compute per-residue the absolute value of rsa error in the diffused regions.

    Args:
        model_1: input model of PDB structure 1.
        model_2: input model of PDB structure 2.
        chains: chain id or list of chain ids to get backbone atoms.
        model_1_diffusion_region: (list of) diffused region tuples (start, end)
            representing the start and end residue indices
        model_2_diffusion_region: (list of) diffused region tuples (start, end)
            representing the start and end residue indices

    Returns:
        Absolute rsa_error (absolute difference between the two rsa values) Per Residue.
    """
    errors = residue_rsa_error(
        model_1, model_2, chains, model_1_diffusion_region, model_2_diffusion_region
    )
    abs_error = {k: {k2: abs(v2) for k2, v2 in v.items()} for k, v in errors.items()}
    return abs_error


def asa_square_error(
    model_1: Model.Model,
    model_2: Model.Model,
    chains: list[str] | str,
    model_1_diffusion_region: list[tuple[int, int]] | tuple[int, int],
    model_2_diffusion_region: list[tuple[int, int]] | tuple[int, int],
) -> dict[str, dict[int, float]]:
    """Compute per-residue the squared value of asa error in the diffused regions.

    Args:
        model_1: input model of PDB structure 1.
        model_2: input model of PDB structure 2.
        chains: chain id or list of chain ids to get backbone atoms.
        model_1_diffusion_region: (list of) diffused region tuples (start, end)
            representing the start and end residue indices
        model_2_diffusion_region: (list of) diffused region tuples (start, end)
            representing the start and end residue indices

    Returns:
        squared asa_error (squared difference between the two asa values) Per Residue.
    """
    errors = residue_asa_error(
        model_1, model_2, chains, model_1_diffusion_region, model_2_diffusion_region
    )
    abs_error = {k: {k2: v2**2 for k2, v2 in v.items()} for k, v in errors.items()}
    return abs_error


def rsa_square_error(
    model_1: Model.Model,
    model_2: Model.Model,
    chains: list[str] | str,
    model_1_diffusion_region: list[tuple[int, int]] | tuple[int, int],
    model_2_diffusion_region: list[tuple[int, int]] | tuple[int, int],
) -> dict[str, dict[int, float]]:
    """Compute per-residue the squared value of rsa error in the diffused regions.

    Args:
        model_1: input model of PDB structure 1.
        model_2: input model of PDB structure 2.
        chains: chain id or list of chain ids to get backbone atoms.
        model_1_diffusion_region: (list of) diffused region tuples (start, end)
            representing the start and end residue indices
        model_2_diffusion_region: (list of) diffused region tuples (start, end)
            representing the start and end residue indices

    Returns:
        squared rsa_error (squared difference between the two rsa values) Per Residue.
    """
    errors = residue_rsa_error(
        model_1, model_2, chains, model_1_diffusion_region, model_2_diffusion_region
    )
    abs_error = {k: {k2: v2**2 for k2, v2 in v.items()} for k, v in errors.items()}
    return abs_error


def residue_asa_to_rsa(residue: Residue.Residue) -> float:
    """Return RSA value for a given `BioPython` `Residue`, converting from SASA.

    Args:
        residue (Residue): `BioPython` `Residue` object.

    Returns:
        float: RSA value for given `BioPython` `Residue` object.
    """
    max_sasa = residue_constants.MAX_SASAs[
        residue_constants.restype_3to1[residue.get_resname()]
    ]
    return residue.sasa / max_sasa


#  Max size = 3 means we'll only ever cache groundtruth, old sample and new sample vals,
#  reducing memory usage.
@lru_cache(maxsize=3)
def get_sasa(
    model: Model.Model,
    chains: tuple[str, ...] | str,
    starts: tuple[int, ...] | int,
    ends: tuple[int, ...] | int,
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Get residue-level SASA (Solvent Accessible Surface Area) of diffused region.

    Args:
        model: protein structure model.
        chains: chain id or list of chain ids to get SASA.
        starts: start index or list of start indexes
            of the chains to get SASA.
        ends: end index or list of end indexes
            of the chains to get SASA.

    Returns:
        Tuple of dictionary of SASAs ({chain: list of ASAs}, {chain: list of RSAs}).
    """
    # R means residue level: each residue will have a SASA value.
    SASA.ShrakeRupley().compute(model, level="R")

    if isinstance(chains, str):
        chains = (chains,)
    if isinstance(starts, int):
        starts = (starts,)
    if isinstance(ends, int):
        ends = (ends,)

    all_asas: dict[str, list[float]] = {}
    all_rsas: dict[str, list[float]] = {}
    for tcr_chain, chain_id, region_start_idx, region_end_idx in zip(
        TCR_CHAINS, chains, starts, ends
    ):
        chain = model[chain_id]
        residues = list(chain.get_residues())
        asas = []
        rsas = []
        for residue_idx in range(region_start_idx, region_end_idx + 1):
            residue = residues[residue_idx]
            asas.append(residue.sasa)
            rsas.append(residue_asa_to_rsa(residue))
        all_asas[tcr_chain] = asas
        all_rsas[tcr_chain] = rsas

    return all_asas, all_rsas


def square_error(v1: NDArrayFloat, v2: NDArrayFloat) -> NDArrayFloat:
    return np.square(v1 - v2)


def absolute_error(v1: NDArrayFloat, v2: NDArrayFloat) -> NDArrayFloat:
    return np.abs(v1 - v2)


def average_metrics_for_middle_residues(
    df_metrics: pd.DataFrame,
    metric: str,
) -> dict[str, list[npt.NDArray[float]]]:
    """Compute average metrics for middle residues.

    Args:
        df_metrics: dataframe containing pre-computed metrics.
        metric: the metric to compute, should be in `EVAL_METRICS`.

    Returns:
        Dictionary of computed metrics, with following keys
            - alpha: list of metrics per residue of alpha chain.
            - beta: list of metrics per residue of beta chain.

    Raises:
        NotImplementedError if `metric` is not in `EVAL_METRICS`.
    """
    if metric not in EVAL_METRICS:
        raise NotImplementedError(
            f"Metrics {metric} not implemented. It should be in {EVAL_METRICS}."
        )

    columns = df_metrics.columns
    metrics_alpha_beta: dict[str, list[npt.NDArray[float]]] = {}
    for tcr_chain in ["alpha", "beta"]:
        # Get columns of left side residue metrics,
        # with residue index in [1, 2, 3, 4].
        metrics_left_side_residue_columns = [
            f"{metric}_{tcr_chain}_{idx}" for idx in [1, 2, 3, 4]
        ]
        # Get columns of right side residue metrics,
        # with residue index in [-4, -3, -2, -1].
        metrics_right_side_residue_columns = [
            f"{metric}_{tcr_chain}_{idx}" for idx in [-4, -3, -2, -1]
        ]
        # Collect left side residue metrics.
        metrics_chain = [
            df_metrics[col].dropna().to_numpy()
            for col in metrics_left_side_residue_columns
        ]
        # Get columns of middle residue metrics.
        metrics_middle_residue_columns = [
            col
            for col in columns
            if col.startswith(f"{metric}_{tcr_chain}")
            and col
            not in metrics_left_side_residue_columns
            + metrics_right_side_residue_columns
        ]
        metrics_middle_raw = df_metrics[metrics_middle_residue_columns].to_numpy()
        # Compute the average metric for middle residues.
        metrics_middle = np.nanmean(metrics_middle_raw, axis=1)
        metrics_chain.append(metrics_middle[~np.isnan(metrics_middle)])
        # Collect right side residue metrics.
        metrics_chain += [
            df_metrics[col].dropna().to_numpy()
            for col in metrics_right_side_residue_columns
        ]
        metrics_alpha_beta[tcr_chain] = metrics_chain

    return metrics_alpha_beta


def get_coil_helix_sheet(
    pdb_path: pathlib.Path | str,
    pdb_name: str,
    chains: list[str],
    starts: list[int],
    ends: list[int],
) -> tuple[int, int, int]:
    """Get coil, helix, sheet lengths in the diffused region.

    Args:
        pdb_path: path to the PDB file.
        pdb_name: PDB name.
        chains: list of chain ids to get coil/helix/sheet info.
        starts: list of start indexes of chains to get coil/helix/sheet info.
        ends: list of end indexes of chains to get coil/helix/sheet info.

    Returns:
        Tuple of coil, helix, sheet lengths.
    """
    # MDtraj
    traj = md.load(pdb_path)
    # SS calculation
    pdb_ss = md.compute_dssp(traj, simplified=True)[0]

    model = read_pdb(pdb_path, pdb_name=pdb_name)
    model_chains = list(model.child_dict.keys())
    curr_chain = 0
    offset = 0
    coil = 0
    helix = 0
    sheet = 0
    for chain_id, start, end in zip(chains, starts, ends):
        model_chain = model_chains[curr_chain]
        while curr_chain < len(model_chains) and model_chain != chain_id:
            offset += len(list(model[model_chain].get_residues()))
            curr_chain += 1
            model_chain = model_chains[curr_chain]
        pdb_start_idx = offset + start
        pdb_end_idx = offset + end + 1
        coil += np.sum(pdb_ss[pdb_start_idx:pdb_end_idx] == "C")
        helix += np.sum(pdb_ss[pdb_start_idx:pdb_end_idx] == "H")
        sheet += np.sum(pdb_ss[pdb_start_idx:pdb_end_idx] == "E")
        offset += len(list(model[model_chain].get_residues()))
        curr_chain += 1

    return coil, helix, sheet


def get_backbone_atom_coords(
    model: Model.Model,
    chain_id: str,
    start: int | None,
    end: int | None,
) -> np.ndarray:
    """Get backbone atoms' coordinates.

    Args:
        model: model of a PDB structure.
        chain_id: id of the chain to get backbone atoms.
        start: start index of the chain to get backbone atoms.
        end: end index of the chain to get backbone atoms.

    Returns:
        Backbone atoms' coordinates, shape [N_atoms, 3].
    """
    if chain_id not in model.child_dict:
        raise ValueError(f"Chain {chain_id} does not exist in model.")

    chain = model[chain_id]
    chain_id_int = data_utils.chain_str_to_int(chain_id)
    chain_prot = parsers.process_chain(chain, chain_id_int)
    chain_dict = dataclasses.asdict(chain_prot)

    if start is None:
        start = 0
    if end is None:
        end = chain_dict["atom_positions"].shape[0] - 1

    backbone_idxs = [
        residue_constants.atom_order[bb_atom]
        for bb_atom in residue_constants.BACKBONE_ATOMS
    ]
    backbone_coords = chain_dict["atom_positions"][start : end + 1, backbone_idxs]

    return backbone_coords


# calculate dihedral angles defined by 4 sets of points
def dihedrals(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Compute dihedral angles given four arrays of 3D points.

    Reference: https://leimao.github.io/blog/Dihedral-Angles/
        https://stackoverflow.com/questions/20305272/dihedral-
        torsion-angle-from-four-points-in-cartesian-coordinates-in-python

    Args:
        a: 3D coordinates of points a, shape [..., 3].
        b: 3D coordinates of points b, shape [..., 3].
        c: 3D coordinates of points c, shape [..., 3].
        d: 3D coordinates of points d, shape [..., 3].

    Returns:
        Computed dihedral angles in radius, shape [...].
    """
    # Flip the first vector so that eclipsing vectors have dihedral=0.
    b0: np.ndarray = a - b
    b1: np.ndarray = c - b
    b2: np.ndarray = d - c

    # Normalize b1 so that it does not influence magnitude of
    # vector rejections that come next.
    b1 /= np.linalg.norm(b1, axis=-1, keepdims=True)

    # Vector rejections.
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    dot_product_1: np.ndarray = np.sum(b0 * b1, axis=-1, keepdims=True)
    dot_product_2: np.ndarray = np.sum(b2 * b1, axis=-1, keepdims=True)
    v = b0 - dot_product_1 * b1
    w = b2 - dot_product_2 * b1

    # Angle between v and w in a plane is the torsion angle.
    # v and w may not be normalized but that's fine since tan is y/x.
    x = np.sum(v * w, axis=-1)
    y = np.sum(np.cross(b1, v) * w, axis=-1)
    out = np.arctan2(y, x)
    if np.any(np.isnan(out)):
        raise ValueError("Found NaN values in computed dihedral angles.")

    return out


def calc_dihedrals(
    n_coords: np.ndarray, ca_coords: np.ndarray, c_coords: np.ndarray
) -> dict[str, np.ndarray]:
    """Calculate dihedral angles in protein structure.

    psi is the dihedral angle over N_i - CA_i - C_i - N_i+1,
        value 0 is added at the end.
    omega is the dihedral angle over CA_i - C_i - N_i+1 - CA_i+1,
        value 0 is added at the end.
    phi is the dihedral angle over C_i-1 - N_i - CA_i - C_i.
        value 0 is added at the beginning.

    Args:
        n_coords: 3D coordinates of N atoms, shape [N_res, 3].
        ca_coords: 3D coordinates of CA atoms, shape [N_res, 3].
        c_coords: 3D coordinates of C atoms, shape [N_res, 3].

    Returns:
        Dictionary of dihedral angles psi, omega and phi, all of shape [N_res,].
    """
    psi = dihedrals(n_coords[:-1], ca_coords[:-1], c_coords[:-1], n_coords[1:])
    # Add 0 at the end because the residue at C-terminus does not have psi angle.
    psi = np.append(psi, [0.0])
    omega = dihedrals(ca_coords[:-1], c_coords[:-1], n_coords[1:], ca_coords[1:])
    # Add 0 at the end because the residue at C-terminus does not have omega angle.
    omega = np.append(omega, [0.0])
    phi = dihedrals(c_coords[:-1], n_coords[1:], ca_coords[1:], c_coords[1:])
    # Add 0 at the beginning because the residue at N-terminus does not have phi angle.
    phi = np.append([0.0], phi)

    return {"psi": psi, "omega": omega, "phi": phi}


#  Max size = 3 means we'll only ever cache groundtruth, old sample and new sample vals,
#  reducing memory usage.
@lru_cache(maxsize=3)
def get_dihedral_angles(
    model: Model.Model,
    chains: tuple[str, ...] | str,
    starts: tuple[int, ...] | int,
    ends: tuple[int, ...] | int,
) -> dict[str, dict[str, np.ndarray]]:
    """Get dihedral angles of diffused region.

    Args:
        model: protein structure model.
        chains: chain id or list of chain ids to get dihedral angles.
        starts: start index or list of start indexes
            of the chains to get dihedral angles.
        ends: end index or list of end indexes
            of the chains to get dihedral angles.

    Returns:
        Dictionary of dihedral angles {chain: {dihedral angle: list of angles}}.
    """
    if isinstance(chains, str):
        chains = (chains,)
    if isinstance(starts, int):
        starts = (starts,)
    if isinstance(ends, int):
        ends = (ends,)

    chains_dihedral_angles: dict[str, dict[str, np.ndarray]] = {}
    for tcr_chain, chain_id, region_start_idx, region_end_idx in zip(
        TCR_CHAINS, chains, starts, ends
    ):
        chain = model[chain_id]
        chain_id_int = data_utils.chain_str_to_int(chain_id)
        chain_prot = parsers.process_chain(chain, chain_id_int)
        chain_dict = dataclasses.asdict(chain_prot)

        n_idx = residue_constants.atom_order["N"]
        ca_idx = residue_constants.atom_order["CA"]
        c_idx = residue_constants.atom_order["C"]

        n_coords = chain_dict["atom_positions"][:, n_idx]
        ca_coords = chain_dict["atom_positions"][:, ca_idx]
        c_coords = chain_dict["atom_positions"][:, c_idx]

        dihedral_angles = calc_dihedrals(
            n_coords=n_coords,
            ca_coords=ca_coords,
            c_coords=c_coords,
        )

        for angle_name, angles in dihedral_angles.items():
            dihedral_angles[angle_name] = np.rad2deg(
                angles[region_start_idx : region_end_idx + 1]
            )

        for angle_name, angles in dihedral_angles.items():
            if tcr_chain not in chains_dihedral_angles:
                chains_dihedral_angles[tcr_chain] = {}
            chains_dihedral_angles[tcr_chain][angle_name] = angles

    return chains_dihedral_angles


def get_dihedral_angles_new_eval_shim(
    model: Model.Model,
    chains: tuple[str, ...] | str,
    starts: tuple[int, ...] | int,
    ends: tuple[int, ...] | int,
) -> dict[str, dict[str, dict[int, float]]]:
    """A shim around `get_dihedral_angles`. Returns indexed values instead of an array.

    Args:
        model: protein structure model.
        chains: chain id or list of chain ids to get dihedral angles.
        starts: start index or list of start indexes
            of the chains to get dihedral angles.
        ends: end index or list of end indexes
            of the chains to get dihedral angles.

    Returns:
        Dictionary of dihedral angles {chain: {dihedral angle: {index: value}}}.

    """
    old_style_results = get_dihedral_angles(model, chains, starts, ends)
    new_style_results: dict[str, dict[str, dict[int, float]]] = {}
    for chain, entry in old_style_results.items():
        for angle, array in entry.items():
            if angle not in new_style_results:
                new_style_results[angle] = {}
            new_style_results[angle][chain] = convert_to_eval_idx(list(array))
    return new_style_results


def residue_angle_error(
    model_1: Model.Model,
    model_2: Model.Model,
    chains: list[str] | str,
    model_1_diffusion_region: list[tuple[int, int]] | tuple[int, int],
    model_2_diffusion_region: list[tuple[int, int]] | tuple[int, int],
) -> dict[str, dict[str, dict[int, float]]]:
    """Compute the absolute angular errors between dihedral angles in diffused regions.

    Args:
        model_1: input model of PDB structure 1.
        model_2: input model of PDB structure 2.
        chains: chain id or list of chain ids to get backbone atoms.
        model_1_diffusion_region: (list of) diffused region tuples (start, end)
            representing the start and end residue indices
        model_2_diffusion_region: (list of) diffused region tuples (start, end)
            representing the start and end residue indices

    Returns:
        dictionary of form {angle : {chain: {residue_idx: error}}} where:
            angle keys are "phi", "psi" and "omega"
            chain keys are "alpha" and "beta"
            residue_idx keys are from [1, 2, 3, 4, ..., -4, -3, -2, -1, ]
    """
    signed = residue_signed_angle_error(
        model_1, model_2, chains, model_1_diffusion_region, model_2_diffusion_region
    )
    unsigned = {
        k: {k2: {k3: abs(v3) for k3, v3 in v2.items()} for k2, v2 in v.items()}
        for k, v in signed.items()
    }
    return unsigned


def residue_signed_angle_error(
    model_1: Model.Model,
    model_2: Model.Model,
    chains: list[str] | str,
    model_1_diffusion_region: list[tuple[int, int]] | tuple[int, int],
    model_2_diffusion_region: list[tuple[int, int]] | tuple[int, int],
) -> dict[str, dict[str, dict[int, float]]]:
    """Compute the signed angular error between dihedral angles in the diffused regions.

    Args:
        model_1: input model of PDB structure 1.
        model_2: input model of PDB structure 2.
        chains: chain id or list of chain ids to get backbone atoms.
        model_1_diffusion_region: (list of) diffused region tuples (start, end)
            representing the start and end residue indices
        model_2_diffusion_region: (list of) diffused region tuples (start, end)
            representing the start and end residue indices

    Returns:
        dictionary of form {angle : {chain: {residue_idx: error}}} where:
            angle keys are "phi", "psi" and "omega"
            chain keys are "alpha" and "beta"
            residue_idx keys are from [1, 2, 3, 4, ..., -4, -3, -2, -1, ]
    """
    chains_tuple = tuple(chains)
    if isinstance(model_1_diffusion_region, tuple):
        model_1_diffusion_region = [model_1_diffusion_region]
    if isinstance(model_2_diffusion_region, tuple):
        model_2_diffusion_region = [model_2_diffusion_region]

    model_1_starts = tuple(x[0] for x in model_1_diffusion_region)
    model_1_ends = tuple(x[1] for x in model_1_diffusion_region)
    model_2_starts = tuple(x[0] for x in model_2_diffusion_region)
    model_2_ends = tuple(x[1] for x in model_2_diffusion_region)
    dihedral1 = get_dihedral_angles_new_eval_shim(
        model_1, chains_tuple, model_1_starts, model_1_ends
    )
    dihedral2 = get_dihedral_angles_new_eval_shim(
        model_2, chains_tuple, model_2_starts, model_2_ends
    )
    diff: dict[str, dict[str, dict[int, float]]] = {}
    for k, v in dihedral1.items():
        diff[k] = {}
        for k2, v2 in v.items():
            errs = angle_error_with_sign(
                np.array(list(v2.values())),
                np.array(list(dihedral2[k][k2].values())),
            )
            diff[k][k2] = dict(zip(v2, errs))
    return diff


def residue_sample_angle(  # pylint: disable=unused-argument
    model_1: Model.Model,  # noqa: ARG001
    model_2: Model.Model,
    chains: list[str] | str,
    model_1_diffusion_region: list[tuple[int, int]] | tuple[int, int],  # noqa: ARG001
    model_2_diffusion_region: list[tuple[int, int]] | tuple[int, int],
) -> dict[str, dict[str, dict[int, float]]]:
    """Compute the dihedral angles in the diffused regions of the sample model.

    Args:
        model_1: input model of PDB structure 1. Unused, kept to maintain function
        signature
        model_2: input model of PDB structure 2.
        chains: chain id or list of chain ids to get backbone atoms.
        model_1_diffusion_region: (list of) diffused region tuples (start, end)
            representing the start and end residue indices. Unused, kept to maintain
            function signature
        model_2_diffusion_region: (list of) diffused region tuples (start, end)
            representing the start and end residue indices.

    Returns:
        dictionary of form {angle : {chain: {residue_idx: error}}} where:
            angle keys are "phi", "psi" and "omega"
            chain keys are "alpha" and "beta"
            residue_idx keys are from [1, 2, 3, 4, ..., -4, -3, -2, -1, ]
    """
    chains_tuple = tuple(chains)
    if isinstance(model_2_diffusion_region, tuple):
        model_2_diffusion_region = [model_2_diffusion_region]
    starts = tuple(x[0] for x in model_2_diffusion_region)
    ends = tuple(x[1] for x in model_2_diffusion_region)
    dihedral2 = get_dihedral_angles_new_eval_shim(model_2, chains_tuple, starts, ends)
    return dihedral2


def residue_groundtruth_angle(  # pylint: disable=unused-argument
    model_1: Model.Model,
    model_2: Model.Model,  # noqa: ARG001
    chains: list[str] | str,
    model_1_diffusion_region: list[tuple[int, int]] | tuple[int, int],
    model_2_diffusion_region: list[tuple[int, int]] | tuple[int, int],  # noqa: ARG001
) -> dict[str, dict[str, dict[int, float]]]:
    """Compute the dihedral angles in the diffused regions of the ground truth model.

    Args:
        model_1: input model of PDB structure 1.
        model_2: input model of PDB structure 2. Unused, kept to maintain function
        signature
        chains: chain id or list of chain ids to get backbone atoms.
        model_1_diffusion_region: (list of) diffused region tuples (start, end)
            representing the start and end residue indices.
        model_2_diffusion_region: (list of) diffused region tuples (start, end)
            representing the start and end residue indices. Unused, kept to maintain
            function signature

    Returns:
        dictionary of form {angle : {chain: {residue_idx: error}}} where:
            angle keys are "phi", "psi" and "omega"
            chain keys are "alpha" and "beta"
            residue_idx keys are from [1, 2, 3, 4, ..., -4, -3, -2, -1, ]
    """
    chains_tuple = tuple(chains)
    if isinstance(model_1_diffusion_region, tuple):
        model_1_diffusion_region = [model_1_diffusion_region]
    starts = tuple(x[0] for x in model_1_diffusion_region)
    ends = tuple(x[1] for x in model_1_diffusion_region)
    dihedral1 = get_dihedral_angles_new_eval_shim(model_1, chains_tuple, starts, ends)
    return dihedral1


def flatten(
    obj: Any, depth: int = -1, delim: str = "_", parent: str = ""
) -> dict[str, Any]:
    """Flatten nested containers (list, dict etc.) recursively into a single dict.
    Args:
        obj: Mappable, Iterable or other object to flatten. Any non Mappable/Iterable
            obj (with nonzero depth) will result in a dictionary of {parent: obj}.
        depth: How many levels (counted from top) to flatten.
            if 0, no flattening will occur.
            if negative, all levels will be flattened.
        delim: How to delimit nested levels. i.e. a delim of "+" would result in
            {"key" : ["a", "b"]} being turned into: {"key+1": "a", "key+2": "b"}
        parent: The prefix to apply to all keys in the final dictionary.
    Returns:
        flattened dictionary of form dict[str,Any].
    """
    if depth == 0:
        return obj
    child_list: list[tuple[str, Any]] = []
    if isinstance(obj, Mapping):
        for key, val in obj.items():
            new_key = f"{parent}{delim}{key}" if parent else key
            child_list.extend(flatten(val, depth=depth - 1, parent=new_key).items())
    elif isinstance(obj, Iterable):
        for i, val in enumerate(obj):
            new_key = f"{parent}{delim}{i+1}"
            child_list.extend(flatten(val, depth=depth - 1, parent=new_key).items())
    else:
        child_list.append((parent, obj))
    return dict(child_list)


T = TypeVar("T")


def convert_to_eval_idx(vals: Sequence[T]) -> dict[int, T]:
    """Convert a sequence to a dictionary of index-value pairs for use in evaluation.

    Args:
        vals: A sequence (indexable like a list or tuple) of values

    Returns:
        dict of the form {-4: vals[-4], ... , -1: vals[-1], 1: vals[0] 2: vals[1],...}.
        dictionary keys do not "overlap". i.e. vals[-4] will only be accessible with
        the -4 key and cannot be accessed using a positively valued key.
    """
    val_dict = {}
    for idx in [-4, -3, -2, -1]:
        val_dict[idx] = vals[idx]
    for i, val in enumerate(vals[:-4]):
        val_dict[i + 1] = val
    return val_dict
