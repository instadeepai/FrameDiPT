"""Module of analysis utils."""
from __future__ import annotations

import os
import pathlib
import re

import numpy as np
from scipy.spatial.transform import Rotation

from framedipt.protein import protein, residue_constants
from openfold.utils import rigid_utils

CA_IDX = residue_constants.atom_order["CA"]
Rigid = rigid_utils.Rigid


def create_full_prot(
    atom37: np.ndarray,
    atom37_mask: np.ndarray,
    aatype: np.ndarray | None = None,
    b_factors: np.ndarray | None = None,
    residue_index: np.ndarray | None = None,
    chain_index: np.ndarray | None = None,
) -> protein.Protein:
    """Create protein using atom37 positions.

    Args:
        atom37: atom37 positions, shape [N_res, 37, 3].
        atom37_mask: atom37 mask, shape [N_res, 37].
        aatype: optional AA types, shape [N_res].
        b_factors: optional b-factors, shape [N_res, 37].
        residue_index: optional residue indices, shape [N_res,].
        chain_index: optional chain indices, shape [N_res,].

    Returns:
        Protein object.

    Raises:
        ValueError if atom37 is not a 3-dimensional array
            or it's not of shape [N_res, 37, 3].
    """
    if atom37.ndim != 3:
        raise ValueError(f"atom37 should be of dim 3, got {atom37.ndim}.")
    if atom37.shape[-1] != 3 or atom37.shape[-2] != 37:
        raise ValueError(f"atom37 should have shape [..., 37, 3], got {atom37.shape}.")
    n = atom37.shape[0]
    final_residue_index = np.arange(n)
    final_chain_index = np.zeros(n)
    if residue_index is not None and chain_index is not None:
        unique_chain_indexes = np.unique(chain_index)
        prev_residue_idx = 0
        for i, index in enumerate(unique_chain_indexes):
            curr_chain = chain_index[chain_index == index]
            curr_chain_len = len(curr_chain)
            final_chain_index[prev_residue_idx : prev_residue_idx + curr_chain_len] = i
            final_residue_index[
                prev_residue_idx : prev_residue_idx + curr_chain_len
            ] = np.arange(curr_chain_len)
            prev_residue_idx += curr_chain_len

    if b_factors is None:
        b_factors = np.zeros([n, 37])
    if aatype is None:
        aatype = np.zeros(n, dtype=np.int64)
    return protein.Protein(
        atom_positions=atom37,
        atom_mask=atom37_mask,
        aatype=aatype,
        residue_index=final_residue_index,
        chain_index=final_chain_index,
        b_factors=b_factors,
    )


def write_prot_to_pdb(
    prot_pos: np.ndarray,
    file_path: pathlib.Path | str,
    aatype: np.ndarray | None = None,
    overwrite: bool = False,
    no_indexing: bool = False,
    b_factors: np.ndarray | None = None,
    residue_index: np.ndarray | None = None,
    chain_index: np.ndarray | None = None,
) -> pathlib.Path:
    """Write protein to PDB file.

    Args:
        prot_pos: atom37 positions in protein, shape [..., N_res, 37, 3].
        file_path: path to the PDB file to be saved.
        aatype: optional AA types, shape [N_res,].
        overwrite: whether to overwrite existing files.
            If true, the saved files will start with suffix "_0".
            Otherwise, it will get the max index
            and start with max index + 1.
        no_indexing: if true, no index is added as suffix.
        b_factors: optional b-factors, shape [N_res, 37].
        residue_index: optional residue indices, shape [N_res,].
        chain_index: optional chain indices, shape [N_res,].

    Returns:
        Path to the saved PDB file.
    """
    if isinstance(file_path, str):
        file_path = pathlib.Path(file_path)

    atom_mask_eps = 1e-7

    if overwrite:
        max_existing_idx = 0
    else:
        file_dir = os.path.dirname(file_path)
        file_name = os.path.basename(file_path).strip(".pdb")
        existing_files = [x for x in os.listdir(file_dir) if file_name in x]
        max_existing_idx = max(
            [
                int(re.findall(r"_(\d+).pdb", x)[0])
                for x in existing_files
                if re.findall(r"_(\d+).pdb", x)
            ]
            + [0]
        )
    if not no_indexing:
        curr_stem = file_path.stem
        save_path = file_path.with_name(f"{curr_stem}_{max_existing_idx+1}.pdb")
    else:
        save_path = file_path
    with open(save_path, "w", encoding="utf-8") as f:
        if prot_pos.ndim == 4:
            for t, pos37 in enumerate(prot_pos):
                atom37_mask = np.sum(np.abs(pos37), axis=-1) > atom_mask_eps
                prot = create_full_prot(
                    pos37,
                    atom37_mask,
                    aatype=aatype,
                    b_factors=b_factors,
                    residue_index=residue_index,
                    chain_index=chain_index,
                )
                pdb_prot = protein.to_pdb(prot, model=t + 1, add_end=False)
                f.write(pdb_prot)
        elif prot_pos.ndim == 3:
            atom37_mask = np.sum(np.abs(prot_pos), axis=-1) > atom_mask_eps
            prot = create_full_prot(
                prot_pos,
                atom37_mask,
                aatype=aatype,
                b_factors=b_factors,
                residue_index=residue_index,
                chain_index=chain_index,
            )
            pdb_prot = protein.to_pdb(prot, model=1, add_end=False)
            f.write(pdb_prot)
        else:
            raise ValueError(f"Invalid positions shape {prot_pos.shape}")
        f.write("END")
    return save_path


def rigids_to_se3_vec(frame: np.ndarray, scale_factor: float = 1.0) -> np.ndarray:
    """Convert rigid frame representation to the vector of another format.

    A single rigid frame representation is a vector of shape [7,]:
        quaternion (shape [4,]) and translation (shape [3,]).
    The quaternion will be converted to axis-angle representation (shape [3,]).

    Args:
        frame: rigid frame, shape [N_res, 7].
        scale_factor: scale factor to apply for translation.

    Returns:
        Vector with rotation vector and scaled translation,
            shape [N_res, 6].
    """
    trans = frame[:, 4:] * scale_factor
    rotvec = Rotation.from_quat(frame[:, :4]).as_rotvec()
    se3_vec = np.concatenate([rotvec, trans], axis=-1)
    return se3_vec
