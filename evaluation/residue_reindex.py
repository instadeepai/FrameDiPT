"""Rename residue indices to match those produced by FrameDiPT"""
import copy
import pathlib
import shutil

import hydra
import omegaconf
from Bio.PDB import PDBIO, PDBParser
from Bio.PDB.Model import Model
from Bio.PDB.Structure import Structure

from evaluation.utils.directory_parser import traverse_prediction_dir


def get_residue_map(
    model: Model,
) -> dict[str, list[tuple[tuple[str, int, str], tuple[str, int, str]]]]:
    """Calculate residue mapping such that the output residue indices are contiguous.

    Args:
        model: Biopython.PDB Model to calculate mapping for.

    Returns:
        dictionary, keys are chain IDs, values are a list containing tuples.
            returned tuples are a pair of the form (old_index, new_index).
            old_index and new_index take the form of the full residue indexing scheme
            used by BioPython - (hetero flag, sequence identifier, insertion code)
            new indices are of the form (" ", idx, " ") where the idx values are
            contiguous and start from zero, i.e. 0,1,2,3,4,5,... for each chain.

    """
    res_map: dict[str, list[tuple[tuple[str, int, str], tuple[str, int, str]]]] = {}
    for chain in model.get_chains():
        res_map[chain.id] = []
        for new_idx, residue in enumerate(chain.get_residues()):
            old_idx = residue.id
            new_full_idx = (" ", new_idx, " ")
            res_map[chain.id].append((old_idx, new_full_idx))
    return res_map


def apply_res_map(
    res_map: dict[str, list[tuple[tuple[str, int, str], tuple[str, int, str]]]],
    model: Model,
) -> Model:
    """Replace residues indices with new ones according to a map.
    Args:
        res_map: residue map, keys correspond to chain IDs. Each value is a list of
            tuples. Each tuple is a pair consisting of the old, existing residue index
            and the new one to replace it with.
        model: Model to apply residue indices to.

    Returns:
        New Model object with indies replaced."""
    # BioPython PDB Entities hold a lot of references to other Entities, so let's copy
    # everything to make sure we're not going to overwrite something we don't want to.
    new_model = copy.deepcopy(model)
    for chain in new_model.get_chains():
        res_pairs = res_map[chain.id]
        for old_idx, new_idx in res_pairs:
            # biopython runs a check to see we haven't clobbered the id of an existing
            # residue on every id change. To avoid clobbering, we use a temporary id
            # naming scheme, as ids of the form (" ", idx, " ") may often be found in
            # the existing idxs
            residue = chain[old_idx]
            residue.id = ("temp", new_idx[1], new_idx[2])
        # Now we've renamed every ID to a unique temporary value, rename them to the new
        # naming scheme
        for _, new_idx in res_pairs:
            residue = chain[("temp", new_idx[1], new_idx[2])]
            residue.id = new_idx

    return new_model


def strip_hetatms(model: Model) -> Model:
    """Remove water and other hetero atom entries from a model.
    Args:
        model: a Biopython.PDB Model
    Returns:
        A Model object containing only standard residues, no waters or other molecules.
    """

    # BioPython PDB Entities hold a lot of references to other Entities, so let's copy
    # everything to make sure we're not going to overwrite something we don't want to.
    new_model = copy.deepcopy(model)
    for chain in new_model.get_chains():
        res_ids = [res.id for res in chain.get_residues() if res.id[0] != " "]
        for res_id in res_ids:
            chain.detach_child(res_id)
    return new_model


def reindex_structure(structure: Structure) -> Structure:
    """Reindex residue IDs so that they start at 0 and have no gaps
    Args:
        structure: a BioPython.PDB Structure
    Returns:
        A new structure object each resiude re-indexed"""
    base_model = next(structure.get_models())
    clean_model = strip_hetatms(base_model)
    residue_map = get_residue_map(model=clean_model)
    out_structure = Structure(id=f"{structure}_new")
    for model in structure:
        clean_model = strip_hetatms(model)
        new_model = apply_res_map(residue_map, clean_model)
        out_structure.add(new_model)
    return out_structure


def convert_path_root(
    in_dir: pathlib.Path, out_dir: pathlib.Path, current_path: pathlib.Path
) -> pathlib.Path:
    """Replace the root of a path object with another.
    Args:
        in_dir: The root to replace
        out_dir: The root to replace it with
        current_path: The path to swap roots on.
    Returns:
        Path object with root directory replaced."""
    root_index_to_replace = list(current_path.parents).index(in_dir)
    parts_to_keep = current_path.parts[-(root_index_to_replace + 1) :]
    out_path = out_dir.joinpath(*parts_to_keep)
    return out_path


@hydra.main(version_base="1.3.1", config_path="../config", config_name="evaluation")
def run(cfg: omegaconf.DictConfig) -> None:
    in_dir = pathlib.Path(cfg.reindex.in_path)
    out_dir = pathlib.Path(cfg.reindex.out_path)
    legacy = cfg.reindex.legacy

    out_dir.mkdir(parents=True, exist_ok=True)
    file_info = traverse_prediction_dir(in_dir, legacy_file_structure=legacy)
    parser = PDBParser(QUIET=True)
    io = PDBIO()
    for i, pdb_id in enumerate(file_info["pdb_ids"]):
        gt_path = file_info["gt_pdb_paths"][i]
        info_path = file_info["diffusion_info_paths"][i]
        sample_paths = file_info["predicted_pdb_paths"][i]
        # Copy info csv, our resiude indexing in this file is what we want already.
        new_info_path = convert_path_root(in_dir, out_dir, info_path)
        new_info_path.parent.mkdir(exist_ok=True, parents=True)
        shutil.copyfile(info_path, new_info_path)

        # Convert groundtruth structure
        gt_structure = parser.get_structure(id=pdb_id, file=gt_path)
        new_gt_structure = reindex_structure(gt_structure)
        # Setup out directory and write
        new_gt_path = convert_path_root(in_dir, out_dir, gt_path)
        new_gt_path.parent.mkdir(exist_ok=True, parents=True)
        io.set_structure(new_gt_structure)
        io.save(str(new_gt_path))
        # Now convert samples using same residue map
        for j, sample_path in enumerate(sample_paths):
            sample_structure = parser.get_structure(
                id=f"{pdb_id}_{j}", file=sample_path
            )
            new_sample_structure = reindex_structure(sample_structure)
            # Make sure directory exists and write
            new_sample_path = convert_path_root(in_dir, out_dir, sample_path)
            new_sample_path.parent.mkdir(exist_ok=True, parents=True)
            io.set_structure(new_sample_structure)
            io.save(str(new_sample_path))


if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter
