"""Utilities for directory traversing."""

import pathlib
from typing import Any


def sample_file_parser(sample_dir_path: pathlib.Path) -> list[pathlib.Path]:
    """Traverse the directory and read in prediction samples.

    Note: the expected file structure is:
        sample_dir_path/
          sample_0/
            sample_0_1.pdb
          sample_1/
            sample_1_1.pdb
          ...

    Args:
        sample_dir_path: root directory of the predicted samples.

    Returns:
        sample_paths: list of prediction file paths.
    """
    sample_paths = []
    for sample in sample_dir_path.glob("sample_*"):
        # extract sample index from folder name, e.g. sample_0 -> 0
        sample_idx = int(sample.stem.split("_")[-1])
        sample_all_atom_pdb_path = sample / f"sample_{sample_idx}_1_all_atom.pdb"
        if sample_all_atom_pdb_path.exists():
            sample_pdb_path = sample_all_atom_pdb_path
        else:
            sample_pdb_path = sample / f"sample_{sample_idx}_1.pdb"

        sample_paths.append(sample_pdb_path)

    return sample_paths


def traverse_prediction_dir(
    inference_path: pathlib.Path,
    legacy_file_structure: bool = False,
) -> dict[str, list[Any]]:
    """Traverse the prediction directory and extract information for evaluation.

    Args:
       inference_path: path to the inference results.
       legacy_file_structure: assume predicted were saved in legacy file format.

    Returns:
        Dictionary with key-value paris:
            - pdb_ids: list of pdb ids.
            - gt_pdb_path: list of paths to ground truth pdb files.
            - diffusion_info_path: path to diffusion info csv file.
            - predicted_pdb_paths: list of paths to predicted pdb files.
    """
    gt_pdb_paths = []
    diffusion_info_paths = []
    sample_paths = []
    pdb_ids = []
    for sample_folder in inference_path.glob("*_length_*"):
        # Get pdb id from the file name, e.g. 1ao7_length_22 -> 1ao7.
        folder_name = sample_folder.stem
        pdb_id = folder_name.split("_")[0]

        if legacy_file_structure:
            # Read in ground-truth pdb path.
            gt_pdb_path = sample_folder / f"sample_0/{pdb_id}_1.pdb"
            # Read in diffusion info.
            diffusion_info_path = sample_folder / "sample_0/diffusion_info.csv"
        else:
            # Read in ground-truth pdb path.
            gt_pdb_path = sample_folder / f"{pdb_id}_1.pdb"
            # Read in diffusion info.
            diffusion_info_path = sample_folder / "diffusion_info.csv"

        # Read in list of samples.
        sample_path = sample_file_parser(sample_dir_path=sample_folder)

        # Save paths.
        gt_pdb_paths.append(gt_pdb_path)
        diffusion_info_paths.append(diffusion_info_path)
        sample_paths.append(sample_path)
        pdb_ids.append(pdb_id)

    prediction_info_dict: dict[str, list[Any]] = {
        "pdb_ids": pdb_ids,
        "gt_pdb_paths": gt_pdb_paths,
        "diffusion_info_paths": diffusion_info_paths,
        "predicted_pdb_paths": sample_paths,
    }

    return prediction_info_dict
