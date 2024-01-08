"""Script for preprocessing mmcif files for faster consumption.

- Parses all mmcif protein files in a directory.
- Filters out low resolution files.
- Performs any additional processing.
- Writes all processed examples out to specified path.
"""
from __future__ import annotations

import dataclasses
import functools as fn
import multiprocessing as mp
import os
import pathlib
import time
from typing import Any

import Bio
import hydra
import mdtraj as md
import numpy as np
import omegaconf
import pandas as pd
from Bio.PDB import Chain, Model, Structure
from tqdm import tqdm

from framedipt.data import mmcif_parsing, parsers
from framedipt.data import utils as data_utils
from framedipt.tools import errors
from framedipt.tools.log import get_logger

logger = get_logger()


def _retrieve_mmcif_files(
    mmcif_dir: pathlib.Path,
    max_file_size: int | None,
    min_file_size: int | None,
    debug: bool,
    pdb_codes: list[str] | None = None,
    debug_num_files: int = 1000,
) -> list[pathlib.Path]:
    """Set up all the mmcif files to read.

    Args:
        mmcif_dir: directory to read mmcif files.
        max_file_size: optional maximum file size.
        min_file_size: optional minimum file size.
        debug: whether to run on debug mode which only proceeds 1000 files.
        pdb_codes: optional list of PDB codes.
            Only those files in `pdb_codes` will be included.

    Returns:
          List of mmcif file paths.
    """
    logger.info("Gathering mmCIF paths")
    pdb_codes_set = set(pdb_codes) if pdb_codes is not None else None
    total_num_files = 0
    all_mmcif_paths = []
    for mmcif_file_dir in tqdm(mmcif_dir.glob("*/")):
        for mmcif_path in mmcif_file_dir.glob("*.cif"):
            pdb_name = mmcif_path.stem[:4]
            if pdb_codes_set is not None and pdb_name not in pdb_codes_set:
                logger.info(f"File {pdb_name} not in PDB codes.")
                continue
            total_num_files += 1
            mmcif_file_size = mmcif_path.stat().st_size
            if min_file_size is not None and mmcif_file_size < min_file_size:
                logger.info(f"File {mmcif_path.stem} smaller than {min_file_size}.")
                continue
            if max_file_size is not None and mmcif_file_size > max_file_size:
                logger.info(f"File {mmcif_path.stem} bigger than {max_file_size}.")
                continue
            all_mmcif_paths.append(mmcif_path)
            if debug and total_num_files >= debug_num_files:
                # Don't process all files for debugging
                break
    logger.info(f"Processing {len(all_mmcif_paths)} files out of {total_num_files}.")
    return all_mmcif_paths


def extract_features_from_mmcif(
    mmcif_path: pathlib.Path,
    chains: list[str] | None = None,
    chain_max_len: int | None = None,
    chain_min_len: int | None = None,
    max_num_chains: int | None = None,
) -> tuple[
    mmcif_parsing.MmcifObject,
    int,
    list[int],
    list[int],
    dict[str, np.ndarray],
    dict[str, Chain.Chain],
]:
    """Extract structural features from mmCif file.

    Args:
        mmcif_path: path to the mmCif file.
        chains: optional input list of chains to be processed.
            If None, all chains in the mmcif file will be processed.
        chain_max_len: maximum length of the chain.
            Chains longer than this will be filtered.
        chain_min_len: minimum length of the chain.
            Chains shorter than this will be filtered.
        max_num_chains: maximum number of chains.
            Structures with more chains than this will be filtered.

    Returns:
        parsed_mmcif_object: MmcifObject containing parsed metadata.
        num_chains: number of chains in the structure.
        all_chain_lens: list of raw chain lengths.
        all_modeled_chain_lens: list of modeled chain lengths.
        complex_feats: extracted features in the structure.
            - atom_positions: atom positions, shape [N_res, 37, 3].
            - atom_mask: atom mask, shape [N_res, 37].
            - aatype: AA types, shape [N_res].
            - residue_index: residue indices, shape [N_res].
            - chain_index: chain indices, shape [N_res].
            - b_factors: b-factors, shape [N_res, 37].
            - bb_mask: carbon-alpha mask, shape [N_res].
            - bb_positions: carbon-alpha positions, shape [N_res, 3].
        struct_chains: dictionary of included chains {chain id: chain}.

    Raises:
        ValueError if any chain id in the input list of chains
            is not in mmcif file.
    """
    with open(mmcif_path, encoding="utf-8") as f:
        parsed_mmcif = mmcif_parsing.parse(
            file_id=mmcif_path.stem, mmcif_string=f.read()
        )
    if parsed_mmcif.errors:
        raise errors.MmcifParsingError(f"Encountered errors {parsed_mmcif.errors}")
    parsed_mmcif_object = parsed_mmcif.mmcif_object
    if parsed_mmcif_object is None:
        raise ValueError("Parsed mmcif object is None.")

    # Extract all chains
    struct_chains = {
        chain.id.upper(): chain for chain in parsed_mmcif_object.structure.get_chains()
    }
    num_chains = len(struct_chains)

    if chains is None:
        chains = list(struct_chains.keys())

    for chain_id in chains:
        if not struct_chains.get(chain_id):
            raise ValueError(
                "The input list of chains should be in mmcif file, "
                f"got {chain_id} not in {list(struct_chains.keys())}."
            )

    # Extract features
    struct_feats = []
    all_chain_lens = []
    all_modeled_chain_lens = []
    all_min_modeled_idxs = []
    all_max_modeled_idxs = []
    sorted_chain_id_int = 0
    for chain_id in chains:
        chain = struct_chains[chain_id]
        # Convert chain id into int
        chain_id_int = data_utils.chain_str_to_int(
            data_utils.map_to_new_str_name(sorted_chain_id_int)
        )
        chain_prot = parsers.process_chain(chain, chain_id_int)
        chain_dict = dataclasses.asdict(chain_prot)

        try:
            (
                chain_len,
                modeled_chain_len,
                min_modeled_idx,
                max_modeled_idx,
            ) = get_modeled_chain_len(
                aatype=chain_dict["aatype"],
                chain_max_len=chain_max_len,
                chain_min_len=chain_min_len,
            )
        except errors.LengthError as e:
            logger.info(
                f"LengthError for chain {chain_id}: {e}"
                f"It will be filtered from the structure {mmcif_path.stem}."
            )
            del struct_chains[chain_id]
            continue

        struct_feats.append(chain_dict)
        all_chain_lens.append(chain_len)
        all_modeled_chain_lens.append(modeled_chain_len)
        all_min_modeled_idxs.append(min_modeled_idx)
        all_max_modeled_idxs.append(max_modeled_idx)
        sorted_chain_id_int += 1

        # We check in the loop to avoid unnecessary computations.
        if max_num_chains is not None and sorted_chain_id_int > max_num_chains:
            raise errors.NumberOfChainsError(
                f"Too many modeled chains (more than {max_num_chains}), "
                f"overall {num_chains} chains."
            )

    if len(struct_feats) == 0:
        raise errors.NoModeledChainError("No chain is modeled.")

    complex_feats = data_utils.concat_np_features(struct_feats, False)
    complex_feats = data_utils.parse_chain_feats(complex_feats)
    complex_feats["min_modeled_idxs"] = np.array(all_min_modeled_idxs)
    complex_feats["max_modeled_idxs"] = np.array(all_max_modeled_idxs)

    return (
        parsed_mmcif_object,
        num_chains,
        all_chain_lens,
        all_modeled_chain_lens,
        complex_feats,
        struct_chains,
    )


def get_modeled_chain_len(
    aatype: np.ndarray,
    chain_max_len: int | None = None,
    chain_min_len: int | None = None,
) -> tuple[int, int, int, int]:
    """Get modeled chain length by eliminating unknown AAs at two terminus.

    Args:
        aatype: AA types in the chain, shape [N_res,].
        chain_max_len: optional max length to filter the chain.
        chain_min_len: optional min length to filter the chain.

    Returns:
        chain_len: the raw chain length.
        modeled_chain_len: modeled chain length
            by eliminating unknown AAs at two terminus.
        min_modeled_idx: the start index of the modeled chain in the raw chain.
        max_modeled_idx: the end index of the modeled chain in the raw chain.
    """
    # Get modeled indexes corresponding to 20 standard AAs.
    # And for unknown AAs (index 20) of two terminus, we do not model them.
    modeled_idxs = np.where(aatype != 20)[0]
    if np.sum(aatype != 20) == 0:
        raise errors.LengthError("No modeled residues.")
    min_modeled_idx: int = np.min(modeled_idxs).item()
    max_modeled_idx: int = np.max(modeled_idxs).item()
    chain_len = len(aatype)
    modeled_chain_len = max_modeled_idx - min_modeled_idx + 1
    if chain_max_len is not None and modeled_chain_len > chain_max_len:
        raise errors.LengthError(f"Too long {modeled_chain_len}.")
    if chain_min_len is not None and modeled_chain_len < chain_min_len:
        raise errors.LengthError(f"Too short {modeled_chain_len}.")

    return chain_len, modeled_chain_len, min_modeled_idx, max_modeled_idx


def filter_structure(
    structure: Structure.Structure,
    chain_ids: list[str] | dict[str, Any],
) -> Structure.Structure:
    """Filter structure by given chain ids.

    Only the chains with ID in `chain_ids` will be kept.

    Args:
        structure: the structure to be filtered.
        chain_ids: list or dictionary with keys of chain ids to be kept.

    Returns:
        The filtered structure.
    """
    structure_id = structure.id
    new_structure = Structure.Structure(id=f"{structure_id}_filtered")
    for i, model in enumerate(structure):
        new_model = Model.Model(id=f"{structure_id}_filtered_{i}")
        for chain in model:
            if chain.id.upper() in chain_ids:
                new_model.add(chain)
        new_structure.add(new_model)

    return new_structure


def rename_chains(
    structure: Structure.Structure,
    chain_name_mapping: dict[str, str],
) -> Structure.Structure:
    """Rename chains in the structure.

    Args:
        structure: the input structure.
        chain_name_mapping: dictionary of {old chain id: new chain id}.

    Returns:
        The structure with new chain ids.

    Raises:
        ChainNotFoundError if `chain_name_mapping` contains keys not in structure.
    """
    chains = {}
    for model in structure:
        for chain in model:
            chains[chain.id] = chain

    if not all(key in chains for key in chain_name_mapping):
        raise errors.ChainNotFoundError(
            f"{chain_name_mapping=} contains incorrect keys."
        )

    # First, set all ids to hash to avoid collisions between name flips
    old_names = {}
    for model in structure:
        for chain in model:
            if chain.id in chain_name_mapping:
                old_id, new_id = chain.id, hash(chain.id)
                chain.id = str(new_id)
                old_names[new_id] = old_id

    # Construct the new dictionary of renamed data
    new_names = {hash(old): new for old, new in chain_name_mapping.items()}

    for old, new in new_names.items():
        for model in structure:
            for chain in model:
                if chain.id == str(old):
                    chain.id = new

    return structure


def mdtraj_computation(
    mmcif_path: pathlib.Path | str,
    struct_chains: dict[str, Chain.Chain],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute secondary structure by MDtraj.

    Args:
        mmcif_path: path to the mmcif file.
        struct_chains: dictionary of chains in the structure.

    Returns:
        Tuple of secondary structure and radius of gyration.

    Raises:
        DataError exception if MDtraj computation fails.
    """
    if isinstance(mmcif_path, str):
        mmcif_path = pathlib.Path(mmcif_path)

    pdb_path = mmcif_path.with_suffix(".pdb")
    try:
        # Workaround for MDtraj not supporting mmcif in their latest release.
        # MDtraj source does support mmcif https://github.com/mdtraj/mdtraj/issues/652
        # We temporarily save the mmcif as a pdb and delete it after running mdtraj.
        p = Bio.PDB.MMCIFParser(QUIET=True)
        struc = p.get_structure("", mmcif_path)
        chain_ids = [
            chain.id
            for chain in struc.get_chains()
            if chain.id.upper() in struct_chains
        ]
        sorted_chain_ids = [
            data_utils.map_to_new_str_name(i) for i in range(len(chain_ids))
        ]

        # Filter the structure to keep valid chains.
        new_structure = filter_structure(struc, chain_ids)

        # Get chain name mapping to rename chains.
        # Because chain names like A-2, B-2 cannot be recognized by MDtraj.
        chain_name_mapping: dict[str, str] = {}
        for chain_id, new_chain_id in zip(chain_ids, sorted_chain_ids):
            if chain_id != new_chain_id:
                chain_name_mapping[chain_id] = new_chain_id

        # If chain name mapping is not empty, rename chains.
        if chain_name_mapping:
            logger.info(f"Renaming chains so that MDtraj can load {pdb_path}.")
            new_structure = rename_chains(new_structure, chain_name_mapping)

        # Save the filtered structure to pdb.
        io = Bio.PDB.PDBIO()
        io.set_structure(new_structure)
        io.save(str(pdb_path))

        # MDtraj
        traj = md.load(pdb_path)
        # SS calculation
        pdb_ss = md.compute_dssp(traj, simplified=True)
        # DG calculation
        pdb_dg = md.compute_rg(traj)
        pdb_path.unlink(missing_ok=True)
    except errors.DataError as e:
        pdb_path.unlink(missing_ok=True)
        raise errors.DataError(f"Mdtraj failed with error {e}") from e

    return pdb_ss, pdb_dg


def update_process_mmcif_metadata(
    metadata: dict[str, Any],
    parsed_mmcif_object: mmcif_parsing.MmcifObject,
    num_chains: int,
    max_resolution: float | None,
    check_valid_resolution: bool = True,
) -> dict[str, Any]:
    """Update metadata while processing mmcif files.

    Args:
        metadata: dictionary of existing metadata.
        parsed_mmcif_object: mmcif object.
        num_chains: number of chains in the structure.
        max_resolution: max resolution to filter the structure.
        check_valid_resolution: whether to check if resolution
            is valid (not 0.0).

    Returns:
        Updated dictionary of metadata.
        The following keys are added to the dictionary:
            - quaternary_category: "homomer" or "heteromer".
            - oligomeric_count
            - oligomeric_detail
            - resolution
            - structure_method

    Raises:
        ResolutionError exception if the resolution is larger than
            `max_resolution` or it's not valid.
    """
    if num_chains == 1:
        metadata["quaternary_category"] = "homomer"
    else:
        metadata["quaternary_category"] = "heteromer"

    raw_mmcif = parsed_mmcif_object.raw_string
    if "_pdbx_struct_assembly.oligomeric_count" in raw_mmcif:
        raw_olig_count = raw_mmcif["_pdbx_struct_assembly.oligomeric_count"]
        oligomeric_count = ",".join(raw_olig_count).lower()
    else:
        oligomeric_count = ""
    if "_pdbx_struct_assembly.oligomeric_details" in raw_mmcif:
        raw_olig_detail = raw_mmcif["_pdbx_struct_assembly.oligomeric_details"]
        oligomeric_detail = ",".join(raw_olig_detail).lower()
    else:
        oligomeric_detail = ""
    metadata["oligomeric_count"] = oligomeric_count
    metadata["oligomeric_detail"] = oligomeric_detail

    # Parse mmcif header and check resolution
    mmcif_header = parsed_mmcif_object.header
    mmcif_resolution = mmcif_header["resolution"]
    metadata["resolution"] = mmcif_resolution
    metadata["structure_method"] = mmcif_header["structure_method"]
    if max_resolution is not None and mmcif_resolution >= max_resolution:
        raise errors.ResolutionError(f"Too high resolution {mmcif_resolution}")
    invalid_resolution = 0.0
    if check_valid_resolution and mmcif_resolution == invalid_resolution:
        raise errors.ResolutionError(f"Invalid resolution {mmcif_resolution}")

    return metadata


def process_mmcif(
    mmcif_path: pathlib.Path,
    max_resolution: float | None,
    max_len: int | None,
    min_len: int | None,
    chain_max_len: int | None,
    chain_min_len: int | None,
    max_num_chains: int | None,
    write_dir: pathlib.Path,
    chains: list[str] | None = None,
    check_valid_resolution: bool = True,
) -> dict[str, Any]:
    """Processes MMCIF files into usable, smaller pickles.

    Args:
        mmcif_path: Path to mmcif file to read.
        max_resolution: Max resolution to allow.
        max_len: Max length to allow.
        min_len: Min length to allow.
        chain_max_len: optional max length to filter the chain.
        chain_min_len: optional min length to filter the chain.
        max_num_chains: optional max number of chains.
        write_dir: Directory to write pickles to.
        chains: optional input list of chains to be processed.
            If None, all chains in the mmcif file will be processed.
        check_valid_resolution: whether to check if resolution is valid.
            It's not valid if it's 0.

    Returns:
        Saves processed protein to pickle and returns metadata.

    Raises:
        DataError if a known filtering rule is hit.
        All other errors are unexpected and are propagated.
    """
    logger.info(f"Processing {mmcif_path}.")

    metadata: dict[str, Any] = {}

    # Get pdb name
    mmcif_name = mmcif_path.stem
    metadata["pdb_name"] = mmcif_name
    metadata["raw_path"] = str(mmcif_path)

    # Set up directory to save processed mmcif files.
    mmcif_subdir = write_dir / mmcif_name[1:3].lower()
    mmcif_subdir.mkdir(parents=True, exist_ok=True)

    # Set up processed mmcif path
    processed_mmcif_path = mmcif_subdir / f"{mmcif_name}.pkl"
    processed_mmcif_path = processed_mmcif_path.resolve()
    metadata["processed_path"] = str(processed_mmcif_path)

    # Parse mmcif file and get some of metadata,
    # e.g. oligomeric_count, oligomeric_details.
    (
        parsed_mmcif_object,
        num_chains,
        chain_lens,
        modeled_chain_lens,
        complex_feats,
        struct_chains,
    ) = extract_features_from_mmcif(
        mmcif_path=mmcif_path,
        chains=chains,
        chain_max_len=chain_max_len,
        chain_min_len=chain_min_len,
        max_num_chains=max_num_chains,
    )

    metadata["num_chains"] = num_chains
    metadata["seq_len"] = np.sum(chain_lens)
    modeled_seq_len = np.sum(modeled_chain_lens)
    if max_len is not None and modeled_seq_len > max_len:
        raise errors.LengthError(f"Too long {modeled_seq_len}.")
    if min_len is not None and modeled_seq_len < min_len:
        raise errors.LengthError(f"Too short {modeled_seq_len}.")
    metadata["modeled_seq_len"] = modeled_seq_len

    metadata = update_process_mmcif_metadata(
        metadata=metadata,
        parsed_mmcif_object=parsed_mmcif_object,
        num_chains=num_chains,
        max_resolution=max_resolution,
        check_valid_resolution=check_valid_resolution,
    )

    pdb_ss, pdb_dg = mdtraj_computation(mmcif_path, struct_chains)

    metadata["coil_percent"] = np.sum(pdb_ss == "C") / metadata["modeled_seq_len"]
    metadata["helix_percent"] = np.sum(pdb_ss == "H") / metadata["modeled_seq_len"]
    metadata["strand_percent"] = np.sum(pdb_ss == "E") / metadata["modeled_seq_len"]

    # Radius of gyration
    metadata["radius_gyration"] = pdb_dg[0]

    # Write features to pickles.
    data_utils.write_pkl(processed_mmcif_path, complex_feats)

    # Return metadata
    return metadata


def process_serially(
    all_mmcif_paths: list[pathlib.Path],
    max_resolution: float | None,
    max_len: int | None,
    min_len: int | None,
    chain_max_len: int | None,
    chain_min_len: int | None,
    max_num_chains: int | None,
    write_dir: pathlib.Path,
    all_chains_to_process: list[list[str]] | list[None] | None = None,
    check_valid_resolution: bool = True,
) -> list[dict[str, Any]]:
    """Process mmcif files serially.

    Args:
        all_mmcif_paths: list of mmcif file paths.
        max_resolution: max resolution to filter structures.
        max_len: max length to filter structures.
        min_len: min length to filter structures.
        chain_max_len: optional max length to filter the chain.
        chain_min_len: optional min length to filter the chain.
        max_num_chains: optional max number of chains.
        write_dir: path to write the processed results.
        all_chains_to_process: optional list of chain lists to process.
        check_valid_resolution: whether to check if resolution is valid.
            It's not valid if it's 0.

    Returns:
        List of processed metadata dictionary.

    Raises:
        ValueError if lengths of all_mmcif_paths and all_chains_to_process
            are not the same.
    """
    if all_chains_to_process is None:
        all_chains_to_process = [None] * len(all_mmcif_paths)
    if len(all_mmcif_paths) != len(all_chains_to_process):
        raise ValueError(
            f"Length of all_mmcif_paths and all_chains_to_process should be the same, "
            f"got {len(all_mmcif_paths)} != {len(all_chains_to_process)}."
        )

    all_metadata = []
    for i, mmcif_path in enumerate(all_mmcif_paths):
        try:
            start_time = time.time()
            metadata = process_mmcif(
                mmcif_path,
                max_resolution,
                max_len,
                min_len,
                chain_max_len,
                chain_min_len,
                max_num_chains,
                write_dir,
                chains=all_chains_to_process[i],
                check_valid_resolution=check_valid_resolution,
            )
            elapsed_time = time.time() - start_time
            logger.info(f"Finished {mmcif_path} in {elapsed_time:2.2f}s.")
            all_metadata.append(metadata)
        except errors.DataError as e:
            logger.info(f"Failed {mmcif_path}: {e}")
    return all_metadata


def process_fn(
    mmcif_path: pathlib.Path,
    write_dir: pathlib.Path,
    verbose: bool = False,
    max_resolution: float | None = None,
    max_len: int | None = None,
    min_len: int | None = None,
    chain_max_len: int | None = None,
    chain_min_len: int | None = None,
    max_num_chains: int | None = None,
    check_valid_resolution: bool = True,
) -> dict[str, Any] | None:
    try:
        start_time = time.time()
        metadata = process_mmcif(
            mmcif_path,
            max_resolution,
            max_len,
            min_len,
            chain_max_len,
            chain_min_len,
            max_num_chains,
            write_dir,
            check_valid_resolution=check_valid_resolution,
        )
        elapsed_time = time.time() - start_time
        if verbose:
            logger.info(f"Finished {mmcif_path} in {elapsed_time:2.2f}s.")
        return metadata
    except errors.DataError as e:
        if verbose:
            logger.info(f"Failed {mmcif_path}: {e}")
        return None


@hydra.main(
    version_base="1.3.1", config_path="../../config", config_name="data_process"
)
def main(cfg: omegaconf.DictConfig) -> None:
    mmcif_dir = pathlib.Path(cfg.mmcif_dir)
    # Download mmCIF files to mmcif_dir.
    if cfg.csv_file is not None:
        df_pdb_csv = pd.read_csv(cfg.csv_file)
        pdb_codes = list(df_pdb_csv["pdb_id"].unique())
        if cfg.debug:
            pdb_codes = pdb_codes[: cfg.debug_num_files]
        data_utils.download_cifs(
            pdb_codes,
            outdir=mmcif_dir,
            assembly=cfg.download_assembly,
            num_workers=cfg.num_workers_download,
        )
        logger.info(f"Download {len(pdb_codes)} files to {mmcif_dir}.")
    else:
        pdb_codes = None

    # Get all mmcif files to read.
    all_mmcif_paths = _retrieve_mmcif_files(
        mmcif_dir,
        cfg.max_file_size,
        cfg.min_file_size,
        cfg.debug,
        pdb_codes=pdb_codes,
        debug_num_files=cfg.debug_num_files,
    )
    total_num_paths = len(all_mmcif_paths)
    write_dir = pathlib.Path(cfg.write_dir)
    write_dir.mkdir(parents=True, exist_ok=True)
    metadata_file_name = "metadata_debug.csv" if cfg.debug else "metadata.csv"
    metadata_path = write_dir / metadata_file_name
    logger.info(f"Files will be written to {write_dir}.")

    # Process each mmcif file
    if cfg.num_processes == 1 or cfg.debug:
        all_metadata = process_serially(
            all_mmcif_paths,
            cfg.max_resolution,
            cfg.max_len,
            cfg.min_len,
            cfg.chain_max_len,
            cfg.chain_min_len,
            cfg.max_num_chains,
            write_dir,
            check_valid_resolution=cfg.check_valid_resolution,
        )
    else:
        partial_process_fn = fn.partial(
            process_fn,
            verbose=cfg.verbose,
            max_resolution=cfg.max_resolution,
            max_len=cfg.max_len,
            min_len=cfg.min_len,
            chain_max_len=cfg.chain_max_len,
            chain_min_len=cfg.chain_min_len,
            write_dir=write_dir,
            max_num_chains=cfg.max_num_chains,
            check_valid_resolution=cfg.check_valid_resolution,
        )
        # Uses max number of available cores.
        with mp.Pool() as pool:
            all_metadata = pool.map(
                partial_process_fn, all_mmcif_paths  # type: ignore[arg-type]
            )
        all_metadata = [x for x in all_metadata if x is not None]
    metadata_df = pd.DataFrame(all_metadata)
    metadata_df.to_csv(metadata_path, index=False)
    succeeded = len(all_metadata)
    logger.info(f"Finished processing {succeeded}/{total_num_paths} files.")


if __name__ == "__main__":
    # Don't use GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    main()  # pylint: disable=no-value-for-parameter
