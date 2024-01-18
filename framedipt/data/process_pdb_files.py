"""Script for preprocessing PDB files.

WARNING: NOT TESTED WITH SE(3) DIFFUSION.
This is example code of how to preprocess PDB files.
It does not process extra features that are used in process_pdb_dataset.py.
One can use the logic here to create a version of process_pdb_dataset.py
that works on PDB files.

"""
from __future__ import annotations

import argparse
import dataclasses
import functools as fn
import multiprocessing as mp
import os
import pathlib
import time
from typing import Any

import pandas as pd
from Bio import PDB

from framedipt.data import parsers
from framedipt.data import utils as data_utils
from framedipt.tools import errors
from framedipt.tools.log import get_logger

logger = get_logger()

# Define the parser
parser = argparse.ArgumentParser(description="PDB processing script.")
parser.add_argument(
    "--pdb_dir", help="Path to directory with PDB files.", type=pathlib.Path
)
parser.add_argument(
    "--num_processes", help="Number of processes.", type=int, default=50
)
parser.add_argument(
    "--write_dir",
    help="Path to write results to.",
    type=pathlib.Path,
    default=pathlib.Path("./preprocessed_pdbs"),
)
parser.add_argument("--debug", help="Turn on for debugging.", action="store_true")
parser.add_argument("--verbose", help="Whether to log everything.", action="store_true")


def process_file(file_path: pathlib.Path, write_dir: pathlib.Path) -> dict[str, Any]:
    """Processes protein file into usable, smaller pickles.

    Args:
        file_path: Path to file to read.
        write_dir: Directory to write pickles to.

    Returns:
        Saves processed protein to pickle and returns metadata.

    Raises:
        DataError if a known filtering rule is hit.
        All other errors are unexpected and are propagated.
    """
    metadata: dict[str, Any] = {}
    pdb_name = file_path.stem
    metadata["pdb_name"] = pdb_name

    processed_path = write_dir / f"{pdb_name}.pkl"
    metadata["processed_path"] = str(processed_path.resolve())
    metadata["raw_path"] = str(file_path)
    pdb_parser = PDB.PDBParser(QUIET=True)
    structure = pdb_parser.get_structure(pdb_name, file_path)

    # Extract all chains
    struct_chains = {chain.id.upper(): chain for chain in structure.get_chains()}
    metadata["num_chains"] = len(struct_chains)

    # Extract features
    struct_feats = []
    all_seqs = set()
    for chain_id, chain in struct_chains.items():
        # Convert chain id into int
        chain_id_int = data_utils.chain_str_to_int(chain_id)
        chain_prot = parsers.process_chain(chain, chain_id_int)
        chain_dict = dataclasses.asdict(chain_prot)
        chain_dict = data_utils.parse_chain_feats(chain_dict)
        all_seqs.add(tuple(chain_dict["aatype"]))
        struct_feats.append(chain_dict)
    if len(all_seqs) == 1:
        metadata["quaternary_category"] = "homomer"
    else:
        metadata["quaternary_category"] = "heteromer"
    complex_feats = data_utils.concat_np_features(struct_feats, False)

    # Process geometry features
    complex_aatype = complex_feats["aatype"]
    metadata["seq_len"] = len(complex_aatype)

    # Write features to pickles.
    data_utils.write_pkl(processed_path, complex_feats)

    # Return metadata
    return metadata


def process_serially(
    all_paths: list[pathlib.Path], write_dir: pathlib.Path
) -> list[dict[str, Any]]:
    all_metadata = []
    for file_path in all_paths:
        try:
            start_time = time.time()
            metadata = process_file(file_path, write_dir)
            elapsed_time = time.time() - start_time
            logger.info(f"Finished {file_path} in {elapsed_time:2.2f}s")
            all_metadata.append(metadata)
        except errors.DataError as e:
            logger.info(f"Failed {file_path}: {e}")
    return all_metadata


def process_fn(
    file_path: pathlib.Path, write_dir: pathlib.Path, verbose: bool = False
) -> dict[str, Any] | None:
    try:
        start_time = time.time()
        metadata = process_file(file_path, write_dir)
        elapsed_time = time.time() - start_time
        if verbose:
            logger.info(f"Finished {file_path} in {elapsed_time:2.2f}s")
        return metadata
    except errors.DataError as e:
        if verbose:
            logger.info(f"Failed {file_path}: {e}")
        return None


def main(args: argparse.Namespace) -> None:
    pdb_dir = args.pdb_dir
    all_file_paths = list(pdb_dir.glob("*.pdb"))
    total_num_paths = len(all_file_paths)
    write_dir = args.write_dir
    write_dir.mkdir(parents=True, exist_ok=True)
    metadata_file_name = "metadata_debug.csv" if args.debug else "metadata.csv"
    metadata_path = write_dir / metadata_file_name
    logger.info(f"Files will be written to {write_dir}")

    # Process each mmcif file
    if args.num_processes == 1 or args.debug:
        all_metadata = process_serially(all_file_paths, write_dir)
    else:
        partial_process_fn = fn.partial(
            process_fn, verbose=args.verbose, write_dir=write_dir
        )
        with mp.Pool(processes=args.num_processes) as pool:
            all_metadata = pool.map(
                partial_process_fn, all_file_paths  # type: ignore[arg-type]
            )
        all_metadata = [x for x in all_metadata if x is not None]
    metadata_df = pd.DataFrame(all_metadata)
    metadata_df.to_csv(metadata_path, index=False)
    succeeded = len(all_metadata)
    logger.info(f"Finished processing {succeeded}/{total_num_paths} files")


if __name__ == "__main__":
    # Don't use GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    main(parser.parse_args())
