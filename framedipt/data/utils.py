"""Module of data utils."""
from __future__ import annotations

import collections
import concurrent.futures
import dataclasses
import functools
import io
import os
import pathlib
import pickle
import string
from collections.abc import Iterable
from typing import Any, Callable

import numpy as np
import torch
import tree
from Bio import PDB
from Bio.PDB import MMCIFParser, Model, Structure
from omegaconf import DictConfig, OmegaConf
from torch.utils import data

from framedipt import RESIDUE_GAP
from framedipt.data import parsers
from framedipt.protein import chemical, residue_constants
from framedipt.tools import errors
from framedipt.tools.custom_type import TensorNDArray
from framedipt.tools.log import get_logger
from openfold.data import data_transforms
from openfold.utils import rigid_utils

logger = get_logger()

# Global map from chain characters to integers.
ALPHANUMERIC = string.ascii_letters + string.digits
CHAIN_TO_INT = {chain_char: i for i, chain_char in enumerate(ALPHANUMERIC)}
INT_TO_CHAIN = dict(enumerate(ALPHANUMERIC))

CHAIN_FEATS = ["atom_positions", "aatype", "atom_mask", "residue_index", "b_factors"]
UNPADDED_FEATS = ["t", "rot_score_scaling", "trans_score_scaling", "t_seq", "t_struct"]
RIGID_FEATS = ["rigids_0", "rigids_t"]
PAIR_FEATS = ["rel_rots"]


def move_to_np(x: torch.Tensor) -> np.ndarray:
    """Move a torch tensor to numpy array.

    Args:
        x: input tensor.

    Returns:
        Converted numpy array.
    """
    return x.cpu().detach().numpy()


def maybe_move_to_torch(x: TensorNDArray, device: torch.device = None) -> torch.Tensor:
    """Convert a numpy array to a torch tensor.

    Args:
        x: input ndarray.
        device: optionally specify the device, defaults to None.

    Returns:
        Converted torch tensor.
    """
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device)

    return x


def aatype_to_seq(aatype: Iterable) -> str:
    """Convert array of amino acids to the string representation of the sequence.

    Args:
        aatype: iterable of amino acid types.

    Returns:
        Converted sequence.
    """
    return "".join([residue_constants.restypes_with_x[x] for x in aatype])


class CPUUnpickler(pickle.Unpickler):
    """Pytorch pickle loading workaround.

    https://github.com/pytorch/pytorch/issues/16797
    """

    def find_class(self, module: str, name: str) -> Any:
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")

        return super().find_class(module, name)


def download_unit_cif(pdb_code: str, pdir: pathlib.Path | str) -> None:
    """Download first biological assembly of a list of PDBs to the specified directory.

     Remark:
     - Existing assemblies files will not be overwritten.

    Args:
        pdb_code: PDB code to download its first assembly.
        pdir: Path of the directory to store the downloaded assemblies files.
    """
    if isinstance(pdir, pathlib.Path):
        pdir = str(pdir)

    PDB.PDBList().retrieve_assembly_file(
        pdb_code,
        assembly_num=1,
        pdir=pdir,
        file_format="mmCif",
    )


def download_cifs(
    pdb_codes: list[str],
    outdir: pathlib.Path | str,
    assembly: bool = True,
    num_workers: int | None = None,
) -> None:
    """Download mmCif files of given PDB codes.

    Args:
        pdb_codes: list of PDB codes.
        outdir: directory to download mmCif files.
        assembly: whether to download assembly1.
        num_workers: number of workers to download cifs.
    """
    if isinstance(outdir, str):
        outdir = pathlib.Path(outdir)

    outdir = outdir / "cifs"

    if not assembly:
        pdbl = PDB.PDBList()
        pdbl.download_pdb_files(pdb_codes, pdir=str(outdir), file_format="mmCif")
    else:
        with concurrent.futures.ThreadPoolExecutor(num_workers) as executor:
            executor.map(
                functools.partial(download_unit_cif, pdir=str(outdir)),
                pdb_codes,
            )

    logger.info(f"mmCif files are downloaded to {outdir}.")


def write_pkl(
    save_path: pathlib.Path,
    pkl_data: Any,
    create_dir: bool = False,
    use_torch: bool = False,
) -> None:
    """Serialize data into a pickle file.

    Args:
        save_path: path to save the pickle file.
        pkl_data: data to be saved.
        create_dir: whether to create directory.
        use_torch: whether to use torch.
    """
    if create_dir:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if use_torch:
        torch.save(pkl_data, save_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(save_path, "wb") as handle:
            pickle.dump(pkl_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_pkl(
    read_path: pathlib.Path,
    verbose: bool = True,
    use_torch: bool = False,
    map_location: str | None = None,
) -> dict[str, Any]:
    """Read data from a pickle file."""
    try:
        if use_torch:
            return torch.load(read_path, map_location=map_location)

        with open(read_path, "rb") as handle:
            return pickle.load(handle)  # noqa: S301
    except errors.PickleLoadError as e:
        try:
            with open(read_path, "rb") as handle:
                return CPUUnpickler(handle).load()
        except errors.CPUUnpicklerError as e2:
            if verbose:
                logger.info(
                    f"Failed to read {read_path}. First error: {e}\n Second error: {e2}"
                )
            raise e from e2


def compare_conf(conf1: Any, conf2: Any) -> bool:
    return OmegaConf.to_yaml(conf1) == OmegaConf.to_yaml(conf2)


def parse_pdb(filename: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    with open(filename, encoding="utf-8") as f:
        lines = f.readlines()
    return parse_pdb_lines(lines)


def parse_pdb_lines(lines: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    # indices of residues observed in the structure
    idx_s = [
        int(line[22:26])
        for line in lines
        if line[:4] == "ATOM" and line[12:16].strip() == "CA"
    ]

    # 4 BB + up to 10 SC atoms
    xyz = np.full((len(idx_s), 14, 3), np.nan, dtype=np.float32)
    seq = []
    for line in lines:
        if line[:4] != "ATOM":
            continue
        res_no, atom, aa = int(line[22:26]), line[12:16], line[17:20]
        seq.append(residue_constants.restype_3to1[aa])
        idx = idx_s.index(res_no)
        for i_atm, tgtatm in enumerate(chemical.aa2long[chemical.aa2num[aa]]):
            if tgtatm == atom:
                xyz[idx, i_atm, :] = [
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                ]
                break

    # save atom mask
    mask = np.logical_not(np.isnan(xyz[..., 0]))
    xyz[np.isnan(xyz[..., 0])] = 0.0

    return xyz, mask, np.array(idx_s), "".join(seq)


def chain_str_to_int(chain_str: str) -> int:
    chain_int = 0
    if len(chain_str) == 1:
        return CHAIN_TO_INT[chain_str]
    for i, chain_char in enumerate(chain_str):
        chain_int += CHAIN_TO_INT[chain_char] + (i * len(ALPHANUMERIC))
    return chain_int


def map_to_new_str_name(index: int) -> str:
    """Map 0-based index to alphabet-based chain name.

    Examples:
        - 0 --> "A";
        - 25 --> "Z";
        - 26 --> "AA";
        - 676 --> "ZA".
    Args:
        index: integer of index.

    Returns:
        Mapped chain name string.
    """
    num_letters = 26
    if index < num_letters:
        return chr(ord("A") + index)
    reminder = index % num_letters
    name = chr(ord("A") + reminder)
    multiple = index // num_letters - 1  # need to do -1 because it's 0-based.
    return map_to_new_str_name(multiple) + name


def parse_pdb_feats(
    pdb_name: str,
    pdb_path: pathlib.Path | str,
    scale_factor: float = 1.0,
    chain_id: Iterable[str] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Parse chain features in PDB file.

    Args:
        pdb_name: name of PDB to parse.
        pdb_path: path to PDB file to read.
        scale_factor: factor to scale atom positions.
        chain_id: list of chains to parse.
            Default None, parse all chains.

    Returns:
        Dict with CHAIN_FEATS features extracted from PDB with specified
        preprocessing.
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_name, pdb_path)
    struct_chains = {chain.id: chain for chain in structure.get_chains()}

    def _process_chain_id(x: str) -> dict[str, np.ndarray]:
        chain_prot = parsers.process_chain(struct_chains[x], x)
        chain_dict = dataclasses.asdict(chain_prot)

        # Process features
        feat_dict = {x: chain_dict[x] for x in CHAIN_FEATS}
        return parse_chain_feats(feat_dict, scale_factor=scale_factor)

    if chain_id is None:
        return {x: _process_chain_id(x) for x in struct_chains}
    return {x: _process_chain_id(x) for x in chain_id}


def pad_feats(
    raw_feats: dict[str, torch.Tensor | np.ndarray],
    max_len: int,
    use_torch: bool = False,
) -> dict[str, torch.Tensor | np.ndarray]:
    padded_feats = {
        feat_name: pad(feat, max_len, use_torch=use_torch)
        for feat_name, feat in raw_feats.items()
        if feat_name not in UNPADDED_FEATS + RIGID_FEATS
    }
    for feat_name in PAIR_FEATS:
        if feat_name in padded_feats:
            padded_feats[feat_name] = pad(padded_feats[feat_name], max_len, pad_idx=1)
    for feat_name in UNPADDED_FEATS:
        if feat_name in raw_feats:
            padded_feats[feat_name] = raw_feats[feat_name]
    for feat_name in RIGID_FEATS:
        if feat_name in raw_feats:
            padded_feats[feat_name] = pad_rigid(raw_feats[feat_name], max_len)
    return padded_feats


def pad_rigid(rigid: torch.Tensor, max_len: int) -> torch.Tensor:
    num_rigids = rigid.shape[0]
    pad_amt = max_len - num_rigids
    padded_rigid = rigid_utils.Rigid.identity(
        (pad_amt,), dtype=rigid.dtype, device=rigid.device, requires_grad=False
    )
    return torch.cat([rigid, padded_rigid.to_tensor_7()], dim=0)


def pad(
    x: torch.Tensor | np.ndarray,
    max_len: int,
    pad_idx: int = 0,
    use_torch: bool = False,
    reverse: bool = False,
) -> torch.Tensor | np.ndarray:
    """Right pads dimension of numpy array.

    Args:
        x: numpy like array to pad.
        max_len: desired length after padding
        pad_idx: dimension to pad.
        use_torch: use torch padding method instead of numpy.
        reverse: whether to reverse padding.
            Default padding is on the right side,
            if reverse, it's on the left side.

    Returns:
        x with its pad_idx dimension padded to max_len
    """
    # Pad only the residue dimension.
    seq_len = x.shape[pad_idx]
    pad_amt = max_len - seq_len
    pad_widths = [(0, 0)] * x.ndim
    if pad_amt < 0:
        raise ValueError(f"Invalid pad amount {pad_amt}")
    if reverse:
        pad_widths[pad_idx] = (pad_amt, 0)
    else:
        pad_widths[pad_idx] = (0, pad_amt)
    if use_torch:
        # Padding in torch takes flatten tuples,
        # and starts by the last dimension.
        flatten_pad_widths = sum(pad_widths[::-1], ())
        return torch.nn.functional.pad(x, flatten_pad_widths)
    return np.pad(x, pad_widths)


def write_checkpoint(
    ckpt_path: pathlib.Path | str,
    model: dict[str, Any],
    cfg: DictConfig,
    optimizer: dict[str, Any],
    epoch: int,
    step: int,
    use_torch: bool = True,
) -> None:
    """Serialize experiment state and stats to a pickle file.

    Args:
        ckpt_path: Path to save checkpoint.
        step: Experiment step at time of checkpoint.
        exp_state: Experiment state to be written to pickle.
        preds: Model predictions to be written as part of checkpoint.
    """
    if isinstance(ckpt_path, str):
        ckpt_path = pathlib.Path(ckpt_path)

    ckpt_dir = os.path.dirname(ckpt_path)
    for fname in os.listdir(ckpt_dir):
        if ".pkl" in fname or ".pth" in fname:
            os.remove(os.path.join(ckpt_dir, fname))

    logger.info(f"Serializing experiment state to {ckpt_path}.")
    write_pkl(
        ckpt_path,
        {
            "model": model,
            "conf": cfg,
            "optim": optimizer,
            "epoch": epoch,
            "step": step,
        },
        use_torch=use_torch,
    )


def concat_np_features(
    np_dicts: list[dict[str, np.ndarray]], add_batch_dim: bool
) -> dict[str, np.ndarray]:
    """Performs a nested concatenation of feature dicts.

    Args:
        np_dicts: list of dicts with the same structure.
            Each dict must have the same keys and numpy arrays as the values.
        add_batch_dim: whether to add a batch dimension to each feature.

    Returns:
        A single dict with all the features concatenated.
    """
    combined_dict: dict[str, list] = collections.defaultdict(list)
    for chain_dict in np_dicts:
        for feat_name, feat_val in chain_dict.items():
            if feat_val is None:
                continue
            feat_val_batch = feat_val[None] if add_batch_dim else feat_val
            combined_dict[feat_name].append(feat_val_batch)
    # Concatenate each feature
    combined_dict_with_array: dict[str, np.ndarray] = {}
    for feat_name, feat_vals in combined_dict.items():
        combined_dict_with_array[feat_name] = np.concatenate(feat_vals, axis=0)
    return combined_dict_with_array


def length_batching(
    np_dicts: list[dict[str, np.ndarray | torch.Tensor]],
    max_squared_res: int,
) -> list[dict[str, torch.Tensor]]:
    # Convert the input np_dicts which is a list of dictionary [x1, x2, ...]
    # to a list of tuple [(len1, x1), (len2, x2), ...]
    # where len1 is the length of x1["res_mask"]
    dicts_by_length = [(x["res_mask"].shape[0], x) for x in np_dicts]
    length_sorted = sorted(dicts_by_length, key=lambda x: x[0], reverse=True)
    max_len = length_sorted[0][0]
    max_batch_examples = int(max_squared_res // max_len**2)
    if max_batch_examples < 1:
        raise ValueError(
            f"No sample in the batch with max length {max_len}."
            f"Please increase max_squared_res or reduce max_len."
        )
    padded_batch = [
        pad_feats(x, max_len) for (_, x) in length_sorted[:max_batch_examples]
    ]
    return torch.utils.data.default_collate(padded_batch)


def create_data_loader(
    torch_dataset: data.Dataset,
    batch_size: int,
    shuffle: bool,
    sampler: torch.utils.data.Sampler = None,
    num_workers: int = 0,
    np_collate: bool = False,
    max_squared_res: int = 1_000_000,
    length_batch: bool = False,
    drop_last: bool = False,
    prefetch_factor: int = 2,
) -> torch.utils.data.DataLoader:
    """Creates a data loader with jax compatible data structures."""
    collate_fn: Callable[
        [list[dict[str, np.ndarray]]], dict[str, np.ndarray]
    ] | Callable[
        [list[dict[str, np.ndarray | torch.Tensor]]], list[dict[str, torch.Tensor]]
    ] | None = None
    if np_collate:
        collate_fn = functools.partial(concat_np_features, add_batch_dim=True)
    elif length_batch:
        collate_fn = functools.partial(
            length_batching,
            max_squared_res=max_squared_res,
        )
    else:
        collate_fn = None
    persistent_workers = num_workers > 0
    prefetch_factor = 2 if num_workers == 0 else prefetch_factor
    return data.DataLoader(
        torch_dataset,
        sampler=sampler,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        drop_last=drop_last,
        # Need fork https://github.com/facebookresearch/hydra/issues/964
        multiprocessing_context="fork" if num_workers != 0 else None,
    )


def parse_chain_feats(
    chain_feats: dict[str, np.ndarray], scale_factor: float = 1.0
) -> dict[str, np.ndarray]:
    ca_idx = residue_constants.atom_order["CA"]
    chain_feats["bb_mask"] = chain_feats["atom_mask"][:, ca_idx]
    bb_pos = chain_feats["atom_positions"][:, ca_idx]
    bb_center = np.sum(bb_pos, axis=0) / (np.sum(chain_feats["bb_mask"]) + 1e-5)
    centered_pos = chain_feats["atom_positions"] - bb_center[None, None, :]
    scaled_pos = centered_pos / scale_factor
    chain_feats["atom_positions"] = scaled_pos * chain_feats["atom_mask"][..., None]
    chain_feats["bb_positions"] = chain_feats["atom_positions"][:, ca_idx]
    return chain_feats


def pad_pdb_feats(
    raw_feats: dict[str, np.ndarray], max_len: int
) -> dict[str, np.ndarray]:
    padded_feats = {
        feat_name: pad(feat, max_len)
        for feat_name, feat in raw_feats.items()
        if feat_name not in UNPADDED_FEATS
    }
    for feat_name in UNPADDED_FEATS:
        if feat_name in raw_feats:
            padded_feats[feat_name] = raw_feats[feat_name]
    return padded_feats


def calc_distogram(
    pos: torch.Tensor, min_bin: float, max_bin: float, num_bins: int
) -> torch.Tensor:
    dists_2d = torch.linalg.norm(pos[:, :, None, :] - pos[:, None, :, :], axis=-1)[
        ..., None
    ]
    lower = torch.linspace(min_bin, max_bin, num_bins, device=pos.device)
    upper = torch.cat([lower[1:], lower.new_tensor([1e8])], dim=-1)
    dgram = ((dists_2d > lower) * (dists_2d < upper)).type(pos.dtype)
    return dgram


def save_fasta(
    pred_seqs: list[str],
    seq_names: list[str],
    file_path: str,
) -> None:
    if os.path.exists(file_path):
        os.remove(file_path)
    with open(file_path, "w", encoding="utf-8") as f:
        for x, y in zip(seq_names, pred_seqs):
            f.write(f">{x}\n{y}\n")


def preprocess_aatype(
    aatype: torch.Tensor | None,
    fixed_mask: torch.Tensor,
    inpainting: bool,
    input_aatype: bool,
) -> torch.Tensor | None:
    """Preprocess AA type input according to different arguments.

    Args:
        aatype: optional input raw aatype, shape [..., N_res]
            with values in 0-20 inclusive.
        fixed_mask: fixed mask for non-diffused region, shape [..., N_res].
        inpainting: whether to do inpainting.
        input_aatype: whether to input AA type.

    Returns:
        None or pre-processed AA type.
            If aatype is None
                or not in inpainting mode and not input_aatype,
            return None;
            Else if not input AA type,
            replace with unknown AA type in the diffused region;
            Else return aatype.

    Raises:
        ValueError if inpainting is True and aatype is None.
    """
    if aatype is None or (not inpainting and not input_aatype):
        return None

    if inpainting and aatype is None:
        raise ValueError("When inpainting is True, aatype should be given, got None.")

    aatype = aatype.type(torch.int64)
    if not input_aatype:
        aatype = torch.where(
            fixed_mask.bool(),
            aatype,
            torch.full(
                aatype.shape,
                fill_value=20,
                dtype=torch.int64,
                device=aatype.device,
            ),
        )  # unknown
    return aatype


def create_single_redacted_region(
    res_mask: np.ndarray,
    rng: np.random.Generator,
    redact_min_len: int | None,
    redact_max_len: int | None,
) -> np.ndarray:
    """Create single redacted region in a single chain for inpainting.

    If redact_min_len and redact_max_len are both given,
        return the diffused mask of a random continuous region.
    else return the diffused mask of values 1 meaning diffusing the whole chain.

    Args:
        res_mask: residue mask, shape [N_res,].
        rng: numpy random generator.
        redact_min_len: optional minimum length of redacted region.
        redact_max_len: optional maximum length of redacted region.

    Returns:
        diff_mask: diffused mask of shape [N_res,] for redacted region.
    """
    if redact_min_len is None or redact_max_len is None:
        return np.ones_like(res_mask)

    modeled_res_mask = np.where(res_mask != 0)[0]
    min_idx = modeled_res_mask[0]
    max_idx = modeled_res_mask[-1]
    modeled_len = max_idx - min_idx + 1
    diff_mask = np.zeros_like(res_mask)

    redact_max_len = min(redact_max_len, modeled_len)
    redact_min_len = min(redact_min_len, redact_max_len)
    length = rng.integers(low=redact_min_len, high=redact_max_len, endpoint=True)

    start_idx = rng.integers(
        low=min_idx,
        high=max_idx + 1 - length,
        endpoint=True,
    )
    diff_mask[start_idx : start_idx + length] = 1

    return diff_mask


def create_redacted_regions(
    chain_idx: np.ndarray,
    res_mask: np.ndarray,
    rng: np.random.Generator,
    redact_min_len: int,
    redact_max_len: int,
) -> np.ndarray:
    """Create a single redacted region in each chain for inpainting.

    Args:
        chain_idx: chain indices, shape [N_res].
        res_mask: residue mask, shape [N_res,].
        rng: numpy random generator.
        redact_min_len: minimum length of redacted region.
        redact_max_len: maximum length of redacted region.

    Returns:
        diff_mask: diffused mask of shape [N_res,] for redacted region.
            Randomly choose a continuous region for diffusion in each chain.
    """
    chain_ids = np.unique(chain_idx)
    diff_mask = []
    # diffuse a region in each chain
    for chain_id in chain_ids:
        chain_res_mask = res_mask[chain_idx == chain_id]
        chain_diff_mask = create_single_redacted_region(
            res_mask=chain_res_mask,
            rng=rng,
            redact_max_len=redact_max_len,
            redact_min_len=redact_min_len,
        )
        diff_mask.append(chain_diff_mask)
    return np.concatenate(diff_mask)


def process_modeled_chain_features(
    features: dict[str, np.ndarray],
    chain_id: int | None,
    min_idx: int,
    max_idx: int,
    rng: np.random.Generator | None = None,
    chain_max_len: int | None = None,
) -> dict[str, np.ndarray]:
    """Process chain features within the modeled region using min_idx and max_idx.

    Modeled region is the region eliminating unknown residues of two sides of the chain.

    Args:
        features: dictionary of raw chain features.
        chain_id: integer id of the chain to process.
        min_idx: minimum index of the modeled region of the chain.
        max_idx: maximum index of the modeled region of the chain.
        rng: optional numpy random generator to select the single chain to process.
        chain_max_len: optional maximum length of the chain.
            If it's set, the chain larger than `chain_max_len` will be randomly cut.

    Returns:
        Dictionary of processed chain features.
    """
    if chain_id is not None:
        mask = features["chain_index"] == chain_id
        chain_processed_feats = tree.map_structure(
            lambda x: x[mask],
            features,  # extract single chain
        )
    else:
        chain_processed_feats = features
    modeled_idxs = np.arange(min_idx, max_idx + 1, dtype=np.int64)
    modeled_len = max_idx + 1 - min_idx

    # If modeled length bigger than chain_max_len,
    # randomly pick a sub-chain of length chain_max_len.
    if chain_max_len is not None and modeled_len > chain_max_len:
        if rng is not None:
            start_idx = rng.integers(modeled_len - chain_max_len + 1)
        else:
            start_idx = np.random.randint(modeled_len - chain_max_len + 1)
        modeled_idxs = modeled_idxs[start_idx : start_idx + chain_max_len]
        logger.debug(f"Chain too long {modeled_len}, cut from index {start_idx}.")

    chain_processed_feats = tree.map_structure(
        lambda x: x[modeled_idxs],
        chain_processed_feats,  # eliminate unknown residues of two sides of the chain
    )
    return chain_processed_feats


@functools.lru_cache(maxsize=50000)
def process_csv_row(
    processed_file_path: pathlib.Path,
    process_monomer: bool,
    extract_single_chain: bool,
    rng: np.random.Generator | None = None,
    chain_max_len: int | None = None,
) -> dict[str, np.ndarray | torch.Tensor]:
    """Process input features for one csv row.

    Args:
        processed_file_path: path to the file to be processed.
        process_monomer: whether to process monomers.
        extract_single_chain: whether to randomly extract a single chain.
        rng: optional numpy random generator to select the single chain to process.
        chain_max_len: optional maximum length of the chain.
            If it's set, the chain larger than `chain_max_len` will be randomly cut.

    Returns:
         Dictionary of processed features.
            - aatype: amino acid types, shape [N_res, 21].
            - seq_idx: 0-based residue indices, shape [N_res].
            - chain_idx: chain indices, shape [N_res].
            - residx_atom14_to_atom37: indices to convert atom14 to atom 37,
                shape [N_res, 14].
            - residue_index: raw residue indices in PDB file,
                shape [N_res].
            - res_mask: residue mask, shape [N_res].
            - atom37_pos: atom37 coordinates, shape [N_res, 37, 3].
            - atom37_mask: atom37 mask, shape [N_res, 37].
            - atom14_pos: atom14 coordinates, shape [N_res, 14, 3].
            - rigidgroups_0: rigid group representation at t = 0,
                shape [N_res, 8, 4, 4].
            - torsion_angles_sin_cos: torsion angle in sin-cos format,
                shape [N_res, 7, 2].
    """
    processed_feats = read_pkl(processed_file_path)

    # Get unsorted unique chain indexes.
    indexes = np.unique(processed_feats["chain_index"], return_index=True)[1]
    unique_chain_indexes = [
        processed_feats["chain_index"][index] for index in sorted(indexes)
    ]

    # Monomer training uses old processed dataset,
    # which does not have the same key values
    # as the new processed dataset for multimers.
    if process_monomer:
        modeled_idxs = processed_feats["modeled_idx"]
        min_idx: int = np.min(modeled_idxs).item()
        max_idx: int = np.max(modeled_idxs).item()
        del processed_feats["modeled_idx"]
        del processed_feats["chains"]

        processed_feats = process_modeled_chain_features(
            features=processed_feats,
            chain_id=None,
            min_idx=min_idx,
            max_idx=max_idx,
            rng=rng,
            chain_max_len=None,
        )
    else:
        # Only take modeled residues.
        min_idxs: np.ndarray = processed_feats["min_modeled_idxs"]
        max_idxs: np.ndarray = processed_feats["max_modeled_idxs"]
        del processed_feats["min_modeled_idxs"]
        del processed_feats["max_modeled_idxs"]

        if extract_single_chain:
            num_chains = len(min_idxs)
            if rng is not None:
                selected_chain_idx = rng.integers(num_chains)
            else:
                selected_chain_idx = np.random.randint(num_chains)

            chain_id = unique_chain_indexes[selected_chain_idx]
            min_idx = min_idxs[selected_chain_idx]
            max_idx = max_idxs[selected_chain_idx]

            processed_feats = process_modeled_chain_features(
                features=processed_feats,
                chain_id=chain_id,
                min_idx=min_idx,
                max_idx=max_idx,
                rng=rng,
                chain_max_len=chain_max_len,
            )
        else:
            all_processed_feats = []
            for chain_id, min_idx, max_idx in zip(
                unique_chain_indexes, min_idxs, max_idxs
            ):
                chain_processed_feats = process_modeled_chain_features(
                    features=processed_feats,
                    chain_id=chain_id,
                    min_idx=min_idx,
                    max_idx=max_idx,
                    rng=rng,
                    chain_max_len=None,
                )
                all_processed_feats.append(chain_processed_feats)
            processed_feats = concat_np_features(all_processed_feats, False)

    # Run through OpenFold data transforms.
    chain_feats = {
        "aatype": torch.tensor(processed_feats["aatype"]).long(),
        "all_atom_positions": torch.tensor(processed_feats["atom_positions"]).double(),
        "all_atom_mask": torch.tensor(processed_feats["atom_mask"]).double(),
    }
    chain_feats = data_transforms.atom37_to_frames(chain_feats)
    chain_feats = data_transforms.make_atom14_masks(chain_feats)
    chain_feats = data_transforms.make_atom14_positions(chain_feats)
    chain_feats = data_transforms.atom37_to_torsion_angles()(chain_feats)

    # Re-number residue indices for each chain such that it starts from 1.
    # Randomize chain indices.
    chain_idx: np.ndarray = processed_feats["chain_index"]
    res_idx: np.ndarray = processed_feats["residue_index"]
    new_res_idx = np.zeros_like(res_idx)

    all_chain_idx = list(np.unique(chain_idx))

    prev_len = 0
    for chain_id in all_chain_idx:
        chain_mask = (chain_idx == chain_id).astype(np.int64)
        chain_len = np.sum(chain_mask)
        new_res_idx[chain_mask.astype(bool)] = prev_len + np.arange(chain_len)

        # Add residue gap between chains to distinguish different chains.
        prev_len += chain_len + RESIDUE_GAP

    # To speed up processing, only take necessary features
    final_feats = {
        "aatype": chain_feats["aatype"],
        "seq_idx": new_res_idx,
        "chain_idx": chain_idx,
        "residx_atom14_to_atom37": chain_feats["residx_atom14_to_atom37"],
        "residue_index": processed_feats["residue_index"],
        "res_mask": processed_feats["bb_mask"],
        "atom37_pos": chain_feats["all_atom_positions"],
        "atom37_mask": chain_feats["all_atom_mask"],
        "atom14_pos": chain_feats["atom14_gt_positions"],
        "rigidgroups_0": chain_feats["rigidgroups_gt_frames"],
        "torsion_angles_sin_cos": chain_feats["torsion_angles_sin_cos"],
    }
    return final_feats


def read_cif(
    pdb_path: pathlib.Path | str,
    pdb_name: str,
    quiet: bool = True,
    return_first_model: bool = True,
) -> Structure.Structure | Model.Model:
    """Read PDB file and return the structure in it.

    Note: this function will return the BioPython structure, not
    the first model.

    Args:
        pdb_path: path to the PDB file.
        pdb_name: PDB name.
        quiet: whether to keep silence for warning.
            Default to True, no warning will be logged.
        return_first_model: if True, return the first model of the structure,
            else return the full structure.

    Returns:
        BioPython structure of the pdb.
    """
    mmcif_parser = MMCIFParser(QUIET=quiet)
    structure = mmcif_parser.get_structure(structure_id=pdb_name, filename=pdb_path)

    if return_first_model:
        structure = load_first_model(structure, pdb_name)

    return structure


def read_pdb(
    pdb_path: pathlib.Path | str,
    pdb_name: str,
    quiet: bool = True,
    return_first_model: bool = True,
) -> Structure.Structure | Model.Model:
    """Read PDB file and return the first model in it.

    Args:
        pdb_path: path to the PDB file.
        pdb_name: PDB name.
        quiet: whether to keep silence for warning.
            Default to True, no warning will be logged.
        return_first_model: if True, return the first model of the structure,
            else return the full structure.

    Returns:
        The first model in the PDB file.
    """
    pdb_parser = PDB.PDBParser(QUIET=quiet)
    structure = pdb_parser.get_structure(id=pdb_name, file=pdb_path)

    if return_first_model:
        structure = load_first_model(structure=structure, pdb_name=pdb_name)

    return structure


def load_first_model(structure: Structure.Structure, pdb_name: str) -> Model.Model:
    """Load the first model of a BioPython structure.

    Args:
        structure: BioPython structure.
        pdb_name: PDB name.

    Returns:
        First model of BioPython structure.
    """
    model = next(structure.get_models())
    model.detach_parent()
    model.id = pdb_name

    return model


def save_to_pdb(
    structure: Structure.Structure | Model.Model,
    pdb_path: pathlib.Path | str,
) -> None:
    """Write structure to .pdb file.

    Args:
        structure: BioPython structure or model object to write.
        pdb_path: Path object to write .pdb file to.
    """
    if isinstance(pdb_path, pathlib.Path):
        pdb_path = str(pdb_path)

    pdb_io = PDB.PDBIO()
    pdb_io.set_structure(structure)
    pdb_io.save(pdb_path)
