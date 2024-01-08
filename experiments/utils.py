"""Utility functions for experiments.

Notes: in the scope of the code, "gt" stands for "ground truth".

"""
from __future__ import annotations

import copy
import os
import pathlib
import time
from typing import Any

import GPUtil
import numpy as np
import pandas as pd
import torch
from hydra.core.hydra_config import HydraConfig

from framedipt.data import utils as data_utils
from framedipt.diffusion import se3_diffuser
from framedipt.diffusion.r3_utils import gaussian_log_prob
from framedipt.diffusion.se3_diffuser import _extract_trans_rots
from framedipt.protein import all_atom
from framedipt.tools.log import get_logger
from openfold.utils import rigid_utils

logger = get_logger()


def is_running_on_aichor() -> bool:
    """Checks if the code is running on AIchor.

    The environment variable `RUNNING_ON_AICHOR` is defined to "true" on AIchor,
        so it will be automatically detected when we run on AIchor.

    Returns:
        True if the code is running on AIchor, False otherwise.
    """
    return os.environ.get("RUNNING_ON_AICHOR", "false") == "true"


def get_devices(
    use_gpu: bool = True, exp_name: str = ""
) -> tuple[list[str], str | None, str, str]:
    """Get devices on which to launch training/inference.

    Training could be run on multi-GPUs.
    Inference is run on single GPU or on CPU.

    Args:
        use_gpu: whether to use GPU.
        exp_name: experiment name.

    Returns:
        available_gpus: list of available GPUs,
            e.g. [] in case of no GPU and ["0", "1"] in case of 2 GPUs.
        gpu_id: the GPU id or None for CPU.
        device: string of device to use, e.g. "cpu", "cuda:0".
        exp_name: experiment name.
            if hydra.job.num is defined,
            experiment name will be added with suffix "_{hydra.job.num}".
    """
    # Decide which GPU to use using hydra.job.num if it's set.
    # Otherwise, default to 0.
    if HydraConfig.initialized() and "num" in HydraConfig.get().job:
        exp_name = f"{exp_name}_{HydraConfig.get().job.num}"
        replica_id = int(HydraConfig.get().job.num)
    else:
        replica_id = 0

    # Set up available gpus, gpu_id and device.
    if torch.cuda.is_available() and use_gpu:
        available_gpus = [str(x) for x in GPUtil.getAvailable(order="memory", limit=8)]
        gpu_id: str | None = available_gpus[replica_id]
        device = f"cuda:{gpu_id}"
    else:
        available_gpus = []
        gpu_id = None
        device = "cpu"

    logger.info(f"Using device: {device}")

    return available_gpus, gpu_id, device, exp_name


def flatten_dict(raw_dict: dict[str, Any]) -> list[tuple[str, Any]]:
    """Flatten a nested dict.

    It's a recursive function.
    If there is still nested dictionary in input `raw_dict`,
        it will continue to call the function.

    Examples:
        For input dictionary {"A": 1, "B": [2]},
            output is [("A", 1), ("B", [2])].
        For input dictionary {"A": {"alpha": 1, "beta": 2}, "B": [2]},
            output is [("A:alpha", 1), ("A:beta", 2), ("B", [2])].

    Args:
        raw_dict: raw nested dictionary.

    Returns:
        List of (key, value) tuples in the dictionary.
    """
    flattened: list[tuple[str, Any]] = []
    for k, v in raw_dict.items():
        if isinstance(v, dict):
            flattened.extend([(f"{k}:{i}", j) for i, j in flatten_dict(v)])
        else:
            flattened.append((k, v))
    return flattened


def t_stratified_loss(
    batch_t: np.ndarray,
    batch_loss: np.ndarray,
    num_bins: int = 5,
    t_threshold: float = 1.0,
    loss_name: str | None = None,
) -> dict[str, float]:
    """Stratify loss by binning t.

    Given the batch of timesteps and losses,
        we discretize timesteps to `num_bins` of bins,
        then compute average loss over different bins.
    It allows us to check loss values w.r.t. different timesteps.
    We would expect small/big loss for small/big timestep.

    Args:
        batch_t: batch of timesteps.
        batch_loss: batch of losses.
        num_bins: number of bins to use.
        t_threshold: the threshold of timestep t.
            We need this because some losses are only calculated
            for a certain threshold.
        loss_name: loss name.

    Returns:
        Dictionary of stratified losses w.r.t. binned timestep t.
    """
    max_t = 1.0
    flat_losses = batch_loss.flatten()
    flat_t = batch_t.flatten()
    bin_edges = np.linspace(0.0, t_threshold + 1e-3, num_bins + 1)
    bin_idx = np.sum(bin_edges[:, None] <= flat_t[None, :], axis=0) - 1
    t_binned_loss = np.bincount(bin_idx, weights=flat_losses)
    t_binned_n = np.bincount(bin_idx)
    stratified_losses = {}
    if loss_name is None:
        loss_name = "loss"
    unique_bin_idxs = list(np.unique(bin_idx))
    # if t_th is smaller than 1.0, will have an extra bin_idx
    # for t in [t_th, 1.0], need to remove it.
    if t_threshold < max_t:
        unique_bin_idxs = unique_bin_idxs[:-1]
    for t_bin in unique_bin_idxs:
        bin_start = bin_edges[t_bin]
        bin_end = bin_edges[t_bin + 1]
        t_range = f"{loss_name} t=[{bin_start:.2f},{bin_end:.2f})"
        range_loss = t_binned_loss[t_bin] / t_binned_n[t_bin]
        stratified_losses[t_range] = range_loss
    return stratified_losses


def _set_t_feats(
    feats: dict[str, torch.Tensor],
    t: float,
    t_placeholder: torch.Tensor,
    diffuser: se3_diffuser.SE3Diffuser,
) -> dict[str, torch.Tensor]:
    """Set features for timestep.

    Args:
        feats: dictionary of input features.
        t: timestep.
        t_placeholder: tensor of placeholder for timestep.
        diffuser: object of SE3Diffuser to compute score scaling.

    Returns:
        Dictionary of updated features with the following updated keys:
            - "t": updated to t * t_placeholder.
            - "rot_score_scaling": updated to rot_score_scaling * t_placeholder.
            - "trans_score_scaling": updated to trans_score_scaling * t_placeholder.
    """
    feats["t"] = t * t_placeholder
    rot_score_scaling, trans_score_scaling = diffuser.score_scaling(t)
    feats["rot_score_scaling"] = rot_score_scaling * t_placeholder
    feats["trans_score_scaling"] = trans_score_scaling * t_placeholder
    return feats


def self_conditioning(
    model: torch.nn.Module, batch: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Perform self-conditioning, using model predictions
        as self-conditioning inputs.

    Self-conditioning takes the model rigid-frame prediction
        as inputs for self-conditioning channel,
        thus the model will take its own predictions as part of inputs,
        therefore we call it self-conditioning.

    Args:
        model: torch model.
        batch: dictionary of batch data.
            - aatype: amino acid types, shape [Batch, N_res, 21].
            - seq_idx: 0-based residue indices, shape [Batch, N_res].
            - chain_idx: chain indices, shape [Batch, N_res].
            - residx_atom14_to_atom37: indices to convert atom14 to atom 37,
                shape [Batch, N_res, 14].
            - residue_index: raw residue indices in PDB file,
                shape [Batch, N_res].
            - res_mask: residue mask, shape [Batch, N_res].
            - atom37_pos: atom37 coordinates, shape [Batch, N_res, 37, 3].
            - atom37_mask: atom37 mask, shape [Batch, N_res, 37].
            - atom14_pos: atom14 coordinates, shape [Batch, N_res, 14, 3].
            - rigidgroups_0: rigid group representation at t = 0,
                shape [Batch, N_res, 8, 4, 4].
            - torsion_angles_sin_cos: torsion angle in sin-cos format,
                shape [Batch, N_res, 7, 2].
            - fixed_mask: mask for fixed residues, shape [Batch, N_res].
            - rigids_0: rigid representation at t = 0,
                shape [Batch, N_res, 7].
            - sc_ca_t: carbon-alpha coordinates used for self-conditioning,
                shape [Batch, N_res, 3].
            - rigids_t: rigid representation at timestep t,
                shape [Batch, N_res, 7].
            - t: timestep t, shape [Batch].
            Optional:
            - trans_score: translation score, shape [Batch, N_res, 3].
            - rot_score: rotation score, shape [Batch, N_res, 3].
            - trans_score_scaling: translation score scaling, shape [Batch].
            - rot_score_scaling: rotation score scaling, shape [Batch].

    Returns:
        Updated batch data containing self-conditioning inputs.
        The following value is changed:
            - sc_ca_t: carbon-alpha coordinates used for self-conditioning,
                it's updated by the rigids from model output.
    """
    model_sc = model(batch)

    # The first 4 elements in rigids is rotation quaternion,
    # and the last 3 elements is translation.
    # Self-conditioning is based only on translation.
    batch["sc_ca_t"] = model_sc["rigids"][..., 4:]
    return batch


def one_step_inference_score(
    model: torch.nn.Module,
    diffuser: se3_diffuser.SE3Diffuser,
    sample_feats: dict[str, torch.Tensor],
    t: float,
    t_placeholder: torch.Tensor,
    self_condition: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the translations and rotation scores from 1 inference step.

    Args:
        model: torch model to perform inference.
        diffuser: object of SE3Diffuser to perform reverse step.
        sample_feats: dictionary of sample features.
        t: current diffusion inference step, continuous in [0, 1].
        t_placeholder: placeholder to set timestep for inference.
        self_condition: whether to use self-condition.

    Returns:
        Tuple of translation and rotations scores, both with shape (N_res, 3).
    """
    # Set timestep in sample features.
    sample_feats = _set_t_feats(sample_feats, t, t_placeholder, diffuser)

    with torch.no_grad():
        # Perform self-conditioning
        if self_condition:
            sample_feats = self_conditioning(model, sample_feats)

        # Get model output
        model_out = model(sample_feats)

    # Shape [Batch, N_res, 3].
    rot_score = model_out["rot_score"]

    # Shape [Batch, N_res, 3].
    trans_score = model_out["trans_score"]

    return trans_score, rot_score


def one_step_inference(
    model: torch.nn.Module,
    diffuser: se3_diffuser.SE3Diffuser,
    sample_feats: dict[str, torch.Tensor],
    device: str,
    t: float,
    min_t: float,
    dt: float,
    t_placeholder: torch.Tensor,
    center: bool = True,
    aux_traj: bool = False,
    noise_scale: float = 1.0,
    embed_self_conditioning: bool = True,
    aatype: torch.Tensor | None = None,
) -> tuple[
    dict[str, torch.Tensor],
    torch.Tensor,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
]:
    """One-step diffusion inference.

    Args:
        model: torch model to perform inference.
        diffuser: object of SE3Diffuser to perform reverse step.
        sample_feats: dictionary of sample features.
        device: device to run inference.
        t: current diffusion inference step, continuous in [0, 1].
        min_t: minimum diffusion timestep.
        dt: time gap between two steps.
        t_placeholder: placeholder to set timestep for inference.
        center: whether to center the structure during reverse steps.
        aux_traj: whether to return auxiliary trajectories
            apart from protein trajectory.
        noise_scale: noise scale to use during inference.
        embed_self_conditioning: whether to embed self-condition in the model.
        aatype: optional AA types, shape [Batch, N_res].

    Returns:
        psi_pred: predicted torsion angles.
        bb_prot: protein backbone structure.
        aux_rigids: rigids of current timestep for auxiliary outputs.
        bb_0_pred: predicted backbone structure at timestep 0.
        trans_0_pred: predicted translation at timestep 0.
    """
    # Set timestep in sample features.
    # Get fixed mask and diffuse mask.
    sample_feats = _set_t_feats(sample_feats, t, t_placeholder, diffuser)
    # Shape [Batch, N_res].
    fixed_mask = sample_feats["fixed_mask"] * sample_feats["res_mask"]
    diffuse_mask = (1 - sample_feats["fixed_mask"]) * sample_feats["res_mask"]

    # Get model output
    model_out = model(sample_feats)
    # Shape [Batch, N_res, 7].
    rigid_pred = model_out["rigids"]

    # If t > min_t, run reverse step to get the rigids.
    if t > min_t:
        # Shape [Batch, N_res, 3].
        rot_score = model_out["rot_score"]
        trans_score = model_out["trans_score"]
        if embed_self_conditioning:
            # Get predicted translation for self-conditioning.
            sample_feats["sc_ca_t"] = rigid_pred[..., 4:]

        # Perform reverse step to get rigid_t-1 from rigid_t
        rigids_t_1 = diffuser.reverse(
            rigid_t=rigid_utils.Rigid.from_tensor_7(sample_feats["rigids_t"]),
            rot_score=data_utils.move_to_np(rot_score),
            trans_score=data_utils.move_to_np(trans_score),
            diffuse_mask=data_utils.move_to_np(diffuse_mask),
            t=t,
            dt=dt,
            center=center,
            noise_scale=noise_scale,
        )

    # Else take the predicted rigids from model outputs.
    else:
        rigids_t_1 = rigid_utils.Rigid.from_tensor_7(model_out["rigids"])

    # Update current rigid_t
    sample_feats["rigids_t"] = rigids_t_1.to_tensor_7().to(device)
    aux_rigids = None
    if aux_traj:
        # Shape [Batch, N_res, 7].
        aux_rigids = data_utils.move_to_np(rigids_t_1.to_tensor_7())

    # Calculate x_0 prediction derived from score predictions.
    # Shape [Batch, N_res, 3].
    gt_trans_0 = sample_feats["rigids_t"][..., 4:]
    pred_trans_0 = rigid_pred[..., 4:]
    trans_pred_0 = (
        diffuse_mask[..., None] * pred_trans_0 + fixed_mask[..., None] * gt_trans_0
    )

    # Calculate backbone from psi predictions.
    # Shape [Batch, N_res, 2].
    psi_pred = model_out["psi"]
    bb_0_pred = None
    trans_0_pred = None
    if aux_traj:
        # Shape [Batch, N_res, 37, 3].
        bb_0_pred = get_atom_positions_from_rigids(
            rigids=rigid_utils.Rigid.from_tensor_7(rigid_pred),
            psi_torsions=psi_pred,
            aatype=aatype,
        )
        # Shape [Batch, N_res, 3].
        trans_0_pred = data_utils.move_to_np(trans_pred_0)

    bb_prot = get_atom_positions_from_rigids(
        rigids=rigids_t_1,
        psi_torsions=psi_pred,
        aatype=aatype,
    )

    return sample_feats, psi_pred, bb_prot, aux_rigids, bb_0_pred, trans_0_pred


def get_atom_positions_from_rigids(
    rigids: rigid_utils.Rigid,
    psi_torsions: torch.Tensor,
    aatype: torch.Tensor,
) -> np.ndarray:
    """Get atom positions from rigid frames.

    Args:
        rigids: rigid frames, shape [Batch, N_res].
        psi_torsions: sin/cos representation of psi torsion angles,
            shape [Batch, N_res, 2].
        aatype: AA types, shape [Batch, N_res].

    Returns:
        Atom positions, shape [Batch, N_res, 37, 3].
    """
    # Shape [Batch, N_res, 37, 3] and [Batch, N_res, 37].
    atom37_t, atom37_t_mask, _, _ = all_atom.compute_backbone(
        rigids, psi_torsions, aatype=aatype
    )
    atom_positions = data_utils.move_to_np(atom37_t * atom37_t_mask[..., None])

    return atom_positions


def one_step_inference_recycle(
    model: torch.nn.Module,
    diffuser: se3_diffuser.SE3Diffuser,
    batch_feats: dict[str, torch.Tensor],
    device: str,
    t: torch.Tensor,
    dt: torch.Tensor,
    center: bool = True,
    noise_scale: float = 1.0,
    embed_self_conditioning: bool = True,
) -> dict[str, torch.Tensor]:
    """One step diffusion inference for recycling.

    Args:
        model: torch model to perform inference.
        diffuser: object of SE3Diffuser to perform reverse step.
        batch_feats: dictionary of sample features.
        device: device to run inference.
        t: current diffusion timestep.
        dt: time gap between timesteps.
        center: whether to center the structure during reverse steps.
        noise_scale: noise scale to use during inference.
        embed_self_conditioning: whether to embed self-condition in the model.

    Returns:
        batch_feats: updated dictionary of sample features at new timesteps.
    """
    # Shape [Batch, N_res].
    diffuse_mask = (1 - batch_feats["fixed_mask"]) * batch_feats["res_mask"]

    # Get model output
    model_out = model(batch_feats)
    # Shape [Batch, N_res, 7].
    rigid_pred = model_out["rigids"]

    # Shape [Batch, N_res, 3].
    rot_score = model_out["rot_score"]
    trans_score = model_out["trans_score"]
    if embed_self_conditioning:
        # Get predicted translation for self-conditioning.
        batch_feats["sc_ca_t"] = rigid_pred[..., 4:]

    # Perform reverse step to get rigid_t-1 from rigid_t
    # Loop over each sample in the batch and perform reverse step
    rigids_t_1_list: list[torch.Tensor] = []
    for i in range(batch_feats["t"].shape[0]):
        rigids_t_1_i = diffuser.reverse(
            rigid_t=rigid_utils.Rigid.from_tensor_7(batch_feats["rigids_t"][i]),
            rot_score=data_utils.move_to_np(rot_score[i]),
            trans_score=data_utils.move_to_np(trans_score[i]),
            diffuse_mask=data_utils.move_to_np(diffuse_mask[i]),
            t=t[i].item(),
            dt=dt[i].item(),
            center=center,
            noise_scale=noise_scale,
        )
        rigids_t_1_list.append(rigids_t_1_i.to_tensor_7())

    # Update current rigid_t
    batch_feats["rigids_t"] = torch.stack(rigids_t_1_list).to(device)

    # Get rot and trans score scaling for each t.
    score_scaling = torch.tensor([diffuser.score_scaling(t_i.item()) for t_i in t]).to(
        batch_feats["t"].device
    )
    batch_feats["t"] = t
    batch_feats["rot_score_scaling"] = score_scaling[:, 0]
    batch_feats["trans_score_scaling"] = score_scaling[:, 1]
    return batch_feats


def inference_fn(
    model: torch.nn.Module,
    diffuser: se3_diffuser.SE3Diffuser,
    data_init: dict[str, torch.Tensor],
    num_t: int,
    min_t: float,
    center: bool = True,
    aux_traj: bool = False,
    self_condition: bool = True,
    noise_scale: float = 1.0,
    embed_self_conditioning: bool = True,
    inpainting: bool = False,
    input_aatype: bool = False,
) -> dict[str, np.ndarray]:
    """Inference function.

    Args:
        model: torch model to perform inference.
        diffuser: object of SE3Diffuser to perform reverse step.
        data_init: Initial data values for sampling.
        num_t: number of diffusion timesteps to use during inference.
        min_t: minimum diffusion timestep to use during inference.
        center: whether to center the structure during reverse steps.
        aux_traj: whether to return auxiliary trajectories
            apart from protein trajectory.
        self_condition: whether to use self-condition.
        noise_scale: noise scale to use during inference.
        embed_self_conditioning: whether to embed self-condition in the model.
        inpainting: whether to perform inpainting inference.
        input_aatype: whether to input AA type in case of inpainting.

    Returns:
        trajectories during inference.
    """
    # Run reverse process.
    sample_feats = copy.deepcopy(data_init)
    device = sample_feats["rigids_t"].device

    feat_aatype = None if "aatype" not in sample_feats else sample_feats["aatype"]
    aatype = data_utils.preprocess_aatype(
        aatype=feat_aatype,
        fixed_mask=sample_feats["fixed_mask"],
        inpainting=inpainting,
        input_aatype=input_aatype,
    )

    if sample_feats["rigids_t"].ndim == 2:
        t_placeholder = torch.ones((1,)).to(device)
    else:
        t_placeholder = torch.ones((sample_feats["rigids_t"].shape[0],)).to(device)
    reverse_steps = np.linspace(min_t, 1.0, num_t)[::-1]
    dt = 1 / num_t

    # Init all rigids, all backbone trajectory, all x_0 preds, all backbone_0 preds
    all_rigids = [data_utils.move_to_np(copy.deepcopy(sample_feats["rigids_t"]))]
    all_bb_prots = []
    all_trans_0_pred = []
    all_bb_0_pred = []
    with torch.no_grad():
        # Perform self-conditioning
        if embed_self_conditioning and self_condition:
            sample_feats = _set_t_feats(
                sample_feats,
                reverse_steps[0],
                t_placeholder,
                diffuser,
            )
            sample_feats = self_conditioning(model, sample_feats)

        # Iterate over reverse timesteps.
        for t in reverse_steps:
            (
                sample_feats,
                psi_pred,
                bb_prot,
                aux_rigids,
                bb_0_pred,
                trans_0_pred,
            ) = one_step_inference(
                model=model,
                diffuser=diffuser,
                sample_feats=sample_feats,
                device=device,
                t=t,
                min_t=min_t,
                dt=dt,
                t_placeholder=t_placeholder,
                center=center,
                aux_traj=aux_traj,
                noise_scale=noise_scale,
                embed_self_conditioning=embed_self_conditioning,
                aatype=aatype,
            )
            all_bb_prots.append(bb_prot)
            if aux_rigids is not None:
                all_rigids.append(aux_rigids)
                all_bb_0_pred.append(bb_0_pred)
                all_trans_0_pred.append(trans_0_pred)

    # Flip trajectory so that it starts from t = 0.
    # This helps visualization.
    all_bb_prots = np.flip(np.stack(all_bb_prots), (0,))
    if aux_traj:
        all_rigids = np.flip(np.stack(all_rigids), (0,))
        all_trans_0_pred = np.flip(np.stack(all_trans_0_pred), (0,))
        all_bb_0_pred = np.flip(np.stack(all_bb_0_pred), (0,))

    ret = {
        "prot_traj": all_bb_prots,
    }
    if aux_traj:
        ret["rigid_traj"] = all_rigids
        ret["trans_traj"] = all_trans_0_pred
        ret["psi_pred"] = psi_pred[None]
        ret["rigid_0_traj"] = all_bb_0_pred
    return ret


def get_diffused_region_per_chain(
    diffused_mask: np.ndarray,
    chain_index: np.ndarray,
) -> tuple[list[int], list[int], list[int]]:
    """Get start and end indices of diffused region per chain.

    The diffused_mask is a mask over concatenation of all chains.
    This function gets diffused regions per chain
        by returning chain indices, starting indices and ending indices.

    Args:
        diffused_mask: array of diffused mask over all chains, shape [N_res,].
        chain_index: array of chain indices of the protein structure, shape [N_res,].

    Returns:
        chains: diffused chain indexes, starting with 0.
        starts: starting indexes per chain.
        ends: ending indexes per chain.
    """
    diffused_mask = diffused_mask.astype(bool)

    # Map chain indexes to start with 0,
    # and with continuous indexing.
    unique_chains = np.unique(chain_index)
    chain_index_mapping = {chain_idx: i for i, chain_idx in enumerate(unique_chains)}

    diffused_chains = np.unique(chain_index[diffused_mask])
    starts = []
    ends = []
    final_diffused_chains = []
    for chain_idx in diffused_chains:
        chain_mask = (chain_index == chain_idx).astype(bool)

        # Separate the current chain
        current_chain = chain_mask[chain_mask]

        # Separate the diffused region of the current chain
        current_chain_diffused_region = diffused_mask[chain_mask]

        # Get diffused indices of the current chain
        current_chain_diffused_indices = np.where(
            current_chain * current_chain_diffused_region
        )[0]

        # diff > 1 indicates end indexes of each diffused region.
        diff = np.diff(current_chain_diffused_indices)
        end_idxs = np.where(diff > 1)[0]
        start_idxs = [0, *list(end_idxs + 1)]
        end_idxs = [*list(end_idxs), -1]

        current_starts = current_chain_diffused_indices[start_idxs]
        current_ends = current_chain_diffused_indices[end_idxs]

        for current_start, current_end in zip(current_starts, current_ends):
            starts.append(current_start)
            ends.append(current_end)
            final_diffused_chains.append(chain_index_mapping[chain_idx])

    return final_diffused_chains, starts, ends


def save_diffusion_info(
    output_dir: pathlib.Path,
    pdb_name: str,
    seq: str,
    diffused_mask: np.ndarray,
    chain_index: np.ndarray,
) -> None:
    """Save info about the diffusion region as csv file.

    We want to save diffusion regions per chain
        for better readability.

    The saved csv file contains columns:
        - pdb_name: PDB name.
        - seq: AA sequence.
        - chain: the id of the chain being diffused.
        - start: start index of the diffused region in the chain.
        - end: end index of the diffused region in the chain.

    Args:
        output_dir: directory to save the csv file.
        pdb_name: PDB name.
        seq: AA sequence.
        diffused_mask: array of diffused mask, shape [N_res,].
        chain_index: array of chain indices, shape [N_res,].

    Raises:
        ValueError if length of
            diffused_mask and chain_index is not the same.
    """
    if len(diffused_mask) != len(chain_index):
        raise ValueError(
            f"Length of diffused_mask and chain_index should be the same, "
            f"got {len(diffused_mask)} != {len(chain_index)}."
        )

    # Filter non-standard residues.
    standard_aa_mask = np.array([c != "X" for c in seq])
    diffused_mask = diffused_mask[standard_aa_mask]
    chain_index = chain_index[standard_aa_mask]

    chains, start_indexes, end_indexes = get_diffused_region_per_chain(
        diffused_mask=diffused_mask,
        chain_index=chain_index,
    )

    chains_str = ",".join([chr(ord("A") + chain_num) for chain_num in chains])
    start_indexes_str = ",".join([str(start_idx) for start_idx in start_indexes])
    end_indexes_str = ",".join([str(end_idx) for end_idx in end_indexes])

    diffusion_info = {
        "pdb_name": pdb_name,
        "seq": seq,
        "chain": chains_str,
        "start": start_indexes_str,
        "end": end_indexes_str,
    }
    csv_path = output_dir / "diffusion_info.csv"
    df_info = pd.DataFrame([diffusion_info])
    df_info.to_csv(csv_path, sep="\t", index=False)


def logp_confidence_score(
    model: torch.nn.Module,
    diffuser: se3_diffuser.SE3Diffuser,
    rigids_t: rigid_utils.Rigid,
    sample_feats: dict[str, torch.Tensor],
    diffuse_mask: np.ndarray,
    num_t: int,
    min_t: float,
    device: str,
    self_condition: bool,
) -> tuple[float, list[float]]:
    """Confidence score from EigenFold: https://arxiv.org/abs/2304.02198

    Args:
        model: torch model.
        diffuser: object of SE3Diffuser to perform reverse step.
        rigids_0: predicted rigid-frames with shape (N, 6) (axis-angle form).
        sample_feats: Initial data values for sampling.
        diffuse_mask: diffusion mask with 1 indicating
            the residue is diffused, shape [N].
        num_t: number of diffusion timesteps to use during inference.
        min_t: minimum diffusion timestep to use during inference.
        device: device to process torch tensors on.
        self_condition: if True, use self-conditioning information.

    Returns:
        Tuple of:
            log_prob: confidence score from EigenFold.
            log_probs: list of confidence scores from each timestep.
    """
    # Convert timesteps to continuous values between [0, 1].
    forward_steps = np.linspace(min_t, 1.0, num_t)[:-1]
    t_placeholder = torch.ones((1,)).to(device)

    # Calculate time gap between two steps.
    dt = 1 / num_t

    log_probs = []
    log_prob = 0.0
    for i, t_1 in enumerate(forward_steps):
        rigids_t_1 = copy.deepcopy(rigids_t)

        # Do 1 forward noising step on both translations and rotations.
        rigids_t = diffuser.forward(
            rigids_t_1=rigids_t_1,
            t_1=t_1,
            diffuse_mask=diffuse_mask,
            dt=dt,
        )

        # Update the sample feats with the new rigid representation.
        sample_feats["rigids_t"] = rigids_t.to_tensor_7().to(device)

        # Increment the timestep. The last timestep will be set to 1.0.
        t = 1.0 if i == len(forward_steps) - 1 else forward_steps[i + 1]

        # Do a single step of inference to get the score from the model.
        trans_score, rot_score = one_step_inference_score(
            model=model,
            diffuser=diffuser,
            sample_feats=sample_feats,
            t=t,
            t_placeholder=t_placeholder,
            self_condition=self_condition,
        )

        # Convert to numpy and remove extra batch dim.
        trans_score = data_utils.move_to_np(trans_score.squeeze())
        rot_score = data_utils.move_to_np(rot_score.squeeze())

        # Calculate the log probability of the backwards distributions
        log_prob_backward = diffuser.log_prob_backward(
            rigids_t=rigids_t,
            rigids_t_1=rigids_t_1,
            trans_score_t=trans_score,
            rot_score_t=rot_score,
            dt=dt,
            t=t,
            diffuse_mask=diffuse_mask,
        )
        log_prob += log_prob_backward

        # Calculate the log probability of the forwards distribution
        log_prob_forward = diffuser.log_prob_forward(
            rigids_t=rigids_t,
            rigids_t_1=rigids_t_1,
            dt=dt,
            t_1=t_1,
            diffuse_mask=diffuse_mask,
        )
        log_prob -= log_prob_forward

        log_probs.append(log_prob)

    # Extract the translations and rotations at t=T.
    trans, _ = _extract_trans_rots(rigids_t)  # pylint: disable=W0212

    # Scale the trans values.
    trans = diffuser._r3_diffuser._scale(trans)  # pylint: disable=W0212
    trans = torch.tensor(trans).to(device)

    # Evaluate log probability of the final prediction from
    # a standard normal distribution.
    mu = torch.zeros_like(trans)
    std = torch.ones_like(trans)
    diffuse_mask = torch.tensor(diffuse_mask).to(device)
    trans_log_prob = gaussian_log_prob(
        mu=mu, std=std, x=trans, diffuse_mask=diffuse_mask
    )

    # Angles should be uniformly distributed.
    n_diffused_res = diffuse_mask.sum().item()
    rot_log_prob = np.log(1 / np.pi**2) * n_diffused_res

    log_prob += trans_log_prob + rot_log_prob
    log_probs.append(log_prob)

    return log_prob, log_probs


class Timer:
    """Timer class to factor out training/inference time and speed calculations."""

    def __init__(self) -> None:
        """Initialise timer"""
        self._start_time = time.time()
        self._counter = 0

    def reset(self) -> None:
        """Reset timer, sets start time to now, and counter to 0."""
        self._start_time = time.time()
        self._counter = 0

    def step(self, n: int) -> None:
        """Increment counter by n."""
        self._counter += n

    def get_rate(self) -> float:
        """Calculate rate (counter value / elapsed time in seconds)."""
        elapsed_time = self.get_time()
        return self._counter / elapsed_time

    def get_time(self) -> float:
        """Get elapsed time in seconds since init/reset"""
        return time.time() - self._start_time

    def get_rate_and_reset(self) -> float:
        """Calculate rate and reset timer."""
        rate = self.get_rate()
        self.reset()
        return rate
