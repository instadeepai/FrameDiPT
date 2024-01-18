"""Module to evaluation TCR structures."""
from __future__ import annotations

import functools
import pathlib
from collections import defaultdict
from collections.abc import Generator, Iterable, Sequence
from typing import Any, Callable, Protocol

import hydra
import pandas as pd
from Bio.PDB import Model
from omegaconf import omegaconf

from evaluation.utils.constants import (
    DIHEDRAL_ANGLES,
    EVAL_METRICS,
    SAMPLE_SELECTION_STRATEGY,
    XTICKS,
)
from evaluation.utils.directory_parser import traverse_prediction_dir
from evaluation.utils.metrics import (  # full_atom_rmsd,; residue_full_atom_rmsd,
    asa_abs_error,
    asa_square_error,
    average_metrics_for_middle_residues,
    backbone_rmsd,
    chain_backbone_rmsd,
    flatten,
    full_atom_rmsd,
    gt_asa,
    gt_rsa,
    residue_angle_error,
    residue_backbone_rmsd,
    residue_groundtruth_angle,
    residue_sample_angle,
    residue_signed_angle_error,
    rsa_abs_error,
    rsa_square_error,
    sample_asa,
    sample_rsa,
)
from evaluation.utils.plot import (
    boxplot_metrics_alpha_beta,
    swarmplot_metrics_alpha_beta,
)
from evaluation.utils.sample_selection import get_selected_models
from framedipt.data.utils import read_pdb


class ModelCallable(Protocol):
    def __call__(
        self,
        model_1: Model.Model,
        model_2: Model.Model,
        chains: list[str] | str,
        model_1_diffusion_region: list[tuple[int, int]] | tuple[int, int],
        model_2_diffusion_region: list[tuple[int, int]] | tuple[int, int],
    ) -> float:
        ...


class ChainCallable(Protocol):
    def __call__(
        self,
        model_1: Model.Model,
        model_2: Model.Model,
        chains: list[str] | str,
        model_1_diffusion_region: list[tuple[int, int]] | tuple[int, int],
        model_2_diffusion_region: list[tuple[int, int]] | tuple[int, int],
    ) -> dict[str, float]:
        ...


class ResidueCallable(Protocol):
    def __call__(
        self,
        model_1: Model.Model,
        model_2: Model.Model,
        chains: list[str] | str,
        model_1_diffusion_region: list[tuple[int, int]] | tuple[int, int],
        model_2_diffusion_region: list[tuple[int, int]] | tuple[int, int],
    ) -> dict[str, dict[int, float]]:  # chain, idx
        ...


class GroupCallable(Protocol):
    def __call__(
        self,
        model_1: Model.Model,
        model_2: Model.Model,
        chains: list[str] | str,
        model_1_diffusion_region: list[tuple[int, int]] | tuple[int, int],
        model_2_diffusion_region: list[tuple[int, int]] | tuple[int, int],
    ) -> dict[str, dict[str, dict[int, float]]]:  # angle, chain, idx
        ...


MODEL_METRIC_NAME_TO_FN: dict[str, ModelCallable] = {
    "bb_rmsd": backbone_rmsd,
    "full_atom_rmsd": full_atom_rmsd,
}

CHAIN_METRIC_NAME_TO_FN: dict[str, ChainCallable] = {
    "bb_rmsd": chain_backbone_rmsd,
    # "full_atom_rmsd": full_atom_rmsd,
}

RESIUDE_METRIC_NAME_TO_FN: dict[str, ResidueCallable] = {
    "bb_rmsd": residue_backbone_rmsd,
    # "full_atom_rmsd": residue_full_atom_rmsd,
    "gt_asa": gt_asa,
    "sample_asa": sample_asa,
    "asa_abs_error": asa_abs_error,
    "asa_square_error": asa_square_error,
    "gt_rsa": gt_rsa,
    "sample_rsa": sample_rsa,
    "rsa_abs_error": rsa_abs_error,
    "rsa_square_error": rsa_square_error,
}

RESIDUE_GROUP_METRIC_NAME_TO_FN: dict[str, GroupCallable] = {
    "angle_error": residue_angle_error,
    "signed_angle_error": residue_signed_angle_error,
    "sample": residue_sample_angle,
    "gt": residue_groundtruth_angle,
}

METRIC_TYPES: dict[str, dict[str, Callable]] = {
    "model_metrics": MODEL_METRIC_NAME_TO_FN,
    "chain_metrics": CHAIN_METRIC_NAME_TO_FN,
    "residue_metrics": RESIUDE_METRIC_NAME_TO_FN,
    "residue_group_metrics": RESIDUE_GROUP_METRIC_NAME_TO_FN,
}


def parse_gt_structures(
    pdb_ids: list[str], gt_pdb_paths: list[pathlib.Path]
) -> Generator[Model.Model, None, None]:
    """Parse ground-truth structure.

    Args:
        pdb_ids: list of pdb names to process.
        gt_pdb_paths: list of paths to the ground-truth structure.

    Returns:
        models: list of parse BioPython models.
    """
    pdb_ids_and_path = zip(pdb_ids, gt_pdb_paths)
    models = (
        read_pdb(pdb_path=path, pdb_name=pdb_id) for pdb_id, path in pdb_ids_and_path
    )
    return models


def parse_samples(
    pdb_ids: list[str], predicted_pdb_paths: list[list[pathlib.Path]]
) -> list[Generator[Model.Model, None, None]]:
    """Parse the model predictions into BioPython models.

    Args:
        pdb_ids: list of pdb names to process.
        predicted_pdb_paths: nested list of sample prediction,
            many samples per pdb_id.

    Returns:
        predicted_models: nested list of parsed BioPython models.
    """
    pdb_ids_and_path = zip(pdb_ids, predicted_pdb_paths)

    predicted_models = []
    for pdb_id, paths in pdb_ids_and_path:
        predicted_model_generator = samples_generator(pdb_id, paths)
        predicted_models.append(predicted_model_generator)

    return predicted_models


def samples_generator(
    pdb_id: str, paths: list[pathlib.Path]
) -> Generator[Model.Model, None, None]:
    for i, path in enumerate(paths):
        pdb_name = f"{pdb_id}_{i}"
        predicted_model = read_pdb(pdb_path=path, pdb_name=pdb_name)
        yield predicted_model


def parse_diffusion_info_df(
    df: pd.DataFrame,
    cdr_loop_index: int,
) -> tuple[str, list[str], list[int], list[int]]:
    """Parse the diffusion info dataframe.

    Args:
        df: dataframe containing the diffusion metadata.

    Returns:
        Tuple with the following elements:
            seq: extracted chain sequences. Shape (1).
            chains: chains in diffused region. Shape (n_diffused_regions).
            starts: start indices of the diffused region. Shape (n_diffused_regions).
            ends: end indices of the diffused region. Shape (n_diffused_regions).
    """
    # Extract sequence information
    seq = df["seq"].to_numpy()[0]

    # Extract which chains are being diffused.
    chains = df["chain"].to_numpy()[0].split(",")

    # Extract the start indices of the diffusion region per chain
    # and convert from str to int.
    starts = df["start"].to_numpy(dtype=str)[0].split(",")
    starts = list(map(int, starts))

    # Extract the end indices of the diffusion region per chain
    # and convert from str to int.
    ends = df["end"].to_numpy(dtype=str)[0].split(",")
    ends = list(map(int, ends))

    # in case of multiple loop diffusion, e.g. 2 loops per tcr chain.
    # chains will be [A,A,B,B],
    # starts will be the corresponding start indexes of the 4 regions,
    # the same for ends.
    if len(chains) > 2:
        num_loops = len(chains) // 2
        chains = [chains[cdr_loop_index], chains[num_loops + cdr_loop_index]]
        starts = [starts[cdr_loop_index], starts[num_loops + cdr_loop_index]]
        ends = [ends[cdr_loop_index], ends[num_loops + cdr_loop_index]]

    return seq, chains, starts, ends


def parse_diffusion_info(
    diffusion_info_paths: list[pathlib.Path], cdr_loop_index: int = 2
) -> dict[str, Any]:
    """Parse the diffusion_info.csv file.

    Args:
        diffusion_info_paths: list of paths to the diffusion_info.csv files.

    Returns:
        diffusion_info_dict: dictionary with the following key-value paris:
            - diffused_sequences: list of diffused chain sequences, shape (n_pdbs).
              Note, sequences are currently concatenated.
            - diffused_chains: nested list of diffused chains,
              shape (n_pdbs, n_diffused_region).
            - diffused_start_indices: nested list of diffused start indices,
              shape (n_pdbs, n_diffused_region).
            - diffused_end_indices:nested list of diffused end indices,
              shape (n_pdbs, n_diffused_region).
    """
    seqs = []
    chains = []
    starts = []
    ends = []

    for diffusion_info_path in diffusion_info_paths:
        # Read in dataframe from csv.
        df = pd.read_csv(diffusion_info_path, sep="\t")

        # Extract diffused region info from the dataframe.
        seq, chain, start, end = parse_diffusion_info_df(
            df, cdr_loop_index=cdr_loop_index
        )

        # Save the extracted info.
        seqs.append(seq)
        chains.append(chain)
        starts.append(start)
        ends.append(end)

    diffusion_info_dict = {
        "diffused_sequences": seqs,
        "diffused_chains": chains,
        "diffused_start_indices": starts,
        "diffused_end_indices": ends,
    }

    return diffusion_info_dict


def compute_metrics(
    pdb_ids: list[str],
    gt_models: Generator[Model.Model, None, None],
    predicted_models: list[Generator[Model.Model, None, None]],
    predicted_pdb_paths: list[list[pathlib.Path]],
    sequences: list[str],
    diffused_chains: list[list[str]],
    diffused_start_indices: list[list[int]],
    diffused_end_indices: list[list[int]],
    metric_cfg: omegaconf.DictConfig,
) -> dict[str, pd.DataFrame]:
    """Compute metrics.

    Args:
        pdb_ids: list of pdb names to compute metrics on.
        gt_models: list of ground truth BioPython models.
        predicted_models: nested list of predicted models, shape (n_pdb_ids, n_samples).
        predicted_pdb_paths: nested list of predicted pdb paths,
            shape (n_pdb_ids, n_samples).
        sequences: list of diffused sequences, shape (n_pdb_ids).
        diffused_chains: nested list of diffused chains,
            shape (n_pdb_ids, n_diffused_regions).
        diffused_start_indices: list of diffused region start indices,
            shape (n_pdb_ids, n_diffused_regions).
        diffused_end_indices: list of diffused region end indices,
            shape (n_pdb_ids, n_diffused_regions).
        metric_cfg: list configuration of what metrics to compute.

    Returns:
        all_df_metrics: dictionary of dataframes of evaluation metrics
            according to different sample selection strategy.
    """
    all_df_metrics = {}
    all_metrics = []
    all_sample_selection_metrics = defaultdict(list)
    dataset_info = (
        pdb_ids,
        sequences,
        diffused_start_indices,
        diffused_end_indices,
        gt_models,
        predicted_models,
        predicted_pdb_paths,
        diffused_chains,
    )
    for pdb_data in zip(*dataset_info):
        # Get information associated with the current pdb
        (
            pdb_id,
            seq,
            starts,
            ends,
            gt_model,
            predicted_model,
            predicted_pdb_path,
            chains,
        ) = pdb_data

        predicted_model = list(predicted_model)

        selected_models = get_selected_models(
            predicted_models=predicted_model,
            predicted_pdb_paths=predicted_pdb_path,
            starts=starts,
            ends=ends,
            chains=chains,
        )

        all_structure_metrics, sample_selection_metrics = generate_metric_dict(
            metric_cfg=metric_cfg,
            pdb_id=pdb_id,
            seq=seq,
            starts=starts,
            ends=ends,
            chains=chains,
            gt_model=gt_model,
            predicted_model=predicted_model,
            predicted_pdb_path=predicted_pdb_path,
            selected_model=selected_models,
        )
        all_metrics.extend(all_structure_metrics)
        for strategy in SAMPLE_SELECTION_STRATEGY:
            metrics = sample_selection_metrics[strategy]
            all_sample_selection_metrics[strategy].append(metrics)

    all_df_metrics["all"] = pd.DataFrame(all_metrics)
    for strategy in SAMPLE_SELECTION_STRATEGY:
        all_df_metrics[strategy] = pd.DataFrame(all_sample_selection_metrics[strategy])

    return all_df_metrics


def update_metrics(
    metrics: dict[str, Any],
    metric_cfg: omegaconf.DictConfig,
    sample_path: pathlib.Path,
    gt_model: Model.Model,
    sample_model: Model.Model,
    chains: list[str],
    diffusion_region: list[tuple[int, int]],
) -> dict[str, Any]:
    """Update the dictionary of metrics according to `metric_cfg`.

    Args:
        metrics: metrics to be updated.
        metric_cfg: hydra configuration specifying metrics.
        sample_path: path to the sample pdb file.
        gt_model: biopython model of the ground truth structure.
        sample_model: biopython model of the sample structure.
        chains: chains in diffused region. Shape (n_diffused_regions).
        diffusion_region: list of diffused region tuples (start, end)
            representing the start and end residue indices.

    Returns:
        Updated metrics.
    """
    updated_metrics = metrics.copy()
    pdb_path_splits = sample_path.stem.split("_")
    for split in reversed(pdb_path_splits):
        try:
            sample_idx = int(split)
            updated_metrics["sample_idx"] = sample_idx
        except ValueError:
            continue

    metric_groups: dict[str, dict[str, Any]] = {}
    for metric_type, name_to_func in METRIC_TYPES.items():
        metric_groups[metric_type] = {}
        metric_names = metric_cfg[metric_type]
        for metric_name in metric_names:
            metric_fn = name_to_func[metric_name]
            metric = metric_fn(
                model_1=gt_model,
                model_2=sample_model,
                chains=chains,
                model_1_diffusion_region=diffusion_region,
                model_2_diffusion_region=diffusion_region,
            )
            metric_groups[metric_type][metric_name] = metric

    for metric_dict in metric_groups.values():
        updated_metrics.update(flatten(metric_dict))

    return updated_metrics


def generate_metric_dict(
    metric_cfg: omegaconf.DictConfig,
    pdb_id: str,
    seq: str,
    starts: Sequence[int],
    ends: Sequence[int],
    chains: list[str],
    gt_model: Model.Model,
    predicted_model: Iterable[Model.Model],
    predicted_pdb_path: list[pathlib.Path],
    selected_model: dict[str, dict[str, Model.Model | pathlib.Path]],
) -> tuple[list[dict[str, float]], dict[str, dict[str, Any]]]:
    """Create per-sample list of requested metrics for a given model
    Args:
        metric_cfg: hydra configuration specifying metrics,
        pdb_id: PDB identifier, e.g. 2BNU
        seq: string corresponding to residue sequence
        starts: list-like containing the start residue indices of the diffused regions
        ends: list-like containing the end residue indices of the diffused regions
        chains: list-like containing the corresponding diffused region chain IDs
        gt_model: ground truth Bio.PDB.Model to compare predictions against
        predicted_model: selected predicted model to compare against gt_model
        predicted_pdb_path: selected predicted pdb file path
        selected_model: dictionary of selected model according to different
            sample selection strategy.

    Returns:
        all_structure_metrics: list of metric dictionaries.
            Each entry in the list corresponds to one predicted model.
            Each dictionary key corresponds to a metric that has been calculated
            for a particular residue.
        sample_selection_metrics: dictionary of metric dictionaries according to
            sample selection strategy.
    """
    all_structure_metrics = []
    base_metrics: dict[str, Any] = {
        "pdb_name": pdb_id,
        "structure_length": len(seq),
        "tcr_alpha_chain": chains[0],
        "tcr_beta_chain": chains[1],
        "tcr_alpha_chain_start_idx": starts[0],
        "tcr_alpha_chain_end_idx": ends[0],
        "tcr_alpha_chain_diffused_length": ends[0] - starts[0] + 1,
        "tcr_beta_chain_start_idx": starts[1],
        "tcr_beta_chain_end_idx": ends[1],
        "tcr_beta_chain_diffused_length": ends[1] - starts[1] + 1,
    }

    diffusion_region = list(zip(starts, ends))
    for idx, sample_model in enumerate(predicted_model):
        sample_path = predicted_pdb_path[idx]
        metrics = update_metrics(
            metrics=base_metrics,
            metric_cfg=metric_cfg,
            sample_path=sample_path,
            gt_model=gt_model,
            sample_model=sample_model,
            chains=chains,
            diffusion_region=diffusion_region,
        )
        all_structure_metrics.append(metrics.copy())

    sample_selection_metrics = {}
    for strategy, sample_model in selected_model.items():
        metrics = update_metrics(
            metrics=base_metrics,
            metric_cfg=metric_cfg,
            sample_path=sample_model["path"],
            gt_model=gt_model,
            sample_model=sample_model["model"],
            chains=chains,
            diffusion_region=diffusion_region,
        )
        sample_selection_metrics[strategy] = metrics.copy()

    return all_structure_metrics, sample_selection_metrics


@hydra.main(version_base="1.3.1", config_path="../config", config_name="evaluation")
def run(cfg: omegaconf.DictConfig) -> None:
    inference_path = pathlib.Path(cfg.inference_path)
    eval_output_path = pathlib.Path(cfg.eval_output_path)
    # pathlib.Path(cfg.tcr_data_path)

    # Traverse prediction directory
    prediction_info_dict = traverse_prediction_dir(
        inference_path=inference_path,
        legacy_file_structure=cfg.legacy,
    )

    # Parse ground-truth structures.
    pdb_ids = prediction_info_dict["pdb_ids"]
    gt_pdb_paths = prediction_info_dict["gt_pdb_paths"]
    # List of shape (n_pdb_ids).
    gt_models = parse_gt_structures(pdb_ids, gt_pdb_paths)

    # Parse predicted structures.
    predicted_pdb_paths = prediction_info_dict["predicted_pdb_paths"]
    # List of shape (n_pdb_ids, n_samples).
    predicted_models = parse_samples(pdb_ids, predicted_pdb_paths)

    # Parse diffusion info csv file
    diffusion_info_paths = prediction_info_dict["diffusion_info_paths"]
    diffusion_info_dict = parse_diffusion_info(
        diffusion_info_paths, cdr_loop_index=cfg.cdr_loop_index
    )

    # Compute metrics
    all_metrics_df = compute_metrics(
        pdb_ids=pdb_ids,
        gt_models=gt_models,
        predicted_models=predicted_models,
        predicted_pdb_paths=predicted_pdb_paths,
        sequences=diffusion_info_dict["diffused_sequences"],
        diffused_chains=diffusion_info_dict["diffused_chains"],
        diffused_start_indices=diffusion_info_dict["diffused_start_indices"],
        diffused_end_indices=diffusion_info_dict["diffused_end_indices"],
        metric_cfg=cfg.metrics,
    )

    # Write metrics to file
    eval_output_path.mkdir(
        parents=True, exist_ok=True
    )  # Make sure output directory exists.
    for key, metrics_df in all_metrics_df.items():
        metrics_df.to_csv(
            eval_output_path / f"eval_metrics_{key}.csv", index=False, sep="\t"
        )

    if cfg.sample_selection_strategy:
        try:
            metrics_df = all_metrics_df[cfg.sample_selection_strategy]
        except KeyError as e:
            raise NotImplementedError(
                f"Strategy {cfg.sample_selection_strategy} not implemented."
            ) from e
    else:
        metrics_df = all_metrics_df["all"]

    evaluation_plot(
        metrics_df,
        eval_output_path=eval_output_path,
        metric_cfg=cfg.metrics,
        swarmplot=cfg.swarmplot,
    )


def evaluation_plot(
    df_metrics: pd.DataFrame,
    eval_output_path: pathlib.Path,
    metric_cfg: omegaconf.DictConfig,
    swarmplot: bool,
) -> None:
    """Get evaluation plots.

    Args:
        df_metrics: dataframe containing evaluation results.
        eval_output_path: path to save the evaluation output plot.
        metric_cfg: configuration of what metrics to plot.
    """
    df_median_rmsd_grouped_by_pdb_name = df_metrics.groupby("pdb_name")[
        "bb_rmsd"
    ].median()
    df_analyse = pd.merge(df_metrics, df_median_rmsd_grouped_by_pdb_name)
    bb_rmsds_alpha = df_analyse["bb_rmsd_alpha"].to_numpy()
    bb_rmsds_beta = df_analyse["bb_rmsd_beta"].to_numpy()

    # swarm/boxplot backbone RMSDs.
    if swarmplot:
        plot_fn = functools.partial(swarmplot_metrics_alpha_beta, superpose_box=True)
    else:
        plot_fn = boxplot_metrics_alpha_beta  # type: ignore[assignment]
    plot_fn(
        metrics_alpha=[bb_rmsds_alpha],
        metrics_beta=[bb_rmsds_beta],
        eval_output_path=eval_output_path,
        legend="Backbone RMSD",
        xticks=["Backbone RMSD"],
    )

    for metric_type in METRIC_TYPES:
        metric_names = metric_cfg[metric_type]
        for metric_name in metric_names:
            if metric_type == "residue_group_metrics":
                # handle angles separately.
                for angle in DIHEDRAL_ANGLES:
                    metric = f"{metric_name}_{angle}"
                    try:
                        legend = EVAL_METRICS[metric]
                        plot_per_residue_graph(
                            df_analyse, eval_output_path, legend, metric, swarmplot
                        )
                    except KeyError:
                        continue

            else:
                metric = metric_name
                try:
                    legend = EVAL_METRICS[metric_name]
                except KeyError:
                    continue
                plot_per_residue_graph(
                    df_analyse, eval_output_path, legend, metric, swarmplot
                )


def plot_per_residue_graph(
    df_metrics: pd.DataFrame,
    eval_output_path: pathlib.Path,
    legend: str,
    metric: str,
    swarmplot: bool,
) -> None:
    metrics_alpha_beta = average_metrics_for_middle_residues(
        df_metrics=df_metrics,
        metric=metric,
    )
    if swarmplot:
        plot_fn = functools.partial(swarmplot_metrics_alpha_beta, superpose_box=False)
    else:
        plot_fn = boxplot_metrics_alpha_beta  # type: ignore[assignment]
    plot_fn(
        metrics_alpha=metrics_alpha_beta["alpha"],
        metrics_beta=metrics_alpha_beta["beta"],
        eval_output_path=eval_output_path,
        legend=legend,
        xticks=XTICKS,
    )


if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter
