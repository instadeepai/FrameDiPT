"""Module for sample selection utils."""
from __future__ import annotations

import copy
import pathlib
from collections import defaultdict
from collections.abc import Sequence
from typing import Protocol

import numpy as np
import numpy.linalg as la
from Bio.PDB import Model

from evaluation.utils.metrics import get_backbone_atom_coords
from framedipt.data.utils import save_to_pdb
from framedipt.protein.residue_constants import BACKBONE_ATOMS


class SampleSelectionCallable(Protocol):
    def __call__(
        self,
        starts: Sequence[int],
        ends: Sequence[int],
        chains: list[str],
        sample_diffused_region_coords: dict[str, np.ndarray],
        predicted_models: list[Model.Model],
        predicted_pdb_paths: list[pathlib.Path],
    ) -> tuple[Model.Model, pathlib.Path]:  # biopython model, pdb file path
        ...


def get_model_diffused_region_coords(
    model: Model.Model,
    chain_ids: list[str],
    diffused_regions: list[tuple[int, int]],
) -> dict[str, np.ndarray]:
    """Get diffused region backbone atom coordinates of a given model.

    Args:
        model: the biopython model to be processed.
        chain_ids: list of chain ids. Shape (n_diffused_regions).
        diffused_regions: list of diffused region tuples (start, end)
            representing the start and end residue indices.

    Returns:
        Dictionary of diffused region backbone atom coordinates
            according to chain ids.
    """
    chain_coords = {}
    for chain_id, diffused_region in zip(chain_ids, diffused_regions):
        start, end = diffused_region
        coords = get_backbone_atom_coords(
            model=model,
            chain_id=chain_id,
            start=start,
            end=end,
        )
        chain_coords[chain_id] = coords

    return chain_coords


def gaussian_density_estimation(
    samples: np.ndarray,
    sigma: float = 30.0,
) -> np.ndarray:
    """Gaussian density estimation.

    Args:
        samples: input array to estimate the density, shape (N, M).
        sigma: the std of the Gaussian kernel, default value 30.

    Returns:
        Kernel density of each sample, shape (N,).
    """
    diffs = samples[:, None, :] - samples[None, :, :]
    squared_dists = (diffs**2).sum(axis=-1)
    kernel_density = np.exp(-squared_dists / (sigma**2))
    return kernel_density.sum(axis=1)


def weiszfeld_geometric_median(
    samples: np.ndarray,
    start: np.ndarray,
    max_iterations: int = 10000,
) -> np.ndarray:
    """Weiszfeld's algorithm for geometric median.

    Reference: https://en.wikipedia.org/wiki/Geometric_median

    Args:
        samples: input array to estimate the density, shape (N, M).
        start: the starting point of the algorithm, shape (M,).
        max_iterations: Maximum number of iterations, default 10000.

    Returns:
        Geometric median of input samples, shape (M,).
    """
    out = start
    for _ in range(max_iterations):
        displacements = samples - out[None, ...]
        distances = la.norm(displacements, ord=2, axis=-1)  # div 0 issue!
        inv_dist_sum = (1 / distances).sum()
        weighted_samples = samples / distances[..., None]
        out = weighted_samples.sum(axis=0) / inv_dist_sum
    return out


def get_selected_sample_model_and_path(
    starts: Sequence[int],
    ends: Sequence[int],
    chains: list[str],
    predicted_model: Model.Model,
    predicted_pdb_path: pathlib.Path,
    sample_diffused_region_coords: dict[str, np.ndarray] | None,
    sample_selection_strategy: str,
    update_model: bool = True,
) -> tuple[Model.Model, pathlib.Path]:
    """Get selected model and pdb path given a sample selection strategy.

    For `mean` and `median` strategy, the model should be updated
    by replacing the diffused region coordinates.

    Args:
        starts: start indices of the diffused region. Shape (n_diffused_regions).
        ends: end indices of the diffused region. Shape (n_diffused_regions).
        chains: chains in diffused region. Shape (n_diffused_regions).
        predicted_model: biopython model of the predicted structure.
        predicted_pdb_path: path to the predicted pdb file.
        sample_diffused_region_coords: dictionary of the coordinates to
            replace the original coordinates in the diffused region.
            {chain_id: sample coordinates} with shape
            (n_samples, diffused_region_length, 4, 3).
        sample_selection_strategy: sample selection strategy, should be
            mean, median, mode, mean_closest or median_closest.
        update_model: whether to update model, default to True.

    Returns:
        Tuple of model and pdb path.
    """
    if update_model:
        if sample_diffused_region_coords is None:
            raise ValueError("Sample coordinates should be provided to update model.")

        model = replace_coords(
            model=predicted_model,
            starts=starts,
            ends=ends,
            chains=chains,
            coords=sample_diffused_region_coords,
        )
        save_to_pdb(
            structure=model,
            pdb_path=predicted_pdb_path.parent
            / f"sample_{sample_selection_strategy}.pdb",
        )
    else:
        model = predicted_model

    return model, predicted_pdb_path


def get_mean_coordinates(
    sample_diffused_region_coords: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Get mean coordinates of samples.

    Args:
        sample_diffused_region_coords: dictionary of the coordinates to
            replace the original coordinates in the diffused region.
            {chain_id: sample coordinates} with shape
            (n_samples, diffused_region_length, 4, 3).

    Returns:
        Dictionary of median coordinates {chain_id: median coordinates}
        with shape (diffused_region_length, 4, 3).
    """
    mean_sample_coords = {
        k: np.mean(v, axis=0) for k, v in sample_diffused_region_coords.items()
    }
    return mean_sample_coords


def flatten_diffused_region_coords(
    sample_diffused_region_coords: dict[str, np.ndarray],
) -> np.ndarray:
    """Flatten diffused region coordinates.

    For example, if `sample_diffused_region_coords` is
    {"A": array of shape (5, 10, 4, 3),
    "B": array of shape (5, 13, 4, 3)},
    it will be flattened to an array of shape (5, 276).

    Args:
        sample_diffused_region_coords: dictionary of the coordinates to
            replace the original coordinates in the diffused region.
            {chain_id: sample coordinates} with shape
            (n_samples, diffused_region_length, 4, 3).

    Returns:
        Flattened array of diffused region coordinates.
    """
    chains = list(sample_diffused_region_coords.keys())

    num_samples = sample_diffused_region_coords[chains[0]].shape[0]
    flattened_coords = {
        k: v.reshape(num_samples, -1) for k, v in sample_diffused_region_coords.items()
    }
    diffused_region_flattened_coords = np.concatenate(
        list(flattened_coords.values()), axis=1
    )

    return diffused_region_flattened_coords


def get_median_coordinates(
    sample_diffused_region_coords: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Get geometric median coordinates of samples.

    Args:
        sample_diffused_region_coords: dictionary of the coordinates to
            replace the original coordinates in the diffused region.
            {chain_id: sample coordinates} with shape
            (n_samples, diffused_region_length, 4, 3).

    Returns:
        Dictionary of median coordinates {chain_id: median coordinates}
        with shape (diffused_region_length, 4, 3).
    """
    # Initialize median sample coords with mean coords
    median_sample_coords = get_mean_coordinates(sample_diffused_region_coords)

    chains = list(sample_diffused_region_coords.keys())

    diffused_region_flattened_coords = flatten_diffused_region_coords(
        sample_diffused_region_coords
    )

    median_diffused_region_coords = weiszfeld_geometric_median(
        diffused_region_flattened_coords, diffused_region_flattened_coords.mean(axis=0)
    ).reshape(-1, 4, 3)

    chain_index = np.concatenate(
        [
            np.full(sample_diffused_region_coords[chain_id].shape[1], i)
            for i, chain_id in enumerate(chains)
        ]
    )
    for i, chain_id in enumerate(chains):
        median_sample_coords[chain_id] = median_diffused_region_coords[chain_index == i]

    return median_sample_coords


def get_mode_index(
    sample_diffused_region_coords: dict[str, np.ndarray],
) -> int:
    """Get sample index of `mode` strategy.

    The sample with the highest Gaussian kernel density is selected.

    Args:
        sample_diffused_region_coords: dictionary of the coordinates to
            replace the original coordinates in the diffused region.
            {chain_id: sample coordinates} with shape
            (n_samples, diffused_region_length, 4, 3).

    Returns:
        Selected sample index.
    """
    diffused_region_flattened_coords = flatten_diffused_region_coords(
        sample_diffused_region_coords
    )
    density = gaussian_density_estimation(diffused_region_flattened_coords)
    sample_index = density.argmax().item()

    return sample_index


def get_closest_index(
    sample_diffused_region_coords: dict[str, np.ndarray],
    reference_coords: dict[str, np.ndarray],
) -> int:
    """Get sample index of `closest` strategy.

    The closest sample to the reference coordinates is selected.

    Args:
        sample_diffused_region_coords: dictionary of the coordinates to
            replace the original coordinates in the diffused region.
            {chain_id: sample coordinates} with shape
            (n_samples, diffused_region_length, 4, 3).
        reference_coords: the coordinates of the reference
            to compute the distance.

    Returns:
        Selected sample index.
    """
    flattened_sample_coords = np.concatenate(
        list(sample_diffused_region_coords.values()), axis=1
    )

    flattened_reference_coords = np.concatenate(
        list(reference_coords.values()), axis=0
    )[np.newaxis]

    reference_ca_dists = np.nansum(
        np.linalg.norm(
            flattened_sample_coords - flattened_reference_coords,
            axis=-1,
        ),
        axis=(-2, -1),
    )
    sample_index = reference_ca_dists.argmin().item()

    return sample_index


def mean_sample(
    starts: Sequence[int],
    ends: Sequence[int],
    chains: list[str],
    sample_diffused_region_coords: dict[str, np.ndarray],
    predicted_models: list[Model.Model],
    predicted_pdb_paths: list[pathlib.Path],
) -> tuple[Model.Model, pathlib.Path]:
    """Get sample by `mean` strategy
    which is simply the mean coordinates of predicted samples.

    Args:
        starts: start indices of the diffused region. Shape (n_diffused_regions).
        ends: end indices of the diffused region. Shape (n_diffused_regions).
        chains: chains in diffused region. Shape (n_diffused_regions).
        sample_diffused_region_coords: dictionary of the coordinates to
            replace the original coordinates in the diffused region.
            {chain_id: sample coordinates} with shape
            (n_samples, diffused_region_length, 4, 3).
        predicted_models: list of biopython models of the predicted structure.
        predicted_pdb_paths: list of paths to the predicted pdb file.

    Returns:
        Tuple of model and pdb path.
    """
    mean_sample_coords = get_mean_coordinates(sample_diffused_region_coords)

    model, path = get_selected_sample_model_and_path(
        starts=starts,
        ends=ends,
        chains=chains,
        predicted_model=predicted_models[0],
        predicted_pdb_path=predicted_pdb_paths[0],
        sample_diffused_region_coords=mean_sample_coords,
        sample_selection_strategy="mean",
        update_model=True,
    )

    return model, path


def median_sample(
    starts: Sequence[int],
    ends: Sequence[int],
    chains: list[str],
    sample_diffused_region_coords: dict[str, np.ndarray],
    predicted_models: list[Model.Model],
    predicted_pdb_paths: list[pathlib.Path],
) -> tuple[Model.Model, pathlib.Path]:
    """Get sample by `median` strategy
    which is simply the geometric median coordinates of predicted samples.

    Args:
        starts: start indices of the diffused region. Shape (n_diffused_regions).
        ends: end indices of the diffused region. Shape (n_diffused_regions).
        chains: chains in diffused region. Shape (n_diffused_regions).
        sample_diffused_region_coords: dictionary of the coordinates to
            replace the original coordinates in the diffused region.
            {chain_id: sample coordinates} with shape
            (n_samples, diffused_region_length, 4, 3).
        predicted_models: list of biopython models of the predicted structure.
        predicted_pdb_paths: list of paths to the predicted pdb file.

    Returns:
        Tuple of model and pdb path.
    """
    median_sample_coords = get_median_coordinates(sample_diffused_region_coords)

    model, path = get_selected_sample_model_and_path(
        starts=starts,
        ends=ends,
        chains=chains,
        predicted_model=predicted_models[0],
        predicted_pdb_path=predicted_pdb_paths[0],
        sample_diffused_region_coords=median_sample_coords,
        sample_selection_strategy="median",
        update_model=True,
    )

    return model, path


def mode_sample(
    starts: Sequence[int],
    ends: Sequence[int],
    chains: list[str],
    sample_diffused_region_coords: dict[str, np.ndarray],
    predicted_models: list[Model.Model],
    predicted_pdb_paths: list[pathlib.Path],
) -> tuple[Model.Model, pathlib.Path]:
    """Get sample by `mode` strategy
    which is the sample with the highest Gaussian kernel density.

    Args:
        starts: start indices of the diffused region. Shape (n_diffused_regions).
        ends: end indices of the diffused region. Shape (n_diffused_regions).
        chains: chains in diffused region. Shape (n_diffused_regions).
        sample_diffused_region_coords: dictionary of the coordinates to
            replace the original coordinates in the diffused region.
            {chain_id: sample coordinates} with shape
            (n_samples, diffused_region_length, 4, 3).
        predicted_models: list of biopython models of the predicted structure.
        predicted_pdb_paths: list of paths to the predicted pdb file.

    Returns:
        Tuple of model and pdb path.
    """
    sample_index = get_mode_index(sample_diffused_region_coords)

    model, path = get_selected_sample_model_and_path(
        starts=starts,
        ends=ends,
        chains=chains,
        predicted_model=predicted_models[sample_index],
        predicted_pdb_path=predicted_pdb_paths[sample_index],
        sample_diffused_region_coords=None,
        sample_selection_strategy="mode",
        update_model=False,
    )

    return model, path


def mean_closest_sample(
    starts: Sequence[int],
    ends: Sequence[int],
    chains: list[str],
    sample_diffused_region_coords: dict[str, np.ndarray],
    predicted_models: list[Model.Model],
    predicted_pdb_paths: list[pathlib.Path],
) -> tuple[Model.Model, pathlib.Path]:
    """Get sample by `mean_closest` strategy
    which is the closest sample to the mean coordinates of predicted samples.

    Args:
        starts: start indices of the diffused region. Shape (n_diffused_regions).
        ends: end indices of the diffused region. Shape (n_diffused_regions).
        chains: chains in diffused region. Shape (n_diffused_regions).
        sample_diffused_region_coords: dictionary of the coordinates to
            replace the original coordinates in the diffused region.
            {chain_id: sample coordinates} with shape
            (n_samples, diffused_region_length, 4, 3).
        predicted_models: list of biopython models of the predicted structure.
        predicted_pdb_paths: list of paths to the predicted pdb file.

    Returns:
        Tuple of model and pdb path.
    """
    mean_sample_coords = get_mean_coordinates(sample_diffused_region_coords)

    sample_index = get_closest_index(
        sample_diffused_region_coords=sample_diffused_region_coords,
        reference_coords=mean_sample_coords,
    )

    model, path = get_selected_sample_model_and_path(
        starts=starts,
        ends=ends,
        chains=chains,
        predicted_model=predicted_models[sample_index],
        predicted_pdb_path=predicted_pdb_paths[sample_index],
        sample_diffused_region_coords=None,
        sample_selection_strategy="mean_closest",
        update_model=False,
    )

    return model, path


def median_closest_sample(
    starts: Sequence[int],
    ends: Sequence[int],
    chains: list[str],
    sample_diffused_region_coords: dict[str, np.ndarray],
    predicted_models: list[Model.Model],
    predicted_pdb_paths: list[pathlib.Path],
) -> tuple[Model.Model, pathlib.Path]:
    """Get sample by `median_closest` strategy which is
    the closest sample to the geometric median coordinates of predicted samples.

    Args:
        starts: start indices of the diffused region. Shape (n_diffused_regions).
        ends: end indices of the diffused region. Shape (n_diffused_regions).
        chains: chains in diffused region. Shape (n_diffused_regions).
        sample_diffused_region_coords: dictionary of the coordinates to
            replace the original coordinates in the diffused region.
            {chain_id: sample coordinates} with shape
            (n_samples, diffused_region_length, 4, 3).
        predicted_models: list of biopython models of the predicted structure.
        predicted_pdb_paths: list of paths to the predicted pdb file.

    Returns:
        Tuple of model and pdb path.
    """
    median_sample_coords = get_median_coordinates(sample_diffused_region_coords)

    sample_index = get_closest_index(
        sample_diffused_region_coords=sample_diffused_region_coords,
        reference_coords=median_sample_coords,
    )

    model, path = get_selected_sample_model_and_path(
        starts=starts,
        ends=ends,
        chains=chains,
        predicted_model=predicted_models[sample_index],
        predicted_pdb_path=predicted_pdb_paths[sample_index],
        sample_diffused_region_coords=None,
        sample_selection_strategy="median_closest",
        update_model=False,
    )

    return model, path


def replace_coords(
    model: Model.Model,
    starts: Sequence[int],
    ends: Sequence[int],
    chains: list[str],
    coords: dict[str, np.ndarray],
) -> Model.Model:
    """Replace coordinates of diffused region.

    Args:
        model: biopython model to be processed.
        starts: start indices of the diffused region. Shape (n_diffused_regions).
        ends: end indices of the diffused region. Shape (n_diffused_regions).
        chains: chains in diffused region. Shape (n_diffused_regions).
        coords: the coordinates to replace the original coordinates
            in the diffused region.

    Returns:
        Replaced model.
    """
    replaced_model = copy.deepcopy(model)
    for chain_id, start, end in zip(chains, starts, ends):
        chain = replaced_model[chain_id]
        chain_coords = coords[chain_id]
        residues = list(chain.get_residues())
        for idx, res_idx in enumerate(range(start, end + 1)):
            residue = residues[res_idx]
            for atom_idx, atom_name in enumerate(BACKBONE_ATOMS):
                residue[atom_name].coord = chain_coords[idx][atom_idx]

    return replaced_model


def get_selected_models(
    predicted_models: list[Model.Model],
    predicted_pdb_paths: list[pathlib.Path],
    starts: Sequence[int],
    ends: Sequence[int],
    chains: list[str],
) -> dict[str, dict[str, Model.Model | pathlib.Path]]:
    """Get selected models according to different sample selection strategies.

    Args:
        predicted_models: iterable of biopython models of predicted structures.
        predicted_pdb_paths: list of corresponding pdb paths.
        starts: start indices of the diffused region. Shape (n_diffused_regions).
        ends: end indices of the diffused region. Shape (n_diffused_regions).
        chains: chains in diffused region. Shape (n_diffused_regions).

    Returns:
        Dictionary of selected models of different strategies.
    """
    # Get sample diffused region coordinates
    sample_diffused_region_coords_lists = defaultdict(list)
    for sample_model in predicted_models:
        sample_diffused_region_coords = get_model_diffused_region_coords(
            model=sample_model,
            chain_ids=chains,
            diffused_regions=list(zip(starts, ends)),
        )
        for chain_id, diffused_region_coords in sample_diffused_region_coords.items():
            sample_diffused_region_coords_lists[chain_id].append(diffused_region_coords)
    sample_diffused_region_coords = {
        k: np.stack(v, axis=0) for k, v in sample_diffused_region_coords_lists.items()
    }

    # Get the selected model and path for each strategy
    selected_models: dict[str, dict] = defaultdict(dict)
    for strategy, strategy_fn in SAMPLE_SELECTION_STRATEGY_TO_FN.items():
        model, path = strategy_fn(
            starts=starts,
            ends=ends,
            chains=chains,
            sample_diffused_region_coords=sample_diffused_region_coords,
            predicted_models=predicted_models,
            predicted_pdb_paths=predicted_pdb_paths,
        )
        selected_models[strategy]["model"] = model
        selected_models[strategy]["path"] = path

    return selected_models


# Mapping from sample selection strategy to the corresponding function.
SAMPLE_SELECTION_STRATEGY_TO_FN: dict[str, SampleSelectionCallable] = {
    "mean": mean_sample,
    "median": median_sample,
    "mode": mode_sample,
    "mean_closest": mean_closest_sample,
    "median_closest": median_closest_sample,
}
