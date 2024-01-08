"""Module for inpainting evaluation on full atom predictions using cg2all."""
from __future__ import annotations

import pathlib
from typing import Any

import hydra
import omegaconf
import pandas as pd

from evaluation.utils.metrics import backbone_rmsd, full_atom_rmsd
from framedipt.data.utils import read_pdb


def get_full_atom_evaluation_df(
    inference_path: pathlib.Path,
) -> pd.DataFrame:
    """Get dataframe containing evaluation metrics for the full atom model.

    The dataframe will be saved under `eval_output_path`.

    Args:
        inference_path: path to the inference results.
        tcr_data_path: path to the tcr data containing the alpha and beta chain mapping.

    Returns:
        Dataframe containing evaluation metrics.
    """
    all_metrics = []

    for sample_folder in inference_path.glob("*_length_*"):
        folder_name = sample_folder.stem
        pdb_name_diffused_length = folder_name.replace("length_", "")
        pdb_name, diffused_length_str = pdb_name_diffused_length.split("_")
        diffused_length = int(diffused_length_str)

        gt_model = read_pdb(
            pdb_path=sample_folder / f"{pdb_name}_1.pdb",
            pdb_name=pdb_name,
        )

        # Get diffusion info.
        diffusion_info_file = sample_folder / "diffusion_info.csv"
        # Ensure backward compatibility to read old inference results.
        if not diffusion_info_file.exists():
            diffusion_info_file = sample_folder / "sample_0/diffusion_info.csv"
        diffusion_info = pd.read_csv(diffusion_info_file, sep="\t")
        seq = diffusion_info["seq"].to_numpy()[0]
        # chains is comma-separated chains being diffused.
        chains = diffusion_info["chain"].to_numpy()[0].split(",")
        # starts is comma-separated start indexes of the diffused region per chain.
        starts = list(
            map(int, diffusion_info["start"].to_numpy(dtype=str)[0].split(","))
        )
        # ends is comma-separated end indexes of the diffused region per chain.
        ends = list(map(int, diffusion_info["end"].to_numpy(dtype=str)[0].split(",")))

        diffusion_region = list(zip(starts, ends))

        # Iterate over samples.
        for sample in sample_folder.glob("sample_*"):
            metrics: dict[str, Any] = {
                "pdb_name": pdb_name,
                "diffused_length": diffused_length,
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

            sample_idx = int(sample.stem.split("_")[-1])
            metrics["sample_idx"] = sample_idx

            # Get sample model.
            sample_pdb_path = sample / f"sample_{sample_idx}_1_all_atom.pdb"
            sample_model = read_pdb(
                pdb_path=sample_pdb_path,
                pdb_name=f"{pdb_name}_sample",
                return_first_model=True,
            )

            # Get backbone RMSD.
            bb_rmsd = backbone_rmsd(
                model_1=gt_model,
                model_2=sample_model,
                chains=chains,
                model_1_diffusion_region=diffusion_region,
                model_2_diffusion_region=diffusion_region,
            )
            metrics["bb_rmsd"] = bb_rmsd

            # Get full atom RMSD
            fa_rmsd = full_atom_rmsd(
                model_1=gt_model,
                model_2=sample_model,
                chains=chains,
                model_1_diffusion_region=diffusion_region,
                model_2_diffusion_region=diffusion_region,
            )
            metrics["full_atom_rmsd"] = fa_rmsd

            all_metrics.append(metrics)

    df_metrics = pd.DataFrame(all_metrics)

    return df_metrics


@hydra.main(version_base="1.3.1", config_path="../config", config_name="evaluation")
def run(cfg: omegaconf.DictConfig) -> None:
    inference_path = pathlib.Path(cfg.inference_path)
    eval_output_path = pathlib.Path(cfg.eval_output_path)

    df_metrics = get_full_atom_evaluation_df(
        inference_path=inference_path,
    )

    eval_output_path.mkdir(parents=True, exist_ok=True)
    df_metrics.to_csv(eval_output_path / "eval_metrics.csv", sep="\t", index=False)


if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter
