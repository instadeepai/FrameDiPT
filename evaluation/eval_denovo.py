"""Module for de novo protein design evaluation."""
from __future__ import annotations

import logging
import pathlib
import re
import subprocess

import hydra
import matplotlib.colors as mcolor
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial import distance

from framedipt.analysis import metrics
from framedipt.data import utils as data_utils

logger = logging.getLogger(__name__)


def argmedian(array: np.ndarray) -> int:
    """Get the index of median value in array.

    Args:
        array: input array.

    Returns:
        Index of median value.
    """
    middle_pos = len(array) // 2
    arg_partition = np.argpartition(array, middle_pos)
    return arg_partition[middle_pos]


def get_rmsd_df(
    data_path: pathlib.Path,
    output_file: pathlib.Path,
    lengths: list[int] | None = None,
    choice: str = "best",
) -> pd.DataFrame:
    """Get dataframe of RMSD results.

    Args:
        data_path: path to inference results.
        output_file: csv file to write dataframe output.
        lengths: list of lengths to be analysed.
            Default to None, meaning every length is considered.
        choice: choice for ESMFold samples, should be "best" or "median".
            Default to "best".

    Returns:
        Dataframe containing RMSD results.
    """
    metrics_dict: dict[str, list] = {
        "length": [],
        "sample": [],
        "esmf_sample": [],
        "tm_score": [],
        "rmsd": [],
    }

    if not lengths:
        lengths = []
        for directory in data_path.glob("length_*"):
            lengths.append(int(directory.stem.replace("length_", "")))

    for length in lengths:
        directory = data_path / f"length_{length}"
        if not directory.exists():
            continue
        for gen_sample in directory.glob("sample_*"):
            sname = gen_sample.stem
            sc_res = pd.read_csv(gen_sample / "self_consistency/sc_results.csv")
            esmf_samples = sc_res.index[1:].tolist()
            tm_scores = sc_res["tm_score"].iloc[1:].to_numpy()
            rmsds = sc_res["rmsd"].iloc[1:].to_numpy()
            if choice == "best":
                rmsd_idx = np.argmin(rmsds).item()
            elif choice == "median":
                rmsd_idx = argmedian(rmsds)
            else:
                raise ValueError(f"choice should be 'best' or 'median', got {choice}.")
            metrics_dict["length"].append(length)
            metrics_dict["sample"].append(sname)
            metrics_dict["esmf_sample"].append(esmf_samples[rmsd_idx])
            metrics_dict["tm_score"].append(tm_scores[rmsd_idx])
            metrics_dict["rmsd"].append(rmsds[rmsd_idx])

    df = pd.DataFrame(metrics_dict)
    df.to_csv(output_file, sep="\t", index=False)

    return df


def plot_metric(
    df: pd.DataFrame,
    df_pretrained: pd.DataFrame | None,
    lengths: list[int],
    outdir: pathlib.Path,
    metric: str = "rmsd",
    choice: str = "best",
) -> None:
    """Plot RMSD or TM-score comparing our model and pre-trained model.

    Figures will be saved under `outdir`.

    Args:
        df: dataframe containing evaluation metrics of our model.
        df_pretrained: optional dataframe containing evaluation metrics
            of pre-trained model.
        lengths: list of lengths to be analysed.
        outdir: output directory to save plots.
        metric: metric to plot, should be "rmsd" or "tm_score".
            Default to "rmsd".
        choice: choice for ESMFold samples, should be "best" or "median".
            Default to "best".
    """
    if metric not in ["rmsd", "tm_score"]:
        raise ValueError(f"metric should be 'rmsd' or 'tm_score', got {metric}.")
    xs = []
    all_metrics = []
    pretrained_all_metrics = []

    # For each length, get the metric results
    # in df and df_pretrained.
    for i, length in enumerate(lengths):
        xs.append(i)
        metrics_values = df[df["length"] == length][metric].to_numpy()
        all_metrics.append(metrics_values)

        if df_pretrained is not None:
            pretrained_metrics = df_pretrained[df_pretrained["length"] == length][
                metric
            ].to_numpy()
            pretrained_all_metrics.append(pretrained_metrics)

    # Boxplot the retrieved metrics per length.
    plt.figure(figsize=(8, 6))
    bboxs = [
        plt.boxplot(
            all_metrics,
            showfliers=False,
            patch_artist=True,
            positions=np.array(xs) + 0.8,
            widths=0.3,
        )
    ]
    legends = ["Our Model"]
    if len(pretrained_all_metrics) > 0:
        bboxs.append(
            plt.boxplot(
                pretrained_all_metrics,
                showfliers=False,
                patch_artist=True,
                positions=np.array(xs) + 1.2,
                widths=0.3,
            )
        )
        legends.append("Pretrained Model")

    colors = ["seagreen", "royalblue"]
    for i, bplot in enumerate(bboxs):
        color = colors[i]
        for patch in bplot["boxes"]:
            patch.set_facecolor(color)

    if metric == "rmsd":
        plt.plot(np.linspace(0, len(lengths) + 1, 100), [2] * 100, "r--")
    else:
        plt.plot(np.linspace(0, len(lengths) + 1, 100), [0.5] * 100, "r--")
    plt.xticks(np.array(xs) + 1, lengths)
    plt.xlabel("Length")
    plt.ylabel(f"self-consistency {metric.upper()}")
    plt.ylim(bottom=0)
    plt.title(f"{metric.upper()} ({choice} sample)", fontsize=20)
    plt.tight_layout()
    plt.legend([bbox["boxes"][0] for bbox in bboxs], legends, fontsize=16)
    plt.savefig(outdir / f"analyse_{metric}_{choice}.png")


def plot_rmsd_tm_score(
    pretrained_data_path: pathlib.Path | None,
    data_path: pathlib.Path,
    outdir: pathlib.Path,
    choice: str = "best",
    overwrite: bool = False,
) -> None:
    """Plot RMSD and TM-score comparing our model and pre-trained model.

    Args:
        pretrained_data_path: optional path to inference results of pre-trained model.
        data_path: path to inference results of our model.
        outdir: output directory to save plots.
        choice: choice for ESMFold samples, should be "best" or "median".
            Default to "best".
        overwrite: whether to overwrite existing RMSD/TM-score results.
    """
    csv_file = outdir / f"analyse_rmsd_{choice}.csv"

    if not csv_file.exists() or overwrite:
        df = get_rmsd_df(data_path, output_file=csv_file, choice=choice)
    else:
        df = pd.read_csv(csv_file, sep="\t")

    lengths = sorted(df["length"].unique())

    if pretrained_data_path is not None:
        pretrained_csv_file = outdir / f"analyse_rmsd_{choice}_pretrained.csv"
        if not pretrained_csv_file.exists() or overwrite:
            df_pretrained = get_rmsd_df(
                pretrained_data_path,
                output_file=pretrained_csv_file,
                lengths=lengths,
                choice=choice,
            )
        else:
            df_pretrained = pd.read_csv(pretrained_csv_file, sep="\t")
    else:
        df_pretrained = None

    plot_metric(df, df_pretrained, lengths, outdir, metric="rmsd", choice=choice)
    plot_metric(df, df_pretrained, lengths, outdir, metric="tm_score", choice=choice)


def write_samples_pdbs(
    data_path: pathlib.Path,
    outdir: pathlib.Path,
) -> list[pathlib.Path]:
    """Write generated samples' PDB file paths for each length
     and return list of written files.

    Args:
        data_path: path to inference results.
        outdir: output directory to write files.

    Returns:
        List of written files.
    """
    all_list_paths = []
    for directory in data_path.glob("length_*"):
        lines = []
        for gen_sample in directory.glob("sample_*"):
            sample_pdb = gen_sample / f"{gen_sample.stem}_1.pdb"
            lines.append(str(sample_pdb))
            lines.append("\n")

        path_to_pdb_list = outdir / f"all_samples_pdb_{directory.stem}.list"
        with open(path_to_pdb_list, "w", encoding="utf-8") as f:
            f.writelines(lines)

        all_list_paths.append(path_to_pdb_list)

    return all_list_paths


def maxcluster_diversity(
    pdb_list_file: pathlib.Path,
    outdir: pathlib.Path,
    length: int,
    tm_score_th: float = 0.5,
    pretrained: bool = False,
) -> float:
    """Compute diversity using MaxCluster.

    Diversity is defined as number of clusters / number of samples.

    Args:
        pdb_list_file: file containing generated samples' PDB file paths.
        outdir: output directory to save MaxCluster results.
        length: length of sequence to be analysed.
        tm_score_th: TM-score threshold to do clustering.
        pretrained: whether it's pretrained model or not.

    Returns:
        Computed diversity.
    """
    suffix = ""
    if pretrained:
        suffix = "_pretrained"
    path_to_align_score = outdir / f"align_score_length_{length}{suffix}.txt"
    path_to_cluster = outdir / f"cluster_length_{length}{suffix}.txt"

    if not path_to_align_score.exists():
        # Run MaxCluster to get pairwise align score for PDBs in pdb_list_file.
        # Results are saved to path_to_align_score.
        maxcluster_score_args = [
            "maxcluster",
            "-l",
            str(pdb_list_file),
            "-in",
            "-Rl",
            str(path_to_align_score),
        ]

        logger.info(maxcluster_score_args)
        with subprocess.Popen(
            maxcluster_score_args,  # noqa: S603
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        ) as process:
            _ = process.communicate()

    # Search size info by matching regular expression in the results.
    # An example of the searching line: SIZE : 100
    pattern = re.compile("^SIZE : .+$")
    with open(path_to_align_score, encoding="utf-8") as f:
        for i, line in enumerate(f):
            match = re.match(pattern, line)
            if match:
                logger.info("Found on line %s: %s", i + 1, match.group())
                eles = line.split()
                size = int(eles[2])
                break

    # Run MaxCluster to do clustering using previous align score.
    if not path_to_cluster.exists():
        maxcluster_cluster_args = [
            "maxcluster",
            "-C",
            "1",
            "-M",
            str(path_to_align_score),
            "-T",
            str(tm_score_th),
            "-Tm",
            str(tm_score_th),
        ]

        logger.info(maxcluster_cluster_args)
        with open(path_to_cluster, "w", encoding="utf-8") as f, subprocess.Popen(
            maxcluster_cluster_args,  # noqa: S603
            stdout=f,
            stderr=subprocess.STDOUT,
        ) as process:
            _ = process.communicate()

    # Search number of clusters by matching regular expression in the results.
    # An example of the searching line: INFO  : 94 Clusters @ Threshold  0.500 (0.5)
    pattern = re.compile("^.+ Clusters @ Threshold .+$")
    with open(path_to_cluster, encoding="utf-8") as f:
        for i, line in enumerate(f):
            match = re.match(pattern, line)
            if match:
                logger.info("Found on line: %s:%s", i + 1, match.group())
                eles = line.split()
                num_clusters = int(eles[2])
                break

    diversity = num_clusters / size
    return diversity


def get_diversity_df(
    data_path: pathlib.Path,
    output_file: pathlib.Path,
    tm_score_th: float = 0.5,
    use_hierarchy: bool = False,
    pretrained: bool = False,
) -> pd.DataFrame:
    """Get dataframe of diversity results.

    Args:
        data_path: path to inference results of our model.
        output_file: csv file to write dataframe output.
        tm_score_th: TM-score threshold to do clustering.
        use_hierarchy: whether to use hierarchy clustering or MaxCluster.
        pretrained: whether it's pretrained model or not.

    Returns:
        Dataframe containing diversity results.
    """
    lengths = []
    diversities = []
    if use_hierarchy:
        for directory in data_path.glob("length_*"):
            length = int(directory.stem.replace("length_", ""))
            diversity = hierarchy_diversity(
                data_path,
                output_file.parent,
                length,
                tm_score_th,
                pretrained=pretrained,
            )
            lengths.append(length)
            diversities.append(diversity)
    else:
        all_list_paths = write_samples_pdbs(data_path, output_file.parent)
        for list_path in all_list_paths:
            length = int(list_path.stem.split("length_")[-1])
            diversity = maxcluster_diversity(
                list_path,
                output_file.parent,
                length,
                tm_score_th,
                pretrained=pretrained,
            )
            lengths.append(length)
            diversities.append(diversity)
    df = pd.DataFrame({"length": lengths, "diversity": diversities})
    df.to_csv(output_file, sep="\t", index=False)

    return df


def plot_diversity(
    pretrained_data_path: pathlib.Path | None,
    data_path: pathlib.Path,
    outdir: pathlib.Path,
    tm_score_th: float = 0.5,
    use_hierarchy: bool = False,
    overwrite: bool = False,
) -> None:
    """Plot diversity.

    Figure will be saved under `outdir`.

    Args:
        pretrained_data_path: optional path to inference results of pre-trained model.
        data_path: path to inference results of our model.
        outdir: output directory to save plot.
        tm_score_th: TM-score threshold to do clustering.
        use_hierarchy: whether to use hierarchy clustering or MaxCluster.
        overwrite: whether to overwrite existing diversity results.
    """
    suffix = ""
    if use_hierarchy:
        suffix = "_hierarchy"

    res_csv_file = outdir / f"analyse_diversity{suffix}.csv"
    if not res_csv_file.exists() or overwrite:
        df = get_diversity_df(
            data_path=data_path,
            output_file=res_csv_file,
            tm_score_th=tm_score_th,
            use_hierarchy=use_hierarchy,
            pretrained=False,
        )
    else:
        df = pd.read_csv(res_csv_file, sep="\t")

    if pretrained_data_path is not None:
        pretrained_res_csv_file = outdir / f"analyse_diversity_pretrained{suffix}.csv"
        if not pretrained_res_csv_file.exists() or overwrite:
            df_pretrained = get_diversity_df(
                data_path=pretrained_data_path,
                output_file=pretrained_res_csv_file,
                tm_score_th=tm_score_th,
                use_hierarchy=use_hierarchy,
                pretrained=True,
            )
        else:
            df_pretrained = pd.read_csv(pretrained_res_csv_file, sep="\t")
    else:
        df_pretrained = None

    lengths = df["length"].to_numpy()
    diversities = df["diversity"].to_numpy()

    argsort_lengths = np.argsort(lengths)
    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(len(lengths)), diversities[argsort_lengths], label="Our Model")
    if df_pretrained is not None:
        pretrained_diversities = df_pretrained["diversity"].to_numpy()
        plt.plot(
            np.arange(len(lengths)),
            pretrained_diversities[argsort_lengths],
            label="Pretrained Model",
        )
    plt.xticks(np.arange(len(lengths)), lengths[argsort_lengths])
    plt.xlabel("Length")
    plt.ylabel("Diversity")
    plt.tight_layout()
    plt.legend(fontsize=16)
    plt.savefig(outdir / f"diversity{suffix}.png")


def hierarchy_diversity(
    data_path: pathlib.Path,
    outdir: pathlib.Path,
    length: int,
    tm_score_th: float = 0.5,
    pretrained: bool = False,
) -> float:
    """Compute diversity by hierarchy clustering.

    Args:
        data_path: path to inference results.
        outdir: output directory to save clustering results.
        length: length of sequence to be analysed.
        tm_score_th: TM-score threshold for clustering.
        pretrained: whether it's pre-trained model or our model.

    Returns:
        Computed diversity.
    """
    suffix = ""
    if pretrained:
        suffix = "_pretrained"
    directory = data_path / f"length_{length}"
    length = int(directory.stem.replace("length_", ""))
    outfile = outdir / f"pairwise_tm_score_length_{length}{suffix}.npy"
    if outfile.exists():
        pairwise_tm_scores = np.load(str(outfile))
    else:
        all_sample_feats = []
        all_sample_seq = []
        for gen_sample in directory.glob("sample_*"):
            sample_pdb = gen_sample / f"{gen_sample.stem}_1.pdb"
            sample_feats = data_utils.parse_pdb_feats(
                "sample", str(sample_pdb), chain_id=["A"]
            )["A"]
            all_sample_feats.append(sample_feats)
            sample_seq = data_utils.aatype_to_seq(sample_feats["aatype"])
            all_sample_seq.append(sample_seq)

        num = len(all_sample_feats)
        pairwise_tm_scores = np.ones((num, num))
        for i in range(num):
            for j in range(i + 1, num):
                _, tm_score = metrics.calc_tm_score(
                    all_sample_feats[i]["bb_positions"],
                    all_sample_feats[j]["bb_positions"],
                    all_sample_seq[i],
                    all_sample_seq[j],
                )
                pairwise_tm_scores[i, j] = pairwise_tm_scores[j, i] = tm_score

        np.save(str(outfile), pairwise_tm_scores)

    tree = hierarchy.linkage(
        distance.squareform(1 - pairwise_tm_scores, force="tovector"), method="ward"
    )
    clusters = hierarchy.fcluster(tree, t=1 - tm_score_th, criterion="distance")
    num_cluster = len(set(clusters))
    diversity = num_cluster / pairwise_tm_scores.shape[0]

    return diversity


def foldseek_search(
    sample_path: pathlib.Path,
    target_db: pathlib.Path | str,
) -> None:
    """Search for similar structures using foldseek.

    Args:
        sample_path: path to the sample PDB file.
        target_db: path to the target database to search.
    """
    if isinstance(target_db, pathlib.Path):
        target_db = str(target_db)

    foldseek_args = [
        "foldseek",
        "easy-search",
        str(sample_path),
        target_db,
        str(sample_path.parent / "align_score.csv"),
        "tmp",
        "--format-output",
        "query,target,alntmscore,u,t",
    ]

    logger.info(foldseek_args)
    with subprocess.Popen(
        foldseek_args, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT  # noqa: S603
    ) as process:
        _ = process.wait()


def plot_novelty(
    data_path: pathlib.Path,
    target_db: pathlib.Path | str,
    outdir: pathlib.Path,
    choice: str = "best",
    pretrained: bool = False,
    overwrite: bool = False,
) -> None:
    """Plot novelty.

    Figure will be saved under `outdir`.

    Args:
        data_path: path to inference results.
        target_db: path to target db to search.
        outdir: output directory to save plot.
        choice: choice of ESMFold samples, should be "best" or "median".
            Default to "best".
        pretrained: whether to use pre-trained model or our model.
        overwrite: whether to overwrite existing novelty results.
    """
    suffix = ""
    if pretrained:
        suffix = "_pretrained"
    res_csv_file = outdir / f"analyse_pdbtm{suffix}.csv"
    if not res_csv_file.exists() or overwrite:
        csv_file = outdir / f"analyse_rmsd_{choice}.csv"
        if not csv_file.exists():
            df = get_rmsd_df(data_path, output_file=csv_file, choice=choice)
        else:
            df = pd.read_csv(csv_file, sep="\t")

        pdbtms = []
        for _, row in df.iterrows():
            length = row["length"]
            sample = row["sample"]
            esmf_sample = row["esmf_sample"]
            esmf_sample_path = (
                data_path
                / f"length_{length}/{sample}"
                / f"self_consistency/esmf/esmf_sample_{esmf_sample}.pdb"
            )
            foldseek_search(esmf_sample_path, target_db)
            path_to_align_score = esmf_sample_path.parent / "align_score.csv"
            align_score_df = pd.read_csv(
                path_to_align_score,
                sep="\t",
                names=["query", "target", "alntmscore", "u", "t"],
            )
            pdbtm = align_score_df["alntmscore"].max()
            pdbtms.append(pdbtm)

        df["pdbTM"] = pdbtms
        df.to_csv(outdir / f"analyse_pdbtm{suffix}.csv", sep="\t", index=False)
    else:
        df = pd.read_csv(res_csv_file, sep="\t")

    rmsds = df["rmsd"].to_numpy()
    pdbtms = df["pdbTM"].to_numpy()
    lengths = df["length"].to_numpy()
    min_len: int = np.min(lengths).item()
    max_len: int = np.max(lengths).item()
    # Make a user-defined colormap.
    cmap = mcolor.LinearSegmentedColormap.from_list("redblue", ["b", "r"])
    cnorm = mcolor.Normalize(vmin=min_len, vmax=max_len)
    colors = np.array(
        [cmap((length - min_len) / (max_len - min_len)) for length in lengths]
    )

    plt.figure(figsize=(8, 6))
    plt.scatter(rmsds, pdbtms, c=colors, alpha=0.8)
    sm = plt.cm.ScalarMappable(norm=cnorm, cmap=cmap)
    plt.colorbar(sm)
    plt.xlabel("scRMSD")
    plt.ylabel("pdbTM")
    plt.tight_layout()
    plt.savefig(outdir / f"novelty{suffix}.png")


def plot_helix_sheet_percentage(
    data_path: pathlib.Path,
    outdir: pathlib.Path,
    pretrained: bool = False,
    overwrite: bool = False,
) -> None:
    """Plot Helix/Sheet percentage.

    Figure will be saved under `outdir`.

    Args:
        data_path: path to inference results.
        outdir: output directory to save plot.
        pretrained: whether to use pre-trained model or our model.
        overwrite: whether to overwrite existing Helix/Sheet results.
    """
    suffix = ""
    if pretrained:
        suffix = "_pretrained"
    res_csv_file = outdir / f"analyse_helix_sheet{suffix}.csv"
    if not res_csv_file.exists() or overwrite:
        lengths = []
        helix_percentage = []
        sheet_percentage = []
        for directory in data_path.glob("length_*"):
            length = int(directory.stem.replace("length_", ""))
            for gen_sample in directory.glob("sample_*"):
                sample_pdb = gen_sample / f"{gen_sample.stem}_1.pdb"
                protein_metrics = metrics.calc_mdtraj_metrics(sample_pdb)
                helix_percentage.append(protein_metrics["helix_percent"])
                sheet_percentage.append(protein_metrics["strand_percent"])
                lengths.append(length)

        df = pd.DataFrame(
            {
                "length": lengths,
                "helix_percent": helix_percentage,
                "sheet_percent": sheet_percentage,
            }
        )
        df.to_csv(res_csv_file, sep="\t", index=False)
    else:
        df = pd.read_csv(res_csv_file, sep="\t")
        lengths = df["length"].to_numpy()
        helix_percentage = df["helix_percent"].to_numpy()
        sheet_percentage = df["sheet_percent"].to_numpy()

    min_len: int = np.min(lengths).item()
    max_len: int = np.max(lengths).item()
    cmap = mcolor.LinearSegmentedColormap.from_list("redblue", ["b", "r"])
    cnorm = mcolor.Normalize(vmin=min_len, vmax=max_len)
    colors = np.array(
        [cmap((length - min_len) / (max_len - min_len)) for length in lengths]
    )

    plt.figure(figsize=(8, 6))
    plt.scatter(sheet_percentage, helix_percentage, c=colors, alpha=0.8)
    sm = plt.cm.ScalarMappable(norm=cnorm, cmap=cmap)
    plt.colorbar(sm)
    plt.xlabel("Sheet percentage")
    plt.ylabel("Helix percentage")
    plt.tight_layout()
    plt.savefig(outdir / f"helix_sheet{suffix}.png")


@hydra.main(version_base="1.3.1", config_path="../config", config_name="evaluation")
def run(cfg: omegaconf.DictConfig) -> None:
    res_path = pathlib.Path(cfg.inference_path)
    eval_output_path = pathlib.Path(cfg.eval_output_path)
    overwrite = cfg.overwrite

    pretrained_res_path = cfg.denovo.pretrained_inference_path
    pretrained_res_path = (
        pathlib.Path(pretrained_res_path) if pretrained_res_path is not None else None
    )
    esmfold_sample_choice = cfg.denovo.esmfold_sample_choice
    tmscore_th = cfg.denovo.diversity_tm_score_th
    novelty_target_db = cfg.denovo.novelty_target_db

    eval_output_path.mkdir(parents=True, exist_ok=True)

    plot_rmsd_tm_score(
        pretrained_data_path=pretrained_res_path,
        data_path=res_path,
        outdir=eval_output_path,
        choice=esmfold_sample_choice,
        overwrite=overwrite,
    )
    plot_diversity(
        pretrained_data_path=pretrained_res_path,
        data_path=res_path,
        outdir=eval_output_path,
        tm_score_th=tmscore_th,
        use_hierarchy=False,
        overwrite=overwrite,
    )
    plot_novelty(
        data_path=res_path,
        target_db=novelty_target_db,
        outdir=eval_output_path,
        choice=esmfold_sample_choice,
    )

    plot_helix_sheet_percentage(
        data_path=res_path,
        outdir=eval_output_path,
    )

    if pretrained_res_path is not None:
        plot_novelty(
            data_path=pretrained_res_path,
            target_db=novelty_target_db,
            outdir=eval_output_path,
            choice=esmfold_sample_choice,
            pretrained=True,
        )
        plot_helix_sheet_percentage(
            data_path=pretrained_res_path,
            outdir=eval_output_path,
            pretrained=True,
        )


if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter
