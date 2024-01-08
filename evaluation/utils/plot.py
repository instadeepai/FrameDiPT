"""Module for evaluation plots."""
from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


def boxplot_rmsd(
    rmsds: np.ndarray, eval_output_path: pathlib.Path, swarmplot: bool = True
) -> None:
    """Get RMSD boxplot.

    Args:
        rmsds: array of pre-computed RMSDs.
        eval_output_path: path to save the evaluation output plot.
        swarmplot: whether to produce swarmplots. Each data point is
            plotted as a point on the graph, enabling easy analysis of
            data distributions.
    """
    plt.figure(figsize=(8, 6))
    if swarmplot:
        sns.swarmplot(rmsds)

    plt.boxplot(
        rmsds,
        showfliers=False,
        positions=[0.0],
    )

    rmsd_median = np.median(rmsds)
    rmsd_abs_dev = stats.median_abs_deviation(rmsds)

    plt.xticks([0.0], ["Backbone RMSD"])
    plt.title(
        f"Backbone RMSD \nMedian {rmsd_median:.2f}$\\pm${rmsd_abs_dev:.2f}", fontsize=20
    )
    plt.tight_layout()
    plt.savefig(eval_output_path / "rmsd_median_boxplot.png")


def plot_pearsonr(
    rmsds: np.ndarray,
    seq_lengths: np.ndarray,
    legend: str,
    eval_output_path: pathlib.Path,
) -> None:
    """Plot Pearson correlation
        between RMSDs and sequence/diffused/coil lengths.

    Args:
        rmsds: array of pre-computed RMSDs.
        seq_lengths: array of sequence/diffused/coil lengths.
        legend: legend indication for the plot.
        eval_output_path: path to save the evaluation output plot.

    Raises:
        ValueError if lengths of rmsds and seq_lengths are not the same.
    """
    if len(rmsds) != len(seq_lengths):
        raise ValueError(
            f"Length of rmsds and seq_lengths should be the same, "
            f"got {len(rmsds)} != {len(seq_lengths)}."
        )

    pearson_res = stats.pearsonr(seq_lengths, rmsds)
    corr, pvalue = pearson_res

    plt.figure(figsize=(8, 6))
    plt.scatter(
        seq_lengths,
        rmsds,
        label=f"RMSD vs {legend}\nPearson correlation={corr:.3f}\npvalue={pvalue:.3f}",
    )
    plt.xlabel(legend)
    plt.ylabel("Backbone RMSD")
    plt.tight_layout()
    plt.legend(fontsize=16)
    file_legend = legend.lower().replace(" ", "_")
    plt.savefig(eval_output_path / f"rmsd_median_vs_{file_legend}.png")


def boxplot_metrics_alpha_beta(
    metrics_alpha: list[np.ndarray],
    metrics_beta: list[np.ndarray],
    eval_output_path: pathlib.Path,
    legend: str,
    xticks: list[str],
) -> None:
    """Get boxplot of a list of RMSDs.

    Args:
        metrics_alpha: list of arrays of pre-computed metrics for alpha chain.
        metrics_beta: list of arrays of pre-computed metrics for beta chain.
        eval_output_path: path to save the evaluation output plot.
        legend: legend of the plot.
        xticks: list of ticks for x-axis.

    Raises:
        ValueError if length of xticks is not the same as
            the maximum length of metrics_alpha and metrics_beta.
    """
    xs_len = max(len(metrics_alpha), len(metrics_beta))
    if len(xticks) != xs_len:
        raise ValueError(
            f"Length of xticks should be the same as "
            f"the maximum length of metrics_alpha and metrics_beta, "
            f"got {len(xticks)} != {xs_len}."
        )

    plt.figure(figsize=(8, 6))

    xs = np.arange(xs_len) + 1
    boxplot_1 = plt.boxplot(
        metrics_alpha,
        showfliers=False,
        patch_artist=True,
        widths=0.3,
        positions=xs[: len(metrics_alpha)] - 0.2,
    )
    boxplot_2 = plt.boxplot(
        metrics_beta,
        showfliers=False,
        patch_artist=True,
        widths=0.3,
        positions=xs[: len(metrics_beta)] + 0.2,
    )

    for patch in boxplot_1["boxes"]:
        patch.set_facecolor("royalblue")
    for patch in boxplot_2["boxes"]:
        patch.set_facecolor("orange")

    metrics_alpha_median = np.median(np.concatenate(metrics_alpha))
    metrics_alpha_abs_dev = stats.median_abs_deviation(np.concatenate(metrics_alpha))
    metrics_beta_median = np.median(np.concatenate(metrics_beta))
    metrics_beta_abs_dev = stats.median_abs_deviation(np.concatenate(metrics_beta))

    plt.xticks(xs, xticks)
    plt.title(
        f"{legend}\n"
        f"alpha Median {metrics_alpha_median:.2f}$\\pm${metrics_alpha_abs_dev:.2f}\n"
        f"beta Median {metrics_beta_median:.2f}$\\pm${metrics_beta_abs_dev:.2f}",
        fontsize=20,
    )
    plt.tight_layout()
    plt.legend([boxplot_1["boxes"][0], boxplot_2["boxes"][0]], ["alpha", "beta"])
    filename = legend.lower().replace(" ", "_")
    plt.savefig(eval_output_path / f"{filename}_median_boxplot.png")


def transform_metrics_alpha_beta_to_dataframe_for_swarmplot(
    metrics_alpha: list[np.ndarray],
    metrics_beta: list[np.ndarray],
    legend: str,
    xticks: list[str],
) -> pd.DataFrame:
    """Transform list of alpha/beta chain metrics to dataframe.

    Seaborn swarmplot API relies on pandas dataframe inputs,
        data conversion simplifies plotting.

    Args:
        metrics_alpha: list of arrays of pre-computed metrics for alpha chain.
        metrics_beta: list of arrays of pre-computed metrics for beta chain.
        legend: legend of the plot.
        xticks: list of ticks for x-axis.

    Returns:
        Dataframe of 3 columns:
            - Residue index: defined by xticks.
            - Chain: alpha or beta.
            - `legend`: the legend for the plot.
    """
    df_dict: dict[str, list] = {
        "Residue index": [],
        "Chain": [],
        legend: [],
    }
    for xtick, metric_alpha in zip(xticks, metrics_alpha):
        num_samples = metric_alpha.shape[0]
        df_dict["Residue index"] += [xtick] * num_samples
        df_dict["Chain"] += ["alpha"] * num_samples
        df_dict[legend] += list(metric_alpha)

    for xtick, metric_beta in zip(xticks, metrics_beta):
        num_samples = metric_beta.shape[0]
        df_dict["Residue index"] += [xtick] * num_samples
        df_dict["Chain"] += ["beta"] * num_samples
        df_dict[legend] += list(metric_beta)

    df = pd.DataFrame(df_dict)

    return df


def swarmplot_metrics_alpha_beta(
    metrics_alpha: list[np.ndarray],
    metrics_beta: list[np.ndarray],
    eval_output_path: pathlib.Path,
    legend: str,
    xticks: list[str],
    superpose_box: bool = False,
) -> None:
    """Get boxplot of a list of RMSDs.

    Args:
        metrics_alpha: list of arrays of pre-computed metrics for alpha chain.
        metrics_beta: list of arrays of pre-computed metrics for beta chain.
        eval_output_path: path to save the evaluation output plot.
        legend: legend of the plot.
        xticks: list of ticks for x-axis.
        superpose_box: whether to superpose boxplot.

    Raises:
        ValueError if length of xticks is not the same as
            the maximum length of metrics_alpha and metrics_beta.
    """
    xs_len = max(len(metrics_alpha), len(metrics_beta))
    if len(xticks) != xs_len:
        raise ValueError(
            f"Length of xticks should be the same as "
            f"the maximum length of metrics_alpha and metrics_beta, "
            f"got {len(xticks)} != {xs_len}."
        )

    df = transform_metrics_alpha_beta_to_dataframe_for_swarmplot(
        metrics_alpha=metrics_alpha,
        metrics_beta=metrics_beta,
        legend=legend,
        xticks=xticks,
    )

    plt.figure(figsize=(8, 6))

    sns.swarmplot(
        data=df,
        x="Residue index",
        y=legend,
        hue="Chain",
        dodge=True,
    )

    xs = np.arange(xs_len)
    if superpose_box:
        plt.boxplot(
            metrics_alpha,
            showfliers=False,
            widths=0.2,
            positions=xs[: len(metrics_alpha)] - 0.2,
        )
        plt.boxplot(
            metrics_beta,
            showfliers=False,
            widths=0.2,
            positions=xs[: len(metrics_beta)] + 0.2,
        )
        plt.xlabel(None)

    metrics_alpha_median = np.median(np.concatenate(metrics_alpha))
    metrics_alpha_abs_dev = stats.median_abs_deviation(np.concatenate(metrics_alpha))
    metrics_beta_median = np.median(np.concatenate(metrics_beta))
    metrics_beta_abs_dev = stats.median_abs_deviation(np.concatenate(metrics_beta))

    plt.title(
        f"{legend}\n"
        f"alpha Median {metrics_alpha_median:.2f}$\\pm${metrics_alpha_abs_dev:.2f}\n"
        f"beta Median {metrics_beta_median:.2f}$\\pm${metrics_beta_abs_dev:.2f}",
        fontsize=20,
    )
    plt.xticks(xs, xticks)
    plt.tight_layout()
    filename = legend.lower().replace(" ", "_")
    plt.savefig(eval_output_path / f"{filename}_median_swarmplot.png")


def two_models_scatter_plot(
    df_metrics: pd.DataFrame,
    df_esmfold_metrics: pd.DataFrame,
    eval_output_path: pathlib.Path,
    choice: str = "median",
) -> None:
    """Scatter plot between two models' metrics.

    Args:
        df_metrics: dataframe containing evaluation metrics.
        df_esmfold_metrics: dataframe containing evaluation metrics
            for ESMFold (or AlphaFold).
        eval_output_path: path to save the evaluation output plot.
        choice: select "median" or "best" sample to do evaluation.

    Raises:
        ValueError if choice is neither "best", nor "median".
    """
    if choice == "median":
        df_selected_rmsd_grouped_by_pdb_name = df_metrics.groupby("pdb_name")[
            "bb_rmsd"
        ].median()
    elif choice == "best":
        df_selected_rmsd_grouped_by_pdb_name = df_metrics.groupby("pdb_name")[
            "bb_rmsd"
        ].min()
    else:
        raise ValueError(f"Choice need to be median or best, got {choice}.")

    df_analyse = pd.merge(
        df_metrics,
        df_selected_rmsd_grouped_by_pdb_name,
        how="inner",
        on=["pdb_name", "bb_rmsd"],
    )

    framedipt_bb_rmsds_alpha = df_analyse["bb_rmsd_alpha"].to_numpy()
    framedipt_bb_rmsds_beta = df_analyse["bb_rmsd_beta"].to_numpy()

    esmfold_bb_rmsds_alpha = df_esmfold_metrics["bb_rmsd_alpha"].to_numpy()
    esmfold_bb_rmsds_beta = df_esmfold_metrics["bb_rmsd_beta"].to_numpy()

    xs = np.linspace(0, 10, 100)

    plt.figure(figsize=(6, 6))

    plt.plot(xs, xs, color="black", linestyle="dashed")
    plt.scatter(esmfold_bb_rmsds_alpha, framedipt_bb_rmsds_alpha, label="alpha")
    plt.scatter(esmfold_bb_rmsds_beta, framedipt_bb_rmsds_beta, label="beta")

    plt.xlim([0, 10])
    plt.xlabel("ESMFold backbone RMSD", fontsize=16)
    plt.ylabel("FrameDiPT backbone RMSD", fontsize=16)
    plt.title("Backbone RMSD", fontsize=20)
    plt.legend(fontsize=16)

    plt.savefig(eval_output_path / "bb_rmsd_framedipt_esmfold_scatter.png")


def boxplot_tcr_rmsd(
    df_bound: pd.DataFrame,
    df_unbound: pd.DataFrame,
    eval_output_path: pathlib.Path,
    choice: str = "median",
) -> None:
    """Plot unbound and bound TCR RMSD results.

    Args:
        df_bound: dataframe containing bound TCR evaluation metrics.
        df_unbound: dataframe containing unbound TCR evaluation metrics.
        eval_output_path: path to save the evaluation output plot.
        choice: select "median" or "best" sample to do evaluation.
            Defaults to "median".

    Raises:
        ValueError if choice is neither "best", nor "median".
    """

    def _calc_median_and_stddev(values: np.ndarray) -> tuple[float, float]:
        median = np.median(values).item()
        stddev = stats.median_abs_deviation(values)
        return median, stddev

    if choice == "median":
        df_bound_groupby = (
            df_bound.groupby("pdb_name", sort=False)["bb_rmsd"].median().reset_index()
        )
        df_unbound_groupby = (
            df_unbound.groupby("pdb_name", sort=False)["bb_rmsd"].median().reset_index()
        )
    elif choice == "best":
        df_bound_groupby = (
            df_bound.groupby("pdb_name", sort=False)["bb_rmsd"].min().reset_index()
        )
        df_unbound_groupby = (
            df_unbound.groupby("pdb_name", sort=False)["bb_rmsd"].min().reset_index()
        )
    else:
        raise ValueError(f"Choice need to be median or best, got {choice}.")

    # Merge dataframes to row with sample of choice
    df_bound = df_bound.merge(df_bound_groupby, on=["pdb_name", "bb_rmsd"])
    df_unbound = df_unbound.merge(df_unbound_groupby, on=["pdb_name", "bb_rmsd"])

    # Calc median and standrad deviations
    metrics_bound_full_atom = _calc_median_and_stddev(df_bound["full_atom_rmsd"])
    metrics_bound_bb = _calc_median_and_stddev(df_bound["bb_rmsd"])
    metrics_unbound_full_atom = _calc_median_and_stddev(df_unbound["full_atom_rmsd"])
    metrics_unbound_bb = _calc_median_and_stddev(df_unbound["bb_rmsd"])

    plt.figure(figsize=(8, 6))
    # Plot bound rmsd
    boxplot_bound = plt.boxplot(
        df_bound[["full_atom_rmsd", "bb_rmsd"]].to_numpy(),
        showfliers=False,
        patch_artist=True,
        widths=0.3,
        medianprops={"color": "black"},
        positions=[1, 2],
    )
    # Plot unbound rmsd
    boxplot_unbound = plt.boxplot(
        df_unbound[["full_atom_rmsd", "bb_rmsd"]].to_numpy(),
        showfliers=False,
        patch_artist=True,
        widths=0.3,
        medianprops={"color": "black"},
        positions=[4, 5],
    )

    # Set colors of the boxplots
    boxplot_bound["boxes"][0].set_facecolor("royalblue")
    boxplot_bound["boxes"][1].set_facecolor("orange")
    boxplot_unbound["boxes"][0].set_facecolor("royalblue")
    boxplot_unbound["boxes"][1].set_facecolor("orange")

    #  Place median and std dev text
    plt.text(
        0.5,
        6.6,
        "bound full atom median "
        f"{metrics_bound_full_atom[0]:.2f}$\\pm${metrics_bound_full_atom[1]:.2f}\n"
        "bound backbone median "
        f"{metrics_bound_bb[0]:.2f}$\\pm${metrics_bound_bb[1]:.2f}",
    )
    plt.text(
        3.5,
        6.6,
        "unbound full atom median "
        f"{metrics_unbound_full_atom[0]:.2f}$\\pm${metrics_unbound_full_atom[1]:.2f}\n"
        "unbound backbone median "
        f"{metrics_unbound_bb[0]:.2f}$\\pm${metrics_unbound_bb[1]:.2f}",
    )

    plt.xticks((1.5, 4.5), ["Bound", "Unbound"])
    plt.xlabel("TCR Type")
    plt.ylabel("RMSD")
    plt.title("TCR CDR3 RMSD\n\n")
    plt.legend(
        [boxplot_bound["boxes"][0], boxplot_bound["boxes"][1]],
        ["full atom", "backbone"],
    )
    plt.tight_layout()
    plt.savefig(eval_output_path / f"tcr_rmsd_{choice}_boxplot.png")
