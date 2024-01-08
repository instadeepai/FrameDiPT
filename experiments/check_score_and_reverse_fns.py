"Check score and reverse functions actually agree."
from __future__ import annotations

import hydra
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
from matplotlib.figure import Figure

from framedipt.diffusion import se3_diffuser

NUM_CHAINS = 3
T = 1.0
dt = 0.01


def gen_line(size: float = 50.0, spacing: float = 2.0) -> np.ndarray:
    """Produce an array of equally spaced 3D points between two points on a sphere.

    Args:
        size: radius of sphere to sample points from
        spacing: approximate distance between adjacent points

    Returns:
        np.ndarray, (n,3) contains equally spaced points.
    """
    point1 = np.random.randn(1, 3)
    point1 = size * point1 / np.linalg.norm(point1, axis=-1, keepdims=True)
    point2 = np.random.randn(1, 3)
    point2 = size * point2 / np.linalg.norm(point2, axis=-1, keepdims=True)
    offset = point1 - point2
    offset_len = np.linalg.norm(offset)
    point_count = int(offset_len / spacing)
    weights = np.linspace(0, 1, point_count, endpoint=True)[..., None]
    line = point1 * (1 - weights) + point2 * weights
    return line


def plot_chains(
    x_t: np.ndarray,
    diffuse_mask: np.ndarray,
    chain_indices: np.ndarray,
    score: np.ndarray | None = None,
    fig: Figure | None = None,
) -> Figure:
    """Produce a figure plotting points from a diffusion process.

    Args:
        x_t: [n,3] 3D positions
        diffuse_mask: which points are being diffused,
            used to highlight diffused regions.
        chain_indices: per-point chain indices,
            used to differentiate chains from each other
        score: optional [n,3] score array,
            will draw corresponding vector arrows if present.
        fig: optional matplotlib figure object, if present will clear axes and replot

    Returns:
        figure: matplotlib figure object with plotted axes.
            Ready to be displayed with `fig.show()` or saved with `fig.savefig()`"""
    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
    else:
        ax = fig.axes[0]
    bool_mask = diffuse_mask.astype(bool)
    ax.cla()
    ax.scatter(
        x_t[bool_mask, 0],
        x_t[bool_mask, 1],
        x_t[bool_mask, 2],
        c=chain_indices[bool_mask],
    )
    ax.scatter(x_t[~bool_mask, 0], x_t[~bool_mask, 1], x_t[~bool_mask, 2], c="grey")
    if score is not None:
        ax.quiver(
            x_t[:, 0], x_t[:, 1], x_t[:, 2], score[:, 0], score[:, 1], score[:, 2]
        )
    return fig


@hydra.main(version_base="1.3.1", config_path="../config", config_name="base")
def run(cfg: omegaconf.DictConfig) -> None:
    diffuser = se3_diffuser.SE3Diffuser(cfg.diffuser)
    trans_diffuser = diffuser._r3_diffuser  # pylint:disable=protected-access
    chain_idx_list = []
    lines = []
    diffuse_mask_list = []
    for i in range(NUM_CHAINS):
        line = gen_line()
        lines.append(line)
        chain_idx = np.ones(line.shape[0], dtype=np.int32) * i
        diff_mask = np.zeros_like(chain_idx, dtype=np.float32)
        loop_start = line.shape[0] // 3
        loop_end = loop_start * 2
        diff_mask[loop_start:loop_end] = 1.0
        diffuse_mask_list.append(diff_mask)
        chain_idx_list.append(chain_idx)
    x_0 = np.concatenate(lines, axis=0)
    chain_indices = np.concatenate(chain_idx_list, axis=0)
    diffuse_mask = np.concatenate(diffuse_mask_list, axis=0)
    x_t, score_t = trans_diffuser.forward_marginal(
        x_0, T, diffuse_mask=diffuse_mask, chain_indices=chain_indices
    )

    t = T
    step = 0
    fig = plot_chains(x_t, diffuse_mask, chain_indices, score=None)
    fig.savefig(f"out/{cfg.diffuser.r3.diffuser}_{step:03}.png")
    while t > 0.0:  # noqa: PLR2004
        score_t = trans_diffuser.score(
            x_t,
            x_0,
            t=t,
            diffuse_mask=diffuse_mask,
            chain_indices=chain_indices,
            use_torch=False,
            scale=True,
        )
        x_t = trans_diffuser.reverse(
            x_t,
            score_t=score_t,
            t=t,
            dt=dt,
            diffuse_mask=diffuse_mask,
            chain_indices=chain_indices,
            center=False,
        )
        t -= dt
        step += 1
        fig = plot_chains(x_t, chain_indices, diffuse_mask, score=None, fig=fig)
        fig.savefig(f"out/{cfg.diffuser.r3.diffuser}_{step:03}.png")


if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter
