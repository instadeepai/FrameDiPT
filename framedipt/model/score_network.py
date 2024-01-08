"""Score network module."""
from __future__ import annotations

import functools as fn
import math

import torch
from omegaconf import DictConfig
from torch import nn

from framedipt.data import utils as data_utils
from framedipt.diffusion.se3_diffuser import SE3Diffuser
from framedipt.model import ipa_pytorch
from framedipt.protein import all_atom


def get_index_embedding(
    indices: torch.Tensor, embed_size: int, max_len: int = 2056
) -> torch.Tensor:
    """Creates sine / cosine positional embeddings from a prespecified indices.

    Args:
        indices: offsets of size [..., N_edges] of type integer
        max_len: maximum length.
        embed_size: dimension of the embeddings to create

    Returns:
        positional embedding of shape [N, embed_size]
    """
    k = torch.arange(embed_size // 2).to(indices.device)
    pos_embedding_sin = torch.sin(
        indices[..., None] * math.pi / (max_len ** (2 * k[None] / embed_size))
    ).to(indices.device)
    pos_embedding_cos = torch.cos(
        indices[..., None] * math.pi / (max_len ** (2 * k[None] / embed_size))
    ).to(indices.device)
    pos_embedding = torch.cat([pos_embedding_sin, pos_embedding_cos], dim=-1)
    return pos_embedding


def get_timestep_embedding(
    timesteps: torch.Tensor, embedding_dim: int, max_positions: int = 10000
) -> torch.Tensor:
    # Code from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
    if len(timesteps.shape) != 1:
        raise ValueError(f"timesteps should have 1D shape, got {timesteps.shape}.")

    timesteps = timesteps * max_positions
    half_dim = embedding_dim // 2
    emb_factor = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(
        torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
        * -emb_factor
    )
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = nn.functional.pad(emb, (0, 1), mode="constant")
    if emb.shape != (timesteps.shape[0], embedding_dim):
        raise ValueError(
            f"timestep embedding shape should be "
            f"{(timesteps.shape[0], embedding_dim)}, got {emb.shape}."
        )
    return emb


class Embedder(nn.Module):
    def __init__(self, model_conf: DictConfig, inpainting: bool = False) -> None:
        super().__init__()
        self._model_conf = model_conf
        self._embed_conf = model_conf.embed

        # Time step embedding
        index_embed_size = self._embed_conf.index_embed_size
        t_embed_size = index_embed_size
        node_embed_dims = t_embed_size + 1
        if inpainting or self._model_conf.input_aatype:
            node_embed_dims = node_embed_dims + 21
        edge_in = node_embed_dims * 2

        # Sequence index embedding
        node_embed_dims += index_embed_size
        edge_in += index_embed_size

        node_embed_size = self._model_conf.node_embed_size
        self.node_embedder = nn.Sequential(
            nn.Linear(node_embed_dims, node_embed_size),
            nn.ReLU(),
            nn.Linear(node_embed_size, node_embed_size),
            nn.ReLU(),
            nn.Linear(node_embed_size, node_embed_size),
            nn.LayerNorm(node_embed_size),
        )

        if self._embed_conf.embed_self_conditioning:
            edge_in += self._embed_conf.num_bins
        edge_embed_size = self._model_conf.edge_embed_size
        self.edge_embedder = nn.Sequential(
            nn.Linear(edge_in, edge_embed_size),
            nn.ReLU(),
            nn.Linear(edge_embed_size, edge_embed_size),
            nn.ReLU(),
            nn.Linear(edge_embed_size, edge_embed_size),
            nn.LayerNorm(edge_embed_size),
        )

        self.timestep_embedder = fn.partial(
            get_timestep_embedding, embedding_dim=self._embed_conf.index_embed_size
        )
        self.index_embedder = fn.partial(
            get_index_embedding, embed_size=self._embed_conf.index_embed_size
        )

    def _cross_concat(
        self, feats_1d: torch.Tensor, num_batch: int, num_res: int
    ) -> torch.Tensor:
        return (
            torch.cat(
                [
                    torch.tile(feats_1d[:, :, None, :], (1, 1, num_res, 1)),
                    torch.tile(feats_1d[:, None, :, :], (1, num_res, 1, 1)),
                ],
                dim=-1,
            )
            .float()
            .reshape([num_batch, num_res**2, -1])
        )

    def forward(
        self,
        *,
        seq_idx: torch.Tensor,
        t: torch.Tensor,
        fixed_mask: torch.Tensor,
        self_conditioning_ca: torch.Tensor,
        aatype: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Embeds a set of inputs

        Args:
            seq_idx: [..., N] Positional sequence index for each residue.
            t: Sampled t in [0, 1].
            fixed_mask: mask of fixed (motif) residues.
            self_conditioning_ca: [..., N, 3] Ca positions of self-conditioning
                input.

        Returns:
            node_embed: [B, N, D_node]
            edge_embed: [B, N, N, D_edge]
        """
        num_batch, num_res = seq_idx.shape

        fixed_mask = fixed_mask[..., None]

        prot_t_embed = torch.tile(
            self.timestep_embedder(t)[:, None, :], (1, num_res, 1)
        )

        if aatype is not None:
            aatype = torch.nn.functional.one_hot(aatype, num_classes=21)
            # Set time step to epsilon=1e-5 for fixed residues.
            eps_t_embed = torch.tile(
                self.timestep_embedder(torch.ones_like(t) * 1e-5)[:, None, :],
                (1, num_res, 1),
            )
            combined_t_embed = torch.where(
                fixed_mask.bool(),
                eps_t_embed,
                prot_t_embed,
            )
            prot_t_embed = torch.cat([aatype, combined_t_embed, fixed_mask], dim=-1)
        else:
            prot_t_embed = torch.cat([prot_t_embed, fixed_mask], dim=-1)

        node_feats = [prot_t_embed]
        pair_feats = [self._cross_concat(prot_t_embed, num_batch, num_res)]

        # Positional index features.
        node_feats.append(self.index_embedder(seq_idx))
        rel_seq_offset = seq_idx[:, :, None] - seq_idx[:, None, :]
        rel_seq_offset = rel_seq_offset.reshape([num_batch, num_res**2])
        pair_feats.append(self.index_embedder(rel_seq_offset))

        # Self-conditioning distogram.
        if self._embed_conf.embed_self_conditioning:
            sc_dgram = data_utils.calc_distogram(
                self_conditioning_ca,
                self._embed_conf.min_bin,
                self._embed_conf.max_bin,
                self._embed_conf.num_bins,
            )
            pair_feats.append(sc_dgram.reshape([num_batch, num_res**2, -1]))

        node_embed = self.node_embedder(torch.cat(node_feats, dim=-1).float())
        edge_embed = self.edge_embedder(torch.cat(pair_feats, dim=-1).float())
        edge_embed = edge_embed.reshape([num_batch, num_res, num_res, -1])
        return node_embed, edge_embed


class ScoreNetwork(nn.Module):
    def __init__(
        self, model_conf: DictConfig, diffuser: SE3Diffuser, inpainting: bool = False
    ) -> None:
        super().__init__()
        self._model_conf = model_conf

        self.embedding_layer = Embedder(model_conf, inpainting=inpainting)
        self.diffuser = diffuser
        self.score_model = ipa_pytorch.IpaScore(model_conf, diffuser)

        self.inpainting = inpainting

    def _apply_mask(
        self, aatype_diff: torch.Tensor, aatype_0: torch.Tensor, diff_mask: torch.Tensor
    ) -> torch.Tensor:
        return diff_mask * aatype_diff + (1 - diff_mask) * aatype_0

    def forward(self, input_feats: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward computes the reverse diffusion conditionals p(X^t|X^{t+1})
        for each item in the batch

        Args:
            X: the noised samples from the noising process, of shape [Batch, N, D].
                Where the T time steps are t=1,...,T
                 (i.e. not including the un-noised X^0)

        Returns:
            model_out: dictionary of model outputs.
        """

        # Frames as [batch, res, 7] tensors.
        bb_mask = input_feats["res_mask"].type(torch.float32)  # [B, N]
        fixed_mask = input_feats["fixed_mask"].type(torch.float32)
        edge_mask = bb_mask[..., None] * bb_mask[..., None, :]

        feat_aatype = None if "aatype" not in input_feats else input_feats["aatype"]
        aatype = data_utils.preprocess_aatype(
            aatype=feat_aatype,
            fixed_mask=fixed_mask,
            inpainting=self.inpainting,
            input_aatype=self._model_conf.input_aatype,
        )

        # Initial embeddings of positonal and relative indices.
        init_node_embed, init_edge_embed = self.embedding_layer(
            seq_idx=input_feats["seq_idx"],
            t=input_feats["t"],
            fixed_mask=fixed_mask,
            self_conditioning_ca=input_feats["sc_ca_t"],
            aatype=aatype,
        )
        edge_embed = init_edge_embed * edge_mask[..., None]
        node_embed = init_node_embed * bb_mask[..., None]

        # Run main network
        model_out = self.score_model(node_embed, edge_embed, input_feats)

        # Psi angle prediction
        gt_psi = input_feats["torsion_angles_sin_cos"][..., 2, :]
        psi_pred = self._apply_mask(model_out["psi"], gt_psi, 1 - fixed_mask[..., None])

        pred_out = {
            "psi": psi_pred,
            "rot_score": model_out["rot_score"],
            "trans_score": model_out["trans_score"],
        }
        rigids_pred = model_out["final_rigids"]
        pred_out["rigids"] = rigids_pred.to_tensor_7()
        bb_representations = all_atom.compute_backbone(
            rigids_pred, psi_pred, aatype=aatype
        )
        pred_out["atom37"] = bb_representations[0].to(rigids_pred.device)
        pred_out["atom14"] = bb_representations[-1].to(rigids_pred.device)

        return pred_out
