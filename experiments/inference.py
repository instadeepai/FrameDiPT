"""Script for running inference and sampling.

Sample command:
> python experiments/inference.py

"""
from __future__ import annotations

import datetime
import os
import pathlib
import shutil
import subprocess
import time
from typing import Any

import esm
import hydra
import numpy as np
import omegaconf
import pandas as pd
import torch
import tree
from biotite.sequence.io import fasta

from experiments import sampler
from experiments import utils as exp_utils
from experiments.utils import logp_confidence_score
from framedipt.analysis import metrics
from framedipt.analysis import utils as analysis_utils
from framedipt.data import utils as data_utils
from framedipt.diffusion import se3_diffuser
from framedipt.model import score_network
from framedipt.protein import tcr
from framedipt.tools.errors import ProteinMPNNError
from framedipt.tools.log import get_logger
from openfold.utils import rigid_utils

logger = get_logger()


class Inference:
    """Inference class."""

    def __init__(
        self,
        cfg: omegaconf.DictConfig,
        cfg_overrides: dict | None = None,
    ) -> None:
        """Initialize inference.

        Args:
            cfg: inference config.
            cfg_overrides: Dict of fields to override with new values.
        """

        # Allow you to write to unknown fields.
        omegaconf.OmegaConf.set_struct(cfg, False)

        # Prepare configs.
        self._cfg = cfg
        self._cfg.model.inpainting = self._cfg.inference.inpainting
        self._cfg.model.input_aatype = self._cfg.inference.input_aatype

        self._rng = np.random.default_rng(self._cfg.inference.seed)

        _, self.gpu_id, self.device, _ = exp_utils.get_devices(
            self._cfg.inference.use_gpu
        )

        # Set up output directory
        output_dir = pathlib.Path(self._cfg.inference.output_dir)
        if self._cfg.inference.name is None:
            dt_string = datetime.datetime.now(tz=datetime.timezone.utc).strftime(
                "%dD_%mM_%YY_%Hh_%Mm_%Ss"
            )
        else:
            dt_string = self._cfg.inference.name
        self.output_dir = output_dir / dt_string
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving results to {self.output_dir}")

        # Load models and create sampler dataset.
        self._load_ckpt(self._cfg.inference.weights_path, cfg_overrides)
        self.create_dataset()

        # Save config after loading checkpoint.
        config_path = self.output_dir / "inference_conf.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            omegaconf.OmegaConf.save(config=self._cfg, f=f)
        logger.info(f"Saving inference config to {config_path}")

        # Set model hub directory for ESMFold and load EMSFold model.
        if (
            not self._cfg.inference.inpainting
            or self._cfg.inference.inpainting_samples.run_esmfold
        ):
            torch.hub.set_dir(self._cfg.inference.pt_hub_dir)
            self._folding_model = esm.pretrained.esmfold_v1().eval()
            if not torch.cuda.is_available():
                # disable mixed precision if gpu not available
                self._folding_model.esm.float()
            self._folding_model = self._folding_model.to(self.device)

        # Set ProteinMPNN directory
        self._pmpnn_dir = self._cfg.inference.pmpnn_dir

    def _load_ckpt(
        self, weights_path: pathlib.Path | str, conf_overrides: dict | None
    ) -> None:
        """Loads in model checkpoint.

        Args:
            weights_path: path to the checkpoint.
            conf_overrides: optional dictionary of configs to override.

        Raises:
            ValueError if the merged config by conf_overrides is not
                an object of omegaconf.DictConfig.
        """
        if isinstance(weights_path, str):
            weights_path = pathlib.Path(weights_path)

        logger.info(f"Loading weights from {weights_path}.")

        # Read checkpoint.
        weights_pkl = data_utils.read_pkl(
            weights_path, use_torch=True, map_location=self.device
        )

        # Use checkpoint config.
        self._cfg.model = omegaconf.OmegaConf.merge(
            self._cfg.model, weights_pkl["conf"].model
        )
        # handle current and previous ways of specifying diffuser config
        # Load all options.
        self._cfg.diffuser.r3 = weights_pkl["conf"].diffuser.r3
        if conf_overrides is not None:
            new_conf = omegaconf.OmegaConf.merge(self._cfg, conf_overrides)
            if not isinstance(new_conf, omegaconf.DictConfig):
                raise ValueError(
                    f"The merged config should be DictConfig, got {type(new_conf)}."
                )
            self._cfg = new_conf

        # Fix the seed for the diffuser.
        self._cfg.diffuser.so3.seed = self._cfg.inference.seed
        self._cfg.diffuser.r3.seed = self._cfg.inference.seed
        # Create diffuser and model.
        self.diffuser = se3_diffuser.SE3Diffuser(self._cfg.diffuser)
        self.model = score_network.ScoreNetwork(
            self._cfg.model,
            self.diffuser,
            inpainting=self._cfg.inference.inpainting,
        )

        # Remove module prefix if it exists.
        model_weights = weights_pkl["model"]
        model_weights = {k.replace("module.", ""): v for k, v in model_weights.items()}
        self.model.load_state_dict(model_weights)
        self.model = self.model.to(self.device)
        self.model.eval()

    def create_dataset(self) -> None:
        """Create unconditional sampler dataset."""
        if self._cfg.inference.inpainting:
            if self._cfg.inference.inpainting_samples.tcr:
                self.sampler = sampler.TCRSampler(
                    data_conf=self._cfg.inference.inpainting_samples,
                    diffuser=self.diffuser,
                    device=self.device,
                )
            else:
                self.sampler = sampler.ConditionalSampler(
                    data_conf=self._cfg.inference.inpainting_samples,
                    diffuser=self.diffuser,
                    device=self.device,
                )
        else:
            self.sampler = sampler.UnconditionalSampler(
                cfg=self._cfg.inference.samples,
                diffuser=self.diffuser,
                device=self.device,
            )

    def run_sampling(self) -> None:
        """Run inference sampling according to inpainting setting."""
        if self._cfg.inference.inpainting:
            self.run_conditional_sampling()
        else:
            self.run_unconditional_sampling()

    def run_unconditional_sampling(self) -> None:
        """Sets up unconditional inference run.

        All outputs are written to {output_dir}/{date_time}
            where {output_dir} is created at initialization.
        """
        for sample_length, sample_i, sample_feats in self.sampler:
            # Set up length directory, top 1 level
            length_dir = self.output_dir / f"length_{sample_length}"
            length_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Sampling length {sample_length}: {length_dir}")

            # Set up sample directory, top 2 level
            sample_dir = length_dir / f"sample_{sample_i}"
            if sample_dir.exists():
                continue
            sample_dir.mkdir(parents=True, exist_ok=True)

            # Run inference
            sample_output = exp_utils.inference_fn(
                model=self.model,
                diffuser=self.diffuser,
                data_init=sample_feats,
                num_t=self._cfg.inference.diffusion.num_t,
                min_t=self._cfg.inference.diffusion.min_t,
                noise_scale=self._cfg.inference.diffusion.noise_scale,
                aux_traj=True,
                embed_self_conditioning=self._cfg.model.embed.embed_self_conditioning,
                inpainting=self._cfg.inference.inpainting,
                input_aatype=self._cfg.inference.input_aatype,
            )
            sample_output = tree.map_structure(lambda x: x[:, 0], sample_output)

            # Save inference trajectory
            traj_paths = self.save_traj(
                sample_output["prot_traj"],
                sample_output["rigid_0_traj"],
                np.ones(sample_length),
                output_dir=sample_dir,
                sample_idx=sample_i,
            )

            # Run ProteinMPNN
            pdb_path = traj_paths["sample_path"]
            sc_output_dir = sample_dir / "self_consistency"
            sc_output_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(pdb_path, sc_output_dir / pdb_path.name)

            # Run ESMFold for self-consistency
            self.run_self_consistency(sc_output_dir, pdb_path, motif_mask=None)
            logger.info(f"Done sample {sample_i}: {pdb_path}")

    def run_conditional_sampling(self) -> None:
        """Sets up conditional inference run for inpainting.

        All outputs are written to {output_dir}/{date_time}
            where {output_dir} is created at initialization.
        """
        for pdb_name, sample_i, sample_feats in self.sampler:
            gt_prot = exp_utils.get_atom_positions_from_rigids(
                rigids=rigid_utils.Rigid.from_tensor_7(sample_feats["rigids_0"]),
                psi_torsions=sample_feats["torsion_angles_sin_cos"][..., 2, :],
                aatype=sample_feats["aatype"],
            )

            res_mask = data_utils.move_to_np(sample_feats["res_mask"].bool())
            fixed_mask = data_utils.move_to_np(sample_feats["fixed_mask"].bool())
            res_mask_sum: np.ndarray = np.sum(res_mask)
            num_res = res_mask_sum.item()
            diffused_length = num_res - np.sum(fixed_mask * res_mask)
            diffused_mask = (1 - fixed_mask) * res_mask
            aatype = data_utils.move_to_np(sample_feats["aatype"])
            seq = data_utils.aatype_to_seq(aatype[res_mask])
            residue_index = data_utils.move_to_np(sample_feats["residue_index"])
            chain_index = data_utils.move_to_np(sample_feats["chain_idx"])

            # Set up length directory, top 1 level
            length_dir = self.output_dir / f"{pdb_name}_length_{diffused_length}"
            length_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Sampling length {diffused_length}: {length_dir}")

            # Save ground truth to pdb file.
            gt_pdb_path = length_dir / f"{pdb_name}_1.pdb"
            if not gt_pdb_path.exists():
                # Use b-factors to specify which residues are diffused.
                b_factors = np.tile(
                    (diffused_mask.astype(bool) * 100)[..., None], (1, 1, 37)
                )
                analysis_utils.write_prot_to_pdb(
                    prot_pos=gt_prot[res_mask],
                    file_path=length_dir / pdb_name,
                    aatype=aatype[res_mask],
                    b_factors=b_factors[res_mask],
                    residue_index=residue_index[res_mask],
                    chain_index=chain_index[res_mask],
                )

            if self._cfg.inference.inpainting_samples.run_esmfold:
                esmfold_pred_path = length_dir / "esmf_pred.pdb"
                if not esmfold_pred_path.exists():
                    # Save ESMFold prediction to pdb file
                    self.save_esmfold_prediction_to_pdb(
                        seq=seq,
                        chain_index=chain_index[res_mask],
                        output_dir=length_dir,
                    )
                    logger.info(f"ESMFold prediction saved to {esmfold_pred_path}")
                else:
                    logger.info(
                        f"ESMFold prediction already exists: {esmfold_pred_path}. "
                        f"Skip ESMFold inference."
                    )

            # Save diffusion info to csv file.
            diffusion_info_path = length_dir / "diffusion_info.csv"
            if not diffusion_info_path.exists():
                exp_utils.save_diffusion_info(
                    output_dir=length_dir,
                    pdb_name=pdb_name,
                    seq=seq,
                    diffused_mask=diffused_mask[res_mask],
                    chain_index=chain_index[res_mask],
                )

            # Set up sample directory, top 2 level
            sample_dir = length_dir / f"sample_{sample_i}"
            sample_pdb = sample_dir / f"sample_{sample_i}_1.pdb"
            if sample_pdb.exists():
                continue
            sample_dir.mkdir(parents=True, exist_ok=True)

            # Run inference
            sample_output = exp_utils.inference_fn(
                model=self.model,
                diffuser=self.diffuser,
                data_init=sample_feats,
                num_t=self._cfg.inference.diffusion.num_t,
                min_t=self._cfg.inference.diffusion.min_t,
                noise_scale=self._cfg.inference.diffusion.noise_scale,
                aux_traj=True,
                embed_self_conditioning=self._cfg.model.embed.embed_self_conditioning,
                inpainting=self._cfg.inference.inpainting,
                input_aatype=self._cfg.inference.input_aatype,
            )
            sample_output = tree.map_structure(lambda x: x[:, 0], sample_output)

            if self._cfg.inference.confidence_score == "eigenfold":
                # Get the final rigid-frame model prediction at t=0, shape (N_res, 7).
                # First 4 elements are rotations in quaternions and the last
                # 3 translations.
                rigids_0 = sample_output["rigid_traj"][0]

                # Convert from quaternion to axis-angle format, shape = (N_res, 6).
                # Where the first 3 elements are rotations and the last 3 translations.
                rigids_0 = rigid_utils.Rigid.from_tensor_7(torch.tensor(rigids_0))

                log_p, log_probs = logp_confidence_score(
                    model=self.model,
                    diffuser=self.diffuser,
                    sample_feats=sample_feats,
                    rigids_t=rigids_0,
                    diffuse_mask=diffused_mask[0],
                    self_condition=self._cfg.model.embed.embed_self_conditioning,
                    num_t=self._cfg.inference.diffusion.num_t,
                    min_t=self._cfg.inference.diffusion.min_t,
                    device=self.device,
                )
                diffused_region_len = diffused_mask.sum()
                log_p_per_residue = log_p / diffused_region_len
                log_p_per_residue_norm = log_p / (6 * diffused_region_len - 1)

                # Write the log_p score to diffusion_info.csv
                log_p_dict = {
                    f"log_p_sample_{sample_i}": log_p,
                    f"log_p_sample_{sample_i}_per_residue": log_p_per_residue,
                    f"log_p_sample_{sample_i}_per_residue_norm": log_p_per_residue_norm,
                }
                pd.read_csv(diffusion_info_path, sep="\t").assign(**log_p_dict).to_csv(
                    diffusion_info_path, sep="\t"
                )
                pd.DataFrame({"log_probs": log_probs}).to_csv(
                    sample_dir / "log_probs.csv"
                )

            # Save inference trajectory
            traj_paths = self.save_traj(
                sample_output["prot_traj"][:, res_mask[0]],
                sample_output["rigid_0_traj"][:, res_mask[0]],
                diffused_mask[res_mask],
                output_dir=sample_dir,
                sample_idx=sample_i,
                aatype=aatype[res_mask],
                residue_index=residue_index[res_mask],
                chain_index=chain_index[res_mask],
            )

            pdb_path = traj_paths["sample_path"]
            logger.info(f"Done sample {sample_i}: {pdb_path}")

    def save_ground_truth_to_pdb(
        self,
        pdb_name: str,
        diffuse_mask: np.ndarray,
        aatype: np.ndarray,
        residue_index: np.ndarray,
        chain_index: np.ndarray,
        output_dir: pathlib.Path,
    ) -> None:
        """Save ground truth structure to PDB file.

        It will be saved to file `output_dir/{pdb_name}_1.pdb`.

        Args:
            pdb_name: name of the PDB.
            diffuse_mask: mask of residues being diffused, shape [N_res,].
            aatype: AA type, shape [N_res,].
            residue_index: residue indices, shape [N_res,].
            chain_index: chain indices, shape [N_res,].
            output_dir: directory to save the PDB file.

        Raises:
            RuntimeError
                if 0 or multiple processed pickle file exists under `download_dir`.
        """
        # Use b-factors to specify which residues are diffused.
        b_factors = np.tile((diffuse_mask.astype(bool) * 100)[:, None], (1, 37))

        pdb_path = output_dir / pdb_name
        processed_pdb_path_gen = pathlib.Path(
            self._cfg.inference.inpainting_samples.download_dir
        ).rglob(f"*{pdb_name}.pkl")
        processed_pdb_path_list = list(processed_pdb_path_gen)
        if len(processed_pdb_path_list) != 1:
            raise RuntimeError(
                f"Only one processed pickle file should exist "
                f"under {self._cfg.inference.inpainting_samples.download_dir}, "
                f"got {len(processed_pdb_path_list)}."
            )
        processed_pdb_path = processed_pdb_path_list[0]
        processed_pdb_feats = data_utils.read_pkl(processed_pdb_path)

        bb_mask = processed_pdb_feats["bb_mask"].astype(bool)
        _ = analysis_utils.write_prot_to_pdb(
            processed_pdb_feats["atom_positions"][bb_mask],
            pdb_path,
            aatype=aatype,
            b_factors=b_factors,
            residue_index=residue_index,
            chain_index=chain_index,
        )

    def save_esmfold_prediction_to_pdb(
        self,
        seq: str,
        chain_index: np.ndarray,
        output_dir: pathlib.Path,
    ) -> None:
        """Save ESMFold prediction to PDB file.

        ESMFold takes `:`-separated AA sequence as input for multi-mers,
            so for multimers, we need to do the pre-processing
            of adding `:` between different chains.

        Particularly, for TCR complexes, as we may get long chains,
            ESMFold could raise out-of-memory error. In order to handle this,
            we truncate TCR chains to the variable domain only.

        Args:
            seq: input sequence. Multi-mers will be processed
                so that the sequence is separated by `:`.
            chain_index: chain indexes, shape [N_res,].
            output_dir: directory to save the PDB file.
        """
        unique_chain_indexes = np.unique(chain_index)
        seq_array = np.array(list(seq))
        seqs = []
        for cid, index in enumerate(unique_chain_indexes):
            partial_seq = seq_array[chain_index == index]
            partial_seq = "".join(partial_seq)
            # If it's TCRSampler and the current chain is TCR alpha or beta chain,
            # truncate TCR chains to solve out-of-memory error.
            if isinstance(self.sampler, sampler.TCRSampler) and cid < 2:
                partial_seq = tcr.cut_tcr_sequence(partial_seq)
            seqs.append(partial_seq)
        esmfold_seq = ":".join(seqs)
        esmf_sample_path = output_dir / "esmf_pred.pdb"
        _ = self.run_folding(esmfold_seq, esmf_sample_path)

    def save_traj(
        self,
        bb_prot_traj: np.ndarray,
        x0_traj: np.ndarray,
        diffuse_mask: np.ndarray,
        output_dir: pathlib.Path | str,
        sample_idx: int,
        aatype: np.ndarray | None = None,
        residue_index: np.ndarray | None = None,
        chain_index: np.ndarray | None = None,
    ) -> dict[str, pathlib.Path]:
        """Writes final sample and reverse diffusion trajectory.

        Args:
            bb_prot_traj: [T, N, 37, 3] atom37 sampled diffusion states.
                T is number of time steps. First time step is t=eps,
                i.e. bb_prot_traj[0] is the final sample after reverse diffusion.
                N is number of residues.
            x0_traj: [T, N, 3] x_0 predictions of C-alpha at each time step.
            diffuse_mask: [N] which residues are diffused.
            output_dir: where to save samples.
            sample_idx: index of generated samples for the test case.
            aatype: AA type, used in case of inpainting.
            residue_index: residue indices, used in case of inpainting.
            chain_index: chain indices, used in case of inpainting.

        Returns:
            Dictionary with paths to saved samples:
                - "sample_path": PDB file of final state of reverse trajectory.
                - "traj_path": PDB file of all intermediate diffused states.
                - "x0_traj_path": PDB file of C-alpha x_0 predictions at each state.
            b_factors are set to 100 for diffused residues and 0 for motif
                residues if there are any.
        """
        if isinstance(output_dir, str):
            output_dir = pathlib.Path(output_dir)

        # Write sample.
        diffuse_mask = diffuse_mask.astype(bool)
        sample_path = output_dir / f"sample_{sample_idx}"
        prot_traj_path = output_dir / f"bb_traj_{sample_idx}"
        x0_traj_path = output_dir / f"x0_traj_{sample_idx}"

        # Use b-factors to specify which residues are diffused.
        b_factors = np.tile((diffuse_mask * 100)[:, None], (1, 37))

        sample_path = analysis_utils.write_prot_to_pdb(
            bb_prot_traj[0],
            sample_path,
            b_factors=b_factors,
            aatype=aatype,
            residue_index=residue_index,
            chain_index=chain_index,
        )
        if self._cfg.inference.save_backbone_trajectory:
            prot_traj_path = analysis_utils.write_prot_to_pdb(
                bb_prot_traj,
                prot_traj_path,
                b_factors=b_factors,
                aatype=aatype,
                residue_index=residue_index,
                chain_index=chain_index,
            )
        if self._cfg.inference.save_pred_x0_trajectory:
            x0_traj_path = analysis_utils.write_prot_to_pdb(
                x0_traj,
                x0_traj_path,
                b_factors=b_factors,
                aatype=aatype,
                residue_index=residue_index,
                chain_index=chain_index,
            )
        return {
            "sample_path": sample_path,
            "traj_path": prot_traj_path,
            "x0_traj_path": x0_traj_path,
        }

    def run_protein_mpnn(self, self_consistency_dir: pathlib.Path | str) -> None:
        """Run ProteinMPNN to generate sequences.

        ProteinMPNN outputs are saved to self_consistency_dir/seqs.

        Args:
            self_consistency_dir: directory where designed protein files are stored.
        """
        if isinstance(self_consistency_dir, str):
            self_consistency_dir = pathlib.Path(self_consistency_dir)

        # Run ProteinMPNN.
        # First parse pdb file to a jsonl file.
        output_path = self_consistency_dir / "parsed_pdbs.jsonl"
        pmpnn_parser_args = [
            "python",
            f"{self._pmpnn_dir}/helper_scripts/parse_multiple_chains.py",
            f"--input_path={self_consistency_dir!s}",
            f"--output_path={output_path!s}",
        ]
        with subprocess.Popen(pmpnn_parser_args) as process:  # noqa: S603
            _ = process.wait()

        # Run ProteinMPNN sequence generation.
        num_tries = 0
        ret = -1
        pmpnn_args = [
            "python",
            f"{self._pmpnn_dir}/protein_mpnn_run.py",
            "--out_folder",
            str(self_consistency_dir),
            "--jsonl_path",
            str(output_path),
            "--num_seq_per_target",
            str(self._cfg.inference.samples.seq_per_sample),
            "--sampling_temp",
            "0.1",
            "--seed",
            "38",
            "--batch_size",
            "1",
        ]

        # Whether to run on GPU.
        if (self.gpu_id is not None) and torch.cuda.is_available():
            pmpnn_args.append("--device")
            pmpnn_args.append(str(self.gpu_id))
        else:
            # disable gpu if not available
            pmpnn_args.append("--device")
            pmpnn_args.append("cpu")
        while ret < 0:
            try:
                with subprocess.Popen(
                    pmpnn_args,  # noqa: S603
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT,
                ) as process:
                    ret = process.wait()
                logger.info("ProteinMPNN run finished successfully.")
            except ProteinMPNNError as e:
                num_tries += 1
                logger.info(f"Failed to run ProteinMPNN. Attempt {num_tries}/5.")
                torch.cuda.empty_cache()
                if num_tries > 4:
                    raise e

    def run_self_consistency(
        self,
        self_consistency_dir: pathlib.Path | str,
        reference_pdb_path: pathlib.Path | str,
        motif_mask: np.ndarray | None = None,
    ) -> None:
        """Run self-consistency on design proteins against reference protein.

        Writes ProteinMPNN outputs to self_consistency_dir/seqs.
        Writes ESMFold outputs to self_consistency_dir/esmf.
        Writes results in self_consistency_dir/sc_results.csv.

        Args:
            self_consistency_dir: directory where designed protein files are stored.
            reference_pdb_path: path to reference protein file.
            motif_mask: Optional mask of which residues are the motif.
        """
        if isinstance(self_consistency_dir, str):
            self_consistency_dir = pathlib.Path(self_consistency_dir)
        if isinstance(reference_pdb_path, str):
            reference_pdb_path = pathlib.Path(reference_pdb_path)

        self.run_protein_mpnn(self_consistency_dir)

        # Save sequences in fasta file.
        mpnn_fasta_path = (
            self_consistency_dir / "seqs" / reference_pdb_path.with_suffix(".fa").name
        )

        # Run ESMFold on each ProteinMPNN sequence and calculate metrics.
        mpnn_results: dict[str, Any] = {
            "tm_score": [],
            "sample_path": [],
            "header": [],
            "sequence": [],
            "rmsd": [],
        }
        if motif_mask is not None:
            # Only calculate motif RMSD if mask is specified.
            mpnn_results["motif_rmsd"] = []

        # Set up directory to save ESMFold predictions.
        esmf_dir = self_consistency_dir / "esmf"
        esmf_dir.mkdir(parents=True, exist_ok=True)
        fasta_seqs = fasta.FastaFile.read(mpnn_fasta_path)
        sample_feats = data_utils.parse_pdb_feats(
            "sample", reference_pdb_path, chain_id=["A"]
        )["A"]
        for i, (header, string) in enumerate(fasta_seqs.items()):
            # Run ESMFold.
            esmf_sample_path = esmf_dir / f"esmf_sample_{i}.pdb"
            _ = self.run_folding(string, esmf_sample_path)
            esmf_feats = data_utils.parse_pdb_feats(
                "folded_sample", esmf_sample_path, chain_id=["A"]
            )["A"]
            sample_seq = data_utils.aatype_to_seq(sample_feats["aatype"])

            # Calculate scTM of ESMFold outputs with reference protein.
            _, tm_score = metrics.calc_tm_score(
                sample_feats["bb_positions"],
                esmf_feats["bb_positions"],
                sample_seq,
                sample_seq,
            )
            rmsd = metrics.calc_aligned_rmsd(
                sample_feats["bb_positions"], esmf_feats["bb_positions"]
            )
            if motif_mask is not None:
                sample_motif = sample_feats["bb_positions"][motif_mask]
                of_motif = esmf_feats["bb_positions"][motif_mask]
                motif_rmsd = metrics.calc_aligned_rmsd(sample_motif, of_motif)
                mpnn_results["motif_rmsd"].append(motif_rmsd)
            mpnn_results["rmsd"].append(rmsd)
            mpnn_results["tm_score"].append(tm_score)
            mpnn_results["sample_path"].append(esmf_sample_path)
            mpnn_results["header"].append(header)
            mpnn_results["sequence"].append(string)

        # Save results to CSV
        csv_path = self_consistency_dir / "sc_results.csv"
        mpnn_results_df = pd.DataFrame(mpnn_results)
        mpnn_results_df.to_csv(csv_path)

    def run_folding(self, sequence: str, save_path: pathlib.Path | str) -> str:
        """Run ESMFold on sequence.

        Args:
            sequence: AA sequence.
            save_path: path to save folding results.

        Returns:
            ESMFold results in string format,
                i.e. the content of the saved PDB file.
        """
        with torch.no_grad():
            output = self._folding_model.infer_pdb(sequence)

        with open(save_path, "w", encoding="utf-8") as f:
            f.write(output)
        return output


@hydra.main(version_base="1.3.1", config_path="../config", config_name="inference")
def run(cfg: omegaconf.DictConfig) -> None:
    logger.info("Starting inference")
    if not torch.cuda.is_available():
        # disable gpu if not available
        cfg.inference.use_gpu = False

    start_time = time.time()
    inferencer = Inference(cfg=cfg)
    inferencer.run_sampling()
    elapsed_time = time.time() - start_time
    logger.info(f"Finished in {elapsed_time:.2f}s")


if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter
