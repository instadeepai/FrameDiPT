"""Module for inference sampler."""
from __future__ import annotations

import pathlib

import numpy as np
import omegaconf
import pandas as pd
import torch
import tree

from framedipt.data import process_pdb_dataset
from framedipt.data import utils as data_utils
from framedipt.diffusion import se3_diffuser
from framedipt.protein import tcr
from framedipt.tools.log import get_logger
from openfold.utils import rigid_utils

logger = get_logger()


class UnconditionalSampler(torch.utils.data.Dataset):
    """Sampler class for de novo protein design."""

    def __init__(
        self,
        cfg: omegaconf.DictConfig,
        diffuser: se3_diffuser.SE3Diffuser,
        device: str,
    ) -> None:
        """Initialize unconditional sampler.

        Args:
            cfg: sampler config.
            diffuser: object of SE3Diffuser, used for sampling.
            device: device to use, e.g. "cpu", "cuda:0".
        """
        self._cfg = cfg
        self._diffuser = diffuser
        self.device = device
        self.all_sampling_lengths = self.get_sampling_lengths()

    def get_sampling_lengths(self) -> np.ndarray:
        """Get all sampling lengths for inference.

        Sampling length is the length of the amino acid sequence
            to be sampled.

        All sampling lengths are defined in `self._cfg`.
        It starts from `min_length`, ends with `max_length`, with gap `length_step`.
        For each length, it's repeated `samples_per_length` times.
        For example, if we have
            `min_length=100, max_length=200, length_step=50, samples_per_length=3`,
            the generated array will be [100, 100, 100, 150, 150, 150, 200, 200, 200].

        Returns:
            Array containing all sampling lengths.
        """
        all_sample_lengths = range(
            self._cfg.min_length,
            self._cfg.max_length + 1,
            self._cfg.length_step,
        )

        all_sample_lengths = np.repeat(all_sample_lengths, self._cfg.samples_per_length)

        return all_sample_lengths

    def sample(self, sample_length: int) -> dict[str, torch.Tensor]:
        """Sample input data for inference based on length.

        Args:
            sample_length: sequence length to be sampled.

        Returns:
            Dictionary of tensors for inference inputs.
                - res_mask: residue mask, shape [sample_length].
                - seq_idx: residue indices, shape [sample_length].
                - fixed_mask: masked for fixed residues, shape [sample_length].
                - torsion_angles_sin_cos: torsion angles in sin-cos format,
                    shape [sample_length, 7, 2].
                - sc_ca_t: carbon-alpha coordinates used for self-conditioning,
                    shape [sample_length, 3].
                - rigid_t: rigid frame, shape [sample_length, 7].
        """
        # Process motif features.
        res_mask = np.ones(sample_length)
        fixed_mask = np.zeros_like(res_mask)

        # Initialize data
        ref_sample = self._diffuser.sample_ref(
            n_samples=sample_length,
            as_tensor_7=True,
        )
        res_indices = torch.arange(1, sample_length + 1)
        init_feats = {
            "res_mask": res_mask,
            "seq_idx": res_indices,
            "fixed_mask": fixed_mask,
            "torsion_angles_sin_cos": np.zeros((sample_length, 7, 2)),
            "sc_ca_t": np.zeros((sample_length, 3)),
            **ref_sample,
        }
        # Convert to tensors if some features are not.
        init_feats = tree.map_structure(
            lambda x: x if torch.is_tensor(x) else torch.tensor(x), init_feats
        )
        # Add batch dimension and move to GPU.
        init_feats = tree.map_structure(lambda x: x[None].to(self.device), init_feats)

        return init_feats

    def __len__(self) -> int:
        """Get length of the dataset.

        Returns:
            Length of the dataset, i.e. the total number of sampling lengths.
        """
        return len(self.all_sampling_lengths)

    def __getitem__(self, item: int) -> tuple[int, int, dict[str, torch.Tensor]]:
        """Get item of the dataset.

        Args:
            item: index of the item.

        Returns:
            sample_length: sequence length of the sample.
            sample_i: index of the i-th sample among samples of the same length.
            sample_item: dictionary of tensors for inference inputs for the item.
        """
        sample_length = self.all_sampling_lengths[item]
        sample_i = item % self._cfg.samples_per_length
        sample_item = self.sample(sample_length)
        return sample_length, sample_i, sample_item


class ConditionalSampler(torch.utils.data.Dataset):
    """Sampler class for inpainting validation on CASP dataset."""

    def __init__(
        self,
        data_conf: omegaconf.DictConfig,
        diffuser: se3_diffuser.SE3Diffuser,
        device: str,
    ) -> None:
        """Init function.

        Args:
            data_conf: dataset configuration.
            diffuser: object of SE3Diffuer to perform diffusion.
            device: device to use, e.g. "cpu", "cuda:0".
        """
        self._data_conf = data_conf
        self._init_metadata()
        self._diffuser = diffuser
        self.device = device
        self.diffused_masks: dict[int, np.ndarray] = {}
        self.rng = np.random.default_rng(self._data_conf.seed)

    @property
    def diffuser(self) -> se3_diffuser.SE3Diffuser:
        return self._diffuser

    @property
    def data_conf(self) -> omegaconf.DictConfig:
        return self._data_conf

    def get_chains_to_process(self) -> list[list[str]] | list[None]:
        """Get chains to process for each sample in the dataset.

        For generic ConditionalSampler, all chains will be processed.
        Thus, we return a list of None.

        This function is created so that TCRSampler inheriting this class
            could override this function, and select TCR chains to process.

        Returns:
            List of None, meaning all chains of each sample will be processed.
        """
        return [None] * len(self.pdb_files)

    def _init_metadata(self) -> None:
        """Initialize metadata by applying filters in `data_conf`."""
        download_dir = pathlib.Path(self.data_conf.download_dir)
        metadata_path = download_dir / "processed/metadata.csv"

        # Download mmcif files, if already existing, will skip.
        self.pdb_csv = pd.read_csv(self.data_conf.data_path)
        data_utils.download_cifs(
            self.pdb_csv["pdb_id"].to_numpy(),
            download_dir,
            assembly=self.data_conf.first_assembly,
            num_workers=self.data_conf.num_workers_download,
        )
        self.pdb_files = list((download_dir / "cifs").glob("*.cif"))

        # Take intersection of pdb files and pdb ids
        pdb_ids = self.pdb_csv["pdb_id"].to_numpy()
        self.pdb_files = [path for path in self.pdb_files if path.stem[:4] in pdb_ids]

        self.all_chains_to_process = self.get_chains_to_process()

        if metadata_path.exists() and not self.data_conf.overwrite:
            self.metadata = pd.read_csv(metadata_path)
            logger.info(f"Dataset length {len(self.metadata)}.")
        else:
            all_metadata = process_pdb_dataset.process_serially(
                all_mmcif_paths=self.pdb_files,
                max_resolution=self._data_conf.max_resolution,
                max_len=self.data_conf.max_len,
                min_len=self.data_conf.min_len,
                chain_max_len=self.data_conf.chain_max_len,
                chain_min_len=self.data_conf.chain_min_len,
                max_num_chains=self.data_conf.max_num_chains,
                write_dir=download_dir / "processed",
                all_chains_to_process=self.all_chains_to_process,
                check_valid_resolution=self.data_conf.check_valid_resolution,
            )
            self.metadata = pd.DataFrame(all_metadata)
            self.metadata.to_csv(metadata_path, index=False)
            logger.info(
                f"Finished processing {len(self.metadata)}/{len(self.pdb_files)} files"
            )

    def create_diffusion_mask(
        self, chain_feats: dict[str, np.ndarray | torch.Tensor], example_idx: int
    ) -> np.ndarray:
        """Create diffusion mask.

        A randomly selected continuous region is masked for diffusion.

        Args:
            chain_feats: dictionary of processed chain features.
            example_idx: index of the example in the dataset.

        Returns:
            The diffusion mask.
            If example_idx already exists in self.diffused_masks,
                the pre-computed diffused mask will be returned.
            This reduces unnecessary compute.
        """
        if self.diffused_masks.get(example_idx) is not None:
            return self.diffused_masks[example_idx]

        # Use a fixed seed for evaluation.
        rng = np.random.default_rng(example_idx)
        diffused_mask = data_utils.create_redacted_regions(
            chain_feats["chain_idx"],
            chain_feats["res_mask"],
            rng,
            redact_min_len=self.data_conf.redaction.redact_min_len,
            redact_max_len=self.data_conf.redaction.redact_max_len,
        )
        self.diffused_masks[example_idx] = diffused_mask

        return diffused_mask

    def __len__(self) -> int:
        """Get length of the dataset.

        Returns:
            Length of the dataset, i.e. the total number of samples to be generated.
        """
        return len(self.metadata) * self.data_conf.samples

    def __getitem__(self, idx: int) -> tuple[str, int, dict[str, torch.Tensor]]:
        """Get item of the dataset.

        Args:
            idx: index of the item in the dataset.

        Returns:
            Dictionary of features or optional pdb name.
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
        """
        # Sample data example.
        example_idx = idx // self.data_conf.samples
        sample_idx = idx % self.data_conf.samples
        csv_row = self.metadata.iloc[example_idx]
        pdb_name = csv_row["pdb_name"]
        processed_file_path = pathlib.Path(csv_row["processed_path"])
        chain_feats = data_utils.process_csv_row(
            processed_file_path,
            process_monomer=False,
            extract_single_chain=False,
            rng=self.rng,
        )

        gt_bb_rigid = rigid_utils.Rigid.from_tensor_4x4(chain_feats["rigidgroups_0"])[
            :, 0
        ]

        diffused_mask = self.create_diffusion_mask(
            chain_feats=chain_feats, example_idx=example_idx
        )

        if np.sum(diffused_mask) < 1:
            raise ValueError("Must be diffused")
        fixed_mask: np.ndarray = 1 - diffused_mask
        chain_feats["fixed_mask"] = fixed_mask
        chain_feats["rigids_0"] = gt_bb_rigid.to_tensor_7()
        chain_feats["sc_ca_t"] = torch.zeros_like(gt_bb_rigid.get_trans())

        # Fix t to final timestep 1.0
        # and sample from reference distribution and diffuse.
        t = 1.0
        diff_feats_t = self.diffuser.sample_ref(
            n_samples=gt_bb_rigid.shape[0],
            chain_index=chain_feats["chain_idx"],
            impute=gt_bb_rigid,
            diffuse_mask=diffused_mask,
            as_tensor_7=True,
        )

        chain_feats.update(diff_feats_t)
        chain_feats["t"] = t

        # Convert all features to tensors.
        final_feats = tree.map_structure(
            lambda x: x if torch.is_tensor(x) else torch.tensor(x), chain_feats
        )

        # Pad features
        final_feats = data_utils.pad_feats(
            final_feats, csv_row["modeled_seq_len"], use_torch=True
        )

        # Add batch dimension and move to GPU.
        final_feats = tree.map_structure(lambda x: x[None].to(self.device), final_feats)

        return pdb_name, sample_idx, final_feats


class TCRSampler(ConditionalSampler):
    """Sampler class for inpainting validation on TCR dataset.

    It inherits ConditionalSampler.
    """

    def __init__(
        self,
        data_conf: omegaconf.DictConfig,
        diffuser: se3_diffuser.SE3Diffuser,
        device: str,
    ) -> None:
        """Init function.

        Args:
            data_conf: dataset configuration.
            diffuser: object of SE3Diffuer to perform diffusion.
            device: device to use, e.g. "cpu", "cuda:0".
        """
        super().__init__(
            data_conf=data_conf,
            diffuser=diffuser,
            device=device,
        )

    def get_chains_to_process(self) -> list[list[str]] | list[None]:
        """Get chains to process for the whole TCR dataset.

        For each TCR sample, according to whether it's bound to pMHC,
            we may have the following options to process.
                - 2 chains (TCR alpha and beta);
                - 4 chains (TCR alpha and beta, peptide, MHC-I chain);
                - 5 chains (TCR alpha and beta, peptide, MHC-II alpha and beta).

        Returns:
            List of chain lists to process for each tcr sample.
        """
        all_chains_to_process = []
        for pdb_file in self.pdb_files:
            # Find the example in the dataframe corresponding to pdb file.
            pdb_id = pdb_file.stem

            # Note first assembly files are of the format {pdb_id}-assembly1.cif, not
            # {pdb_id}.cif.
            if self.data_conf.first_assembly:
                pdb_id = pdb_id[:4]

            example = self.pdb_csv[self.pdb_csv["pdb_id"] == pdb_id].iloc[0]
            # Get TCR alpha and beta chain ids.
            tcr_alpha_chain = example["tcr_alpha_chain"]
            tcr_beta_chain = example["tcr_beta_chain"]
            chains = [tcr_alpha_chain, tcr_beta_chain]
            # Get pMHC chains if there is any.
            if example["peptide_chain"] is not None and isinstance(
                example["peptide_chain"], str
            ):
                chains.append(example["peptide_chain"])
            if example["mhc_alpha_chain"] is not None and isinstance(
                example["mhc_alpha_chain"], str
            ):
                chains.append(example["mhc_alpha_chain"])
            if example["mhc_beta_chain"] is not None and isinstance(
                example["mhc_beta_chain"], str
            ):
                chains.append(example["mhc_beta_chain"])

            all_chains_to_process.append(chains)
        return all_chains_to_process

    def create_diffusion_mask(
        self,
        chain_feats: dict[str, np.ndarray | torch.Tensor],
        example_idx: int,
    ) -> np.ndarray:
        """Create diffusion mask, overriding the one in ConditionalSampler.

        CDR3 loops in TCR alpha and beta chain are masked for diffusion.

        Args:
            chain_feats: dictionary of processed chain features.
            example_idx: index of the example in the dataset.

        Returns:
            The diffusion mask.
            If example_idx already exists in self.diffused_masks,
                the pre-computed diffused mask will be returned.
            This reduces unnecessary compute.

        Raises:
            ValueError if the chains to process for the sample is None
                or cdr_loops is not given in the data config.
        """
        if self.diffused_masks.get(example_idx) is not None:
            return self.diffused_masks[example_idx]

        chains_to_process = self.all_chains_to_process[example_idx]
        if chains_to_process is None:
            raise ValueError("Should have chains to process for TCRSampler, got None.")
        if self.data_conf.cdr_loops is None or len(self.data_conf.cdr_loops) == 0:
            raise ValueError("CDR loops should be given in the config.")

        diffused_mask = tcr.create_diffusion_mask(
            chain_indexes=chain_feats["chain_idx"],
            aatype=data_utils.move_to_np(chain_feats["aatype"]),
            tcr_chains=chains_to_process[:2],
            cdr_loops=self.data_conf.cdr_loops,
            shifted_region=self.data_conf.shifted_region,
        )
        self.diffused_masks[example_idx] = diffused_mask

        return diffused_mask
