"""Integration test for the inference scripts."""
import pathlib
import shutil

import hydra

from experiments import inference

NUM_SIGMAS = 10
NUM_OMEGAS = 10

NUM_STRUCTURES = 3


def test_inference(tmp_path: pathlib.Path) -> None:
    """Integration test of end-to-end inference.

    This function performs the following checks:
    - Checks that the saved checkpoint can be successfully loaded and applied to
        inference on a small subset of TCR data, consisting of the three
        structures with PDB ID 1fyt, 5ksa, and 7t2d.
    - Checks that all expected files are generated during the inference process.

    The test is designed to be run on CPU as part of the CI/CD pipeline.
    To be a fast test, some config parameters must be overwritten: we use
    a smaller version of the score network. By design, the dataset is small,
    containing only 3 structures. Since each batch contains noised versions
    of the same original structure, the total number of steps in one epoch
    is the number of structures in the dataset, i.e. 3 in this case.

    Args:
        tmp_path: directory in which to save directories and folders created during
            the test.

    The file structure below shows which directories and files are expected to be
    created in the test directory during inference, assuming that
    `NUM_SIGMAS=10` and `NUM_OMEGAS=10`.

    tmp_path/
    ├── cache/
    │   └── eps_10_omega_10_min_sigma_0_1_max_sigma_1_5_schedule_logarithmic/"
    │       ├── pdf_vals.npy
    │       ├── cdf_vals.npy
    │       └── score_norms.npy
    │
    └── inference_dir/
        └── inference_subdir/
            ├── 1fyt-assembly1_length_60/
            │   ├── 1fyt-assembly1_1.pdb
            │   ├── diffusion_info.csv
            │   └── sample_0
            │       └── sample_0_1.pdb
            │
            ├── 5ksa-assembly1_length_65/
            │   ├── ...
            │
            ├── 7t2d-assembly1_length_62/
            │   ├── ...
            │
            └── inference_conf.yaml

    Below is the file structure showing data used in the test:

    ./tests/data
    └── inference_data/
        ├── inference_TCR_pMHC_II.csv
        └── structures/
            ├── cifs/
            │    ├── 1fyt-assembly1.cif
            │    ├── 5ksa-assembly.cif
            │    └── 7t2d-assembly1.cif
            │
            └── processed/ (CREATED DURING TEST AND DELETED AFTER TEST)
                ...


    """
    # Paths for data files being read in the test
    data_dir = pathlib.Path("./tests/data/")
    ckpt_path = pathlib.Path("./FrameDiPTModels/weights/inpainting.pth")

    # INFERENCE
    data_path = data_dir / "inference_data" / "inference_TCR_pMHC_II.csv"
    download_dir = data_dir / "inference_data" / "structures"
    inference_dir = tmp_path / "inference_dir"
    inference_subdir_name = "generated_samples"
    inference_subdir = inference_dir / inference_subdir_name

    inference_overrides = [
        f"inference.name={inference_subdir_name}",
        f"inference.output_dir={inference_dir}",
        "inference.use_gpu=False",
        "inference.inpainting=True",
        "inference.input_aatype=True",
        f"inference.weights_path={ckpt_path}",
        "inference.inpainting_samples.run_esmfold=False",
        "inference.inpainting_samples.overwrite=True",
        "inference.inpainting_samples.tcr=False",
        "inference.inpainting_samples.samples=1",  # Only generate 1 sample
        f"inference.inpainting_samples.data_path={data_path}",
        f"inference.inpainting_samples.download_dir={download_dir}",
        "inference.diffusion.num_t=1",  # Only take 1 step in backward process
        "inference.confidence_score=null",
        # Avoid multithreading issue
        "inference.inpainting_samples.num_workers_download=1",
    ]

    with hydra.initialize(config_path="../../config"):
        inference_config = hydra.compose(
            config_name="inference.yaml", overrides=inference_overrides
        )

    inference.run(inference_config)

    # Verify correct files are created during inference
    assert (download_dir / "cifs" / "1fyt-assembly1.cif").exists()
    assert (download_dir / "cifs" / "5ksa-assembly1.cif").exists()
    assert (download_dir / "cifs" / "7t2d-assembly1.cif").exists()
    assert (download_dir / "processed" / "metadata.csv").exists()
    assert (download_dir / "processed" / "fy" / "1fyt-assembly1.pkl").exists()
    assert (download_dir / "processed" / "ks" / "5ksa-assembly1.pkl").exists()
    assert (download_dir / "processed" / "t2" / "7t2d-assembly1.pkl").exists()

    subfolder_prefixes = [
        "1fyt-assembly1_length_",
        "5ksa-assembly1_length_",
        "7t2d-assembly1_length_",
    ]
    assert inference_dir.exists()
    assert inference_subdir.exists()
    for prefix in subfolder_prefixes:
        for subfolder in inference_subdir.glob(prefix + "*"):
            assert subfolder.exists()
            assert (subfolder / f"{prefix.split('_', maxsplit=1)[0]}_1.pdb").exists()
            assert (subfolder / "diffusion_info.csv").exists()
            assert (subfolder / "sample_0").exists()
            assert (subfolder / "sample_0" / "sample_0_1.pdb").exists()

    # Remove subdirectory generated in local directory during inference
    shutil.rmtree(download_dir / "processed")
