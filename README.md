# FrameDiPT: SE(3) Diffusion Model for Protein Structure Inpainting

[![Python Version](https://img.shields.io/badge/python-3.9-blue.svg)](https://docs.python.org/3.9/library/index.html)
[![license](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-blue.svg)](./LICENSE.md)

## Table of Contents
- [Description](#description)
- [Installation](#installation)
  - [Conda](#local-with-conda)
  - [Docker](#docker)
  - [Third Party Source Code](#third-party-source-code)
- [Inference](#inference)
  - [Download Pre-trained Weights](#download-pre-trained-weights)
  - [Inference Config](#inference-config)
  - [Inference Outputs](#inference-outputs)
  - [Generate Full-atom Model](#generate-full-atom-model)
- [Evaluation](#evaluation)
  - [Inpainting Model Evaluation on TCR](#inpainting-model-evaluation-on-tcr)
  - [De novo Protein Design Evaluation](#de-novo-protein-design-evaluation)
- [Step-by-step Tutorial for Paper Reproduction](#step-by-step-tutorial-for-paper-reproduction)
- [License](#license)
- [Disclaimer of Warranties](#disclaimer-of-warranties)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)
- [Contributing](#contributing)

## Description
FrameDiPT (**FrameD**iff **i**n**P**ain**T**ing) aims to do protein structure inpainting using SE(3) diffusion model.

Below is the summary of current functionalities of the codebase.
#### Models
- [x] SE(3) graph-based diffusion model for de novo protein backbone design.
- [x] SE(3) graph-based diffusion model for protein backbone structure inpainting.
#### Diffusion Processes
- [x] SE(3) rigid-frame diffusion: Isotropic Gaussian(SO(3)) for rotation and Gaussian(R(3)) for translation.
#### Inference
- [x] De novo protein design inference with evaluation.
- [x] Protein structure inpainting inference and evaluation on TCR.

## Installation

FrameDiPT can be installed either on a host system with conda, or using Docker.

### Local with Conda

#### Conda for Linux

[Install Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
(we recommend [Miniconda](https://docs.conda.io/en/latest/miniconda.html)) and then create the environment.

```bash
conda env create --name framedipt-env --file environment.yml
conda activate framedipt-env
```

#### Install FrameDiPT Package
Install the local framedipt package in editable mode:
```bash
pip install --editable .
```

Note that `foldseek`, `anarci`, and `pbdfixer` are not supported by Conda currently for Apple Silicon.

For TCR CDR loop design, `anarci` is required and so it is recommended to use docker.

### Docker
A Dockerfile is also provided, an image can be built using:
```
docker build --file Dockerfile --tag framedipt-image:latest .
```
The image can then be run interactively with:
```
docker run -it framedipt-image
```

### Third Party Source Code

Quote from the original repo:
> Our repo keeps a fork of [OpenFold](https://github.com/aqlaboratory/openfold) since we made a few changes to the source code.
Likewise, we keep a fork of [ProteinMPNN](https://github.com/dauparas/ProteinMPNN).
Each of these codebases are actively under development, and you may want to refork.
We use copied and adapted several files from the [AlphaFold](https://github.com/deepmind/alphafold) primarily in `/data/`, and have left the DeepMind license at the top of these files.
For a differentiable pytorch implementation of the Logarithmic map on SO(3) we adapted two functions form [geomstats](https://github.com/geomstats/geomstats).
Go give these repos a star if you use this codebase!

One of our next steps is to make OpenFold and ProteinMPNN as dependency or git submodule to keep our repo clean, and facilitate further development.

## Inference

### Download Pre-trained Weights
The pre-trained weights are stored on
[InstaDeepAI HuggingFace](https://huggingface.co/InstaDeepAI).
Two models are provided: `denovo.pth` and `inpainting.pth`. Please download them using the following command:
```bash
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/InstaDeepAI/FrameDiPTModels
```
The weights will be stored under ``./FrameDiPTModels/weights``.

### Inference Config
`inference.py` is the inference script, which utilizes [Hydra](https://hydra.cc)
with config defined in ``config/inference.yaml``. See the config for different inference options.
By changing config, you can run de novo protein design and inpainting inference.
You need 1 gpu to run the inference for de novo protein design as it requires running ESMFold.
Inpainting inference could be run on cpu with default config, which may take several minutes for one sample.

Once you have created the environment and set the config,
you can run the following command to launch inference.
```bash
python experiments/inference.py
```

#### TCR CDR loop inpainting
The default config is set for TCR CDR3 loop inpainting.

The TCR dataset has been curated and automatically annotated.
Please refer to our paper for more details.
The csv files for TCR, TCR-pMHC-I and TCR-pMHC-II have been added to this repository in
`database`.

We can run the evaluation on the unbound TCR data by updating the inference config as follows:
  ```yaml
  inference:
    name: unbound_tcr
    inpainting_samples:
      tcr: True
      data_path: ./database/TCR.csv
      download_dir: /path/to/TCR_first_assemblies
      first_assembly: True
      max_len: null
  ```

Note we set the ``first_assembly`` flag to true as we downloaded the first assembly file for each
pdb id. To run inference on TCR-pMHC-I or TCR-pMHC-II, simply update the paths in the config as follows:
  ```yaml
  inference:
    name: tcr_pmhc_I
    inpainting_samples:
      tcr: True
      data_path: ./database/TCR_pMHC_I.csv
      download_dir: /path/to/TCR_pMHC_I_first_assemblies
      first_assembly: True
      max_len: null
  ```

Other options are provided for inference on TCR datasets.

One can choose to diffuse the region before or after CDR3 loops, i.e. the N-terminal/C-terminal flank
using the following config
```yaml
inference:
  inpainting_samples:
    tcr: True
    shifted_region: null # or before, or after
    ...
```

One can also diffuse all CDR loops by the following
```yaml
inference:
  inpainting_samples:
    tcr: True
    cdr_loops: [CDR1,CDR2,CDR3]
    ...
```

Particularly, we provided a small dataset `database/unbound_bound_tcr.csv` containing some examples
of same TCRs in unbound and bound states to evaluate whether the model is capable of distinguishing
unbound and bound TCRs. The PDB ID mapping of unbound and bound TCRs is shown below.
```python
{
  "2bnu": ["2bnq"],
  "1tcr": ["1g6r", "2ckb", "1mwa", "2oi9"],
  "1kgc": ["1mi5"],
  "2nw2": ["2nx5"],
  "2vlm": ["1oga"],
  "2ial": ["2ian"],
  "2z35": ["2pxy"],
}
```

#### De novo protein design
Change the following config to run de novo protein design inference
```yaml
inference:
  name: denovo
  # Whether to perform inpainting inference
  inpainting: False
  # Whether to input AA type
  input_aatype: False
  # Path to model weights.
  weights_path: ./FrameDiPTModels/weights/denovo.pth
```

You can change sample generation settings
```yaml
inference:
  samples:
    samples_per_length: # number of generated sample per sequence length.
    seq_per_sample: # number of generated sequences and therefore ESMFold samples per backbone sample.
    min_length: # minimum sequence length to sample.
    max_length: # maximum sequence length to sample.
    length_step: # gap between lengths to sample.
```
If you set ``min_length: 100, max_length: 500, length_step: 100``, then samples will be generated for length ``100, 200, 300, 400, 500``.


#### Other configs
You can also change diffusion settings
```yaml
inference:
  diffusion:
    num_t: # number of inference time steps
    noise_scale: # the noise level to use for inference, between 0 (exclusive) and 1.
    min_t: # the minimum timestep, should be slightly bigger than 0, e.g. 0.01.
```

### Inference Outputs
Samples will be saved to `output_dir` in the `inference.yaml`. By default, it is
set to `./inference_outputs/`, you can change it to the folder where you want to save the outputs.
You can also give a name to your inference run.
If it's given, the results will be saved under `output_dir/name`.
Otherwise, it will be named by the timestep when the inference is launched.
```yaml
inference:
  name: # name of your inference run
  output_dir: <path>
```
Inpainting sample outputs will be saved as follows,

```shell
inference_outputs
└── 12D_02M_2023Y_20h_46m_13s               # Date time of inference
    ├── inference_conf.yaml                 # Config used during inference
    └── {pdb_id}_length_{diffused_length}   # Sample folder
        ├── {pdb_id}_1.pdb                      # Cleaned ground truth structure
        ├── esmf_pred.pdb                       # ESMFold prediction if set in configs
        ├── diffusion_info.csv                  # CSV file containing diffusion info
        ├── sample_0                            # Sample ID
        │   ├── bb_traj_0_1.pdb               # x_{t-1} diffusion trajectory if set in configs
        │   ├── sample_0_1.pdb                # Final sample
        │   └── x0_traj_0_1.pdb               # x_0 model prediction trajectory if set in configs
        └── sample_1                            # Next sample
```

### Generate Full-atom Model
FrameDiPT generates backbone-only models, we rely on the open-source [cg2all](https://github.com/huhlim/cg2all)
to generate full-atom models. Please refer to [cg2all README](./cg2all/README.md) for more details.

## Evaluation
Once we have saved inference results,
we can run evaluation scripts to get quantitative metrics
to evaluate model performance.

### Inpainting Model Evaluation on TCR
We designed multiple metrics for TCR evaluation, please modify the configs in ``config/evaluation.yaml`` file.

Here is an example,
```yaml
# Path of saved inference results.
inference_path: /path/to/inference/outputs
# Path to save evaluation results.
eval_output_path: /path/to/save/evaluation/outputs
# Sample selection strategy, should be "mean", "median", "mode",
# "mean_closest" or "median_closest".
sample_selection_strategy: mode
# Whether to perform alignment between predictions and ground truths.
alignment: False
# Whether to exclude diffusion regions during alignment.
exclude_diffused_regions_in_alignment: True
# Whether to align structures by separate chains.
separate_alignment: True

metrics:
  model_metrics: # metrics defined per model
    - bb_rmsd
    - full_atom_rmsd
  chain_metrics: # metrics defined per chain
    - bb_rmsd
  residue_metrics: # Metrics defined per residue
    - bb_rmsd
    # - full_atom_rmsd
    - gt_asa
    - sample_asa
    - asa_abs_error
    - asa_square_error
    - gt_rsa
    - sample_rsa
    - rsa_abs_error
    - rsa_square_error
  residue_group_metrics: # metrics with more than one result per residue
    # For example, angle_error has phi, psi and omega angle metrics
    - angle_error
    - signed_angle_error
    - sample
    - gt
```
Particularly, we developed some sample selection strategies to pick the "most-likely" sample for RMSD evaluation.
The config ``sample_selection_strategy`` is used to specify which strategy we use to select sample.
We provide the following options:
- ``mean``: the mean coordinates of all generated samples
- ``median``: the geometric median coordinates of all generated samples
- ``mode``: the sample with the highest Gaussian kernel density
- ``mean_closest``: the closest sample to the mean coordinates
- ``median_closest``: the closest sample to the median coordinates

The default strategy is ``mode`` which is used for the evaluation results in the paper.

The config ``align`` is used to evaluate the results from protein folding model such as AlphaFold and ESMFold.
We can also choose to exclude diffused regions during alignment
by setting the ``exclude_diffused_regions_in_alignment`` field in the config to `true`
and to align separately the chains by the config ``separate_alignment``.

All potential metrics are listed under the config ``metrics``, we can choose to not evaluate on certain metrics
by commenting them.

Then we can run the evaluation on TCR with the following command line
```bash
python evaluation/evaluate_tcr.py
```

### De novo Protein Design Evaluation
We evaluate the performance of de novo protein design from 3 aspects:
- Designability: the quality of designed structures, which is measured by self-consistency RMSD (scRMSD).
For a generated backbone structure, we use [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) to do sequence design,
then for each designed sequence, we run [ESMFold](https://github.com/facebookresearch/esm) to predict the structure.
The designability metric scRMSD is defined as the RMSD between the generated backbone structure and ESMFold predictions.
- Diversity: we expect the protein design model to generate diverse samples, so we use [MaxCluster](http://www.sbg.bio.ic.ac.uk/~maxcluster/) to do clustering over generated samples
and the diversity is defined as ``number of clusters / number of samples``.
- Novelty: we expect the designed structures are novel w.r.t. existing structures, so we use [foldseek](https://github.com/steineggerlab/foldseek) to search similar structures over a target database (e.g. PDB).
Then the novelty pdbTM is defined as the highest TM-score with similar structures.

Firstly follow the instructions in their pages to install [MaxCluster](http://www.sbg.bio.ic.ac.uk/~maxcluster/) and [foldseek](https://github.com/steineggerlab/foldseek).

To evaluate the performance of de novo protein design, you need to fill in the ``denovo`` option in ``config/evaluation.yaml``.
Here is an example:
```yaml
inference_path: /path/to/saved/inference/results
eval_output_path: /path/to/output/evaluation/results
overwrite: False  # Whether to overwrite computed evaluation metrics.
denovo:
  pretrained_inference_path: /optional/path/to/saved/inference/results/of/pretrained/model
  esmfold_sample_choice: best  # Choice for ESMFold sample, should be `best` or `median`.
  diversity_tm_score_th: 0.5  # TM-score threshold for clustering to evaluate diversity.
  novelty_target_db: /path/to/target/database/for/foldseek/searching
```
Then run the command line
```bash
python evaluation/eval_denovo.py
```
You can change the following arguments:
- ``--esmfold_sample_choice``: you can choose `best` or `median` to evaluate the designability.
`best` will take the best sample with the smallest scRMSD
and `median` will take the sample with median scRMSD.
- ``--diversity_tm_score_th``: it's the threshold of TM-score to use for clustering.
- ``--novelty_target_db``: path to the target database to search, refer to [foldseek](https://github.com/steineggerlab/foldseek) for more details.


## Step-by-step Tutorial for Paper Reproduction
In this section, we provide a step-by-step tutorial to reproduce paper results.
- Set up the conda environment, please refer to [Conda](#local-with-conda).
- Run inference:
  - Modify the configs in ``config/inference.yaml``:
    - on TCR CDR3 loops
      ```yaml
      inference:
        name: tcr_cdr3_inpainting
        # Whether to perform inpainting inference
        inpainting: True
        input_aatype: True

        # Path to model weights.
        weights_path: ./FrameDiPTModels/weights/inpainting.pth

        inpainting_samples:
          # Whether to perform inpainting inference on TCR.
          tcr: True
          # Which CDR loops to diffuse, must give at least one loop id.
          # Could be e.g. [CDR1], [CDR1, CDR2, CDR3].
          cdr_loops: [CDR3]
          # CSV data path containing TCR samples.
          data_path: ./database/TCR.csv
          # Directory to download TCR samples.
          download_dir: /path/to/TCR_first_assemblies
          # Number of backbone samples per test case.
          samples: 5
      ```
    - on N-/C-terminal flanks
      ```yaml
      inference:
        name: tcr_n_flank_inpainting/tcr_c_flank_inpainting
        # Whether to perform inpainting inference
        inpainting: True
        input_aatype: True

        # Path to model weights.
        weights_path: ./FrameDiPTModels/weights/inpainting.pth

        inpainting_samples:
          # Whether to perform inpainting inference on TCR.
          tcr: True
          # Which CDR loops to diffuse, must give at least one loop id.
          # Could be e.g. [CDR1], [CDR1, CDR2, CDR3].
          cdr_loops: [CDR3]
          # Whether to shift region around CDR3 loop, should be "null", "before" or "after".
          shifted_region: before
          # CSV data path containing TCR samples.
          data_path: ./database/TCR.csv
          # Directory to download TCR samples.
          download_dir: /path/to/TCR_first_assemblies
          # Number of backbone samples per test case.
          samples: 5
      ```
    - on all CDR loops
      ```yaml
      inference:
        name: tcr_all_cdr_inpainting
        # Whether to perform inpainting inference
        inpainting: True
        input_aatype: True

        # Path to model weights.
        weights_path: ./FrameDiPTModels/weights/inpainting.pth

        inpainting_samples:
          # Whether to perform inpainting inference on TCR.
          tcr: True
          # Which CDR loops to diffuse, must give at least one loop id.
          # Could be e.g. [CDR1], [CDR1, CDR2, CDR3].
          cdr_loops: [CDR1, CDR2, CDR3]
          # Whether to shift region around CDR3 loop, should be "null", "before" or "after".
          shifted_region: null
          # CSV data path containing TCR samples.
          data_path: ./database/TCR.csv
          # Directory to download TCR samples.
          download_dir: /path/to/TCR_first_assemblies
          # Number of backbone samples per test case.
          samples: 5
      ```
  - Launch ``python experiments/inference.py``.
  - Results will be saved under the folder ``./inference_outputs/{inference.name}``.
- Run evaluation
  - Modify the configs in ``config/evaluation.yaml``
    ```yaml
    # Path of saved inference results.
    inference_path: /path/to/inference/outputs
    # Path to save evaluation results.
    eval_output_path: /path/to/save/evaluation/outputs
    ```
    In case of all CDR loops inpainting, evaluation is done separately on each CDR loop.
    Please change the config ``cdr_loop_index`` to 0 for CDR1, 1 for CDR2 and 2 for CDR3
    for this specific case.
  - Launch ``python evaluation/evaluate_tcr.py``

## License
[FrameDiPT: SE(3) Diffusion Model for Protein Structure Inpainting](https://github.com/instadeepai/framedipt) © 2023 by [InstaDeep Ltd](https://www.instadeep.com/) is licensed under
[CC BY-NC-SA 4.0](http://creativecommons.org/licenses/by-nc-sa/4.0/?ref=chooser-v1).

## Disclaimer of Warranties
We refer herein below to the section 5 of the [CC BY-NC-SA 4.0](http://creativecommons.org/licenses/by-nc-sa/4.0/?ref=chooser-v1) license.

```markdown
a. UNLESS OTHERWISE SEPARATELY UNDERTAKEN BY THE LICENSOR, TO THE
   EXTENT POSSIBLE, THE LICENSOR OFFERS THE LICENSED MATERIAL AS-IS
   AND AS-AVAILABLE, AND MAKES NO REPRESENTATIONS OR WARRANTIES OF
   ANY KIND CONCERNING THE LICENSED MATERIAL, WHETHER EXPRESS,
   IMPLIED, STATUTORY, OR OTHER. THIS INCLUDES, WITHOUT LIMITATION,
   WARRANTIES OF TITLE, MERCHANTABILITY, FITNESS FOR A PARTICULAR
   PURPOSE, NON-INFRINGEMENT, ABSENCE OF LATENT OR OTHER DEFECTS,
   ACCURACY, OR THE PRESENCE OR ABSENCE OF ERRORS, WHETHER OR NOT
   KNOWN OR DISCOVERABLE. WHERE DISCLAIMERS OF WARRANTIES ARE NOT
   ALLOWED IN FULL OR IN PART, THIS DISCLAIMER MAY NOT APPLY TO YOU.

b. TO THE EXTENT POSSIBLE, IN NO EVENT WILL THE LICENSOR BE LIABLE
   TO YOU ON ANY LEGAL THEORY (INCLUDING, WITHOUT LIMITATION,
   NEGLIGENCE) OR OTHERWISE FOR ANY DIRECT, SPECIAL, INDIRECT,
   INCIDENTAL, CONSEQUENTIAL, PUNITIVE, EXEMPLARY, OR OTHER LOSSES,
   COSTS, EXPENSES, OR DAMAGES ARISING OUT OF THIS PUBLIC LICENSE OR
   USE OF THE LICENSED MATERIAL, EVEN IF THE LICENSOR HAS BEEN
   ADVISED OF THE POSSIBILITY OF SUCH LOSSES, COSTS, EXPENSES, OR
   DAMAGES. WHERE A LIMITATION OF LIABILITY IS NOT ALLOWED IN FULL OR
   IN PART, THIS LIMITATION MAY NOT APPLY TO YOU.

c. The disclaimer of warranties and limitation of liability provided
   above shall be interpreted in a manner that, to the extent
   possible, most closely approximates an absolute disclaimer and
   waiver of all liability.
```

## Acknowledgements
This is a modified extended version of the
GitHub repository [se3_diffusion](https://github.com/jasonkyuyim/se3_diffusion)
from [Yim et al., 2023](https://arxiv.org/pdf/2302.02277.pdf).

## Citation
If you find this repository useful in your work, please add the following citation to our [paper](https://www.biorxiv.org/content/10.1101/2023.11.21.568057v1).
```bibtex
@article {Zhang2023.11.21.568057,
	author = {Cheng Zhang and Adam Leach and Thomas Makkink and Miguel Arbes{\'u} and Ibtissem Kadri and Daniel Luo and Liron Mizrahi and Sabrine Krichen and Maren Lang and Andrey Tovchigrechko and Nicolas Lopez Carranza and U{\u g}ur {\c S}ahin and Karim Beguir and Michael Rooney and Yunguan Fu},
	title = {FrameDiPT: SE(3) Diffusion Model for Protein Structure Inpainting},
	elocation-id = {2023.11.21.568057},
	year = {2023},
	doi = {10.1101/2023.11.21.568057},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {Protein structure prediction field has been revolutionised by deep learning with protein folding models such as AlphaFold 2 and ESMFold. These models enable rapid in silico prediction and have been integrated into de novo protein design and protein-protein interaction (PPI) prediction. However, biologically relevant features dependent on conformational distributions cannot be estimated with these models. Diffusion models, a novel class of generative models, have been developed to learn conformational distributions and applied to de novo protein design. Limited work has been done on protein structure inpainting, where a masked section is recovered by simultaneously conditioning on its sequence and the rest of the structure. In this work, we propose FrameDiff inPainTing (FrameDiPT), a generalised model for protein inpainting. This is important for T-cells given the hyper-variability of the complementarity determining region (CDR) loops. We evaluated the model on CDR loop design for T-cell receptors and achieved comparable prediction accuracy to ProteinGenerator and RFdiffusion with limited training data and learnable parameters. Different from deterministic structure prediction models, FrameDiPT captures the conformational distribution at different regions and binding states, highlighting a key advantage of generative models.Competing Interest StatementCheng Zhang, Adam Leach, Thomas Makkink, Miguel Arbesu, Ibtissem Kadri, Daniel Luo, Liron Mizrahi, Sabrine Krichen, Nicolas Lopez Carranza, Karim Beguir and Yunguan Fu were affiliated with InstaDeep Ltd during the preparation of this manuscript. Maren Lang, Andrey Tovchigrechko, Ugur {\c S}ahin and Michael Rooney were affiliated with BioNTech during the preparation of this manuscript. InstaDeep Ltd was acquired by BioNTech.},
	URL = {https://www.biorxiv.org/content/early/2023/11/21/2023.11.21.568057},
	eprint = {https://www.biorxiv.org/content/early/2023/11/21/2023.11.21.568057.full.pdf},
	journal = {bioRxiv}
}
```

## Contributing

### Pre-commit

Install pre-commit hooks:

```bash
pre-commit install
```

Update hooks, and re-verify all files.

```bash
pre-commit autoupdate
pre-commit run --all-files
```
