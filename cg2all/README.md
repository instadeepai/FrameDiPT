# cg2all

## Description

FrameDiPT is currently a backbone-only model. We would like to extend FrameDiPT to be full-atom, as important binding interactions may be captured by the
position of the side chains. A recent tool [cg2all](https://github.com/huhlim/cg2all) has been published which converts coarse-grain (including backbone) protein structures to all-atom models.

### Installation

Cg2all can be installed using conda.

```base
conda env create --name cg2all --file cg2all/cg2all.yml
conda activate cg2all
```

Having activated the conda environment, a bash script has been provided to run predictions on the FrameDiPT folder.
The script has only been tested on CPU. To launch the script, use the following command:

```bash
bash cg2all/convert_backbone_to_full_atom.sh path/to/prediction/dir
```
whereby the first argument to the script is the root directory of the FrameDiPT predictions.

We also provide an option ``--standard-name`` to use the IUPAC standard atom names
instead of CHARMM's atom names, to be consistent with AlphaFold naming format.

 Note the expected folder structure for FrameDiPT predictions is:
```base
  root_dir/
    pdb_id_1_length_x/
      sample_0/
      ...
      sample_n/
    ...
    pdb_id_m_length_x/
```

The full-atom predictions will be saved
in the same folder as the original backbone predictions, with the suffix "_all_atom.pdb". For example, the final directory may contain a folder
with the structure:
```bash
  root_dir/
    pdb_id_1_length_x/
      sample_0/
        sample_0_1.pdb
        sample_0_1_full_atom.pdb
        ...
```
