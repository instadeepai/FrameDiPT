"""Library for parsing different data structures."""
# pylint: disable=W1514,C0103,R1732
from __future__ import annotations

import gzip
import string

import numpy as np
from Bio.PDB.Chain import Chain

from framedipt.protein import protein, residue_constants

Protein = protein.Protein


def process_chain(chain: Chain, chain_id: int | str) -> protein.Protein:
    """Convert a PDB chain object into a AlphaFold Protein instance.

    Forked from alphafold.common.protein.from_pdb_string

    WARNING: All non-standard residue types will be converted into UNK. All
        non-standard atoms will be ignored.

    Took out lines 94-97 which don't allow insertions in the PDB.
    Sabdab uses insertions for the chothia numbering so we need to allow them.

    Took out lines 110-112 since that would mess up CDR numbering.

    Args:
        chain: Instance of Biopython's chain class.

    Returns:
        Protein object with protein features.
    """
    atom_positions = []
    aatype = []
    atom_mask = []
    residue_index = []
    b_factors = []
    chain_ids = []
    for res in chain:
        res_shortname = residue_constants.restype_3to1.get(res.resname, "X")
        restype_idx = residue_constants.restype_order.get(
            res_shortname, residue_constants.restype_num
        )
        pos = np.zeros((residue_constants.atom_type_num, 3))
        mask = np.zeros((residue_constants.atom_type_num,))
        res_b_factors = np.zeros((residue_constants.atom_type_num,))
        for atom in res:
            if atom.name not in residue_constants.atom_types:
                continue
            pos[residue_constants.atom_order[atom.name]] = atom.coord
            mask[residue_constants.atom_order[atom.name]] = 1.0
            res_b_factors[residue_constants.atom_order[atom.name]] = atom.bfactor
        aatype.append(restype_idx)
        atom_positions.append(pos)
        atom_mask.append(mask)
        residue_index.append(res.id[1])
        b_factors.append(res_b_factors)
        chain_ids.append(chain_id)

    return Protein(
        atom_positions=np.array(atom_positions),
        atom_mask=np.array(atom_mask),
        aatype=np.array(aatype),
        residue_index=np.array(residue_index),
        chain_index=np.array(chain_ids),
        b_factors=np.array(b_factors),
    )


# read A3M and convert letters into
# integers in the 0..20 range,
# also keep track of insertions
def parse_a3m(filename: str) -> tuple[np.ndarray, np.ndarray]:
    msa = []
    ins = []

    table = str.maketrans(dict.fromkeys(string.ascii_lowercase))

    if filename.split(".")[-1] == "gz":
        fp = gzip.open(filename, "rt")
    else:
        fp = open(filename)

    # read file line by line
    for line in fp:
        # skip labels
        if line[0] == ">":
            continue

        # remove right whitespaces
        line = line.rstrip()

        if len(line) == 0:
            continue

        # remove lowercase letters and append to MSA
        msa.append(line.translate(table))

        # sequence length
        L = len(msa[-1])

        # 0 - match or gap; 1 - insertion
        a = np.array([0 if c.isupper() or c == "-" else 1 for c in line])
        i = np.zeros(L)

        if np.sum(a) > 0:
            # positions of insertions
            pos = np.where(a == 1)[0]

            # shift by occurrence
            a = pos - np.arange(pos.shape[0])

            # position of insertions in cleaned sequence
            # and their length
            pos, num = np.unique(a, return_counts=True)

            # append to the matrix of insetions
            i[pos] = num

        ins.append(i)
        if len(msa) == 10000:
            break

    # convert letters into numbers
    alphabet = list("ARNDCQEGHILKMFPSTWYV-")
    encoding = np.array(alphabet, dtype="|S1").view(np.uint8)
    msa_array = np.array([list(s) for s in msa], dtype="|S1").view(np.uint8)
    for letter, enc in zip(alphabet, encoding):
        res_cat = residue_constants.restype_order_with_x.get(
            letter, residue_constants.restype_num
        )
        msa_array[msa_array == enc] = res_cat

    # treat all unknown characters as gaps
    msa_array[msa_array > 20] = 20

    ins_array = np.array(ins, dtype=np.uint8)

    return msa_array, ins_array
