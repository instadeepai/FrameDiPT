"""Utils for aligning structures."""
from __future__ import annotations

from typing import Callable, NamedTuple

import numpy as np
from Bio import Align, SeqUtils, SVDSuperimposer
from Bio.PDB import Atom, Chain, Model

from framedipt.protein import filters


class SharedResidue(NamedTuple):
    """Container for shared residue within two chains.

    Fields:
        name: Name of the shared residue.
        atoms_1: List of common atoms belonging to the residue in the first chain.
        atoms_1: List of common atoms belonging to the residue in the second chain.
        res_idx_1: Residue index of the first chain.
        res_idx_2: Residue index of the second chain.
    """

    name: str
    atoms_1: list[Atom]
    atoms_2: list[Atom]
    res_idx_1: int
    res_idx_2: int


# Filter to process atoms
AtomFilter = Callable[[Atom], bool]


def align(
    model_1: Model.Model,
    model_2: Model.Model,
    ref_chains: list[str],
    model_1_exclude_region: list[tuple[int, int]] | None = None,
    model_2_exclude_region: list[tuple[int, int]] | None = None,
    atom_filter: AtomFilter = filters.is_backbone,
    subs_matrix: str = "BLOSUM62",
    open_penalty: float = -10,
    extend_penalty: float = -0.5,
) -> None:
    """Align two models using a list of shared chains as reference objects.

    The `model_1` will be aligned to the `model_2`.
    The alignment transforms all atoms in the `model_1`.

    Args:
        model_1: Model 1 to be aligned.
        model_2: Model 2 to which Model 1 will be aligned to.
        ref_chains: Reference chains used in the alignment.
        model_1_exclude_region: list of tuple of start and end indexes
            indicating the excluded region in model 1.
        model_2_exclude_region: list of tuple of start and end indexes
            indicating the excluded region in model 2.
        atom_filter: Filter function to be applied to each atom.
        subs_matrix: Substitution matrix to use in the alignment.
        open_penalty: Penalty to apply to each open "-" residue. PyMol default.
        extend_penalty: Penalty to apply to each "-" appended. PyMol default.
    """
    # Atoms forming the reference frame of the transformation
    self_atoms, other_atoms = get_shared_residues_and_atoms(
        model_1=model_1,
        model_2=model_2,
        chains=ref_chains,
        model_1_exclude_region=model_1_exclude_region,
        model_2_exclude_region=model_2_exclude_region,
        atom_filter=atom_filter,
        subs_matrix=subs_matrix,
        open_penalty=open_penalty,
        extend_penalty=extend_penalty,
    )
    align_model_from_shared_atoms(model_1, self_atoms, other_atoms)


def get_shared_residues_and_atoms(
    model_1: Model.Model,
    model_2: Model.Model,
    chains: list[str],
    model_1_exclude_region: list[tuple[int, int]] | None = None,
    model_2_exclude_region: list[tuple[int, int]] | None = None,
    atom_filter: AtomFilter = filters.is_backbone,
    subs_matrix: str = "BLOSUM62",
    open_penalty: float = -10,
    extend_penalty: float = -0.5,
) -> tuple[list[Atom], list[Atom]]:
    """Get shared residues and shared atoms among those residues among two chains.

    Args:
        model_1: Model 1 used in the comparison.
        model_2: Model 2 used in the comparison.
        chains: Chains used in the comparison.
        model_1_exclude_region: list of tuple of start and end indexes
            indicating the excluded region in model 1.
        model_2_exclude_region: list of tuple of start and end indexes
            indicating the excluded region in model 2.
        atom_filter: Filter to be applied to each atom.
        subs_matrix: Substitution matrix to use in the alignment.
        open_penalty: Penalty to apply to each open "-" residue. PyMol default.
        extend_penalty: Penalty to apply to each "-" appended. PyMol default.

    Raises:
        ChainNotFoundError: If models do not contain provided chains.

    Returns:
        Tuple containing shared atoms between both BenchmarkModel. The
        atoms are ordered: They are 1-to-1 comparable.
    """
    if not all(chain in model_1 for chain in chains):
        raise ValueError(f"Chains {chains} cannot be found in {model_1}.")
    if not all(chain in model_2 for chain in chains):
        raise ValueError(f"Chains {chains} cannot be found in {model_2}.")

    if model_1_exclude_region is None:
        model_1_exclude_region = [(-1, -1)] * len(chains)
    if model_2_exclude_region is None:
        model_2_exclude_region = [(-1, -1)] * len(chains)

    self_shared_atoms, other_shared_atoms = [], []
    for compared_chain, chain_1_exclude_region, chain_2_exclude_region in zip(
        chains, model_1_exclude_region, model_2_exclude_region
    ):
        # Obtain the relevant chain for each model
        self_chain = model_1[compared_chain]
        other_chain = model_2[compared_chain]

        # Get the list of shared atoms between both chains
        shared_residues_info = get_shared_residues_info(
            chain_1=self_chain,
            chain_2=other_chain,
            chain_1_exclude_region=chain_1_exclude_region,
            chain_2_exclude_region=chain_2_exclude_region,
            atom_filter=atom_filter,
            subs_matrix=subs_matrix,
            open_penalty=open_penalty,
            extend_penalty=extend_penalty,
        )
        self_chain_atoms, other_chain_atoms = unroll_shared_residues(
            shared_residues_info
        )

        self_shared_atoms.extend(self_chain_atoms)
        other_shared_atoms.extend(other_chain_atoms)

    return self_shared_atoms, other_shared_atoms


def align_model_from_shared_atoms(
    model: Model.Model,
    self_atoms: list[Atom],
    other_atoms: list[Atom],
) -> None:
    """Align a model to another one using the precomputed list of shared atoms.

    The model will be aligned in place
        by transforming `self_atoms` to align with `other_atoms`.
    The shared atoms should be retrieved from .get_shared_residues_and_atoms().

    Args:
        model: Structure model to be aligned.
        self_atoms: List of all self shared atoms between model and another model.
        other_atoms: List of all other shared atoms between model and another model.
    """
    # Obtain the coordinates
    self_transcoords = np.array([atom.coord for atom in self_atoms])
    other_transcoords = np.array([atom.coord for atom in other_atoms])

    # Superimpose the set of reference atoms and translate all atoms
    super_imposer = SVDSuperimposer.SVDSuperimposer()
    super_imposer.set(other_transcoords, self_transcoords)
    super_imposer.run()

    # Obtain the rotation and translation matrix
    rot, tran = super_imposer.get_rotran()

    # Transform all atoms in the model
    for atom in model.get_atoms():
        atom.transform(rot, tran)


def get_chain_sequence(chain: Chain.Chain) -> str:
    """Retrieve 1-hot encoding (A for Alanine) of the chain as a string.

    Args:
        chain: chain to retrieve residue sequence from.

    Returns:
        1-letter residue sequence.
    """
    residue_names = []
    for residue in chain.get_residues():
        # Get 3-letter residue name.
        residue_name_3letter = residue.get_resname()
        # Get 1-letter residue name.
        residue_name_1letter = SeqUtils.seq1(residue_name_3letter)
        residue_names.append(residue_name_1letter)
    return "".join(residue_names)


def get_shared_residues_info(
    chain_1: Chain.Chain,
    chain_2: Chain.Chain,
    chain_1_exclude_region: tuple[int, int] | None = None,
    chain_2_exclude_region: tuple[int, int] | None = None,
    atom_filter: AtomFilter = filters.is_backbone,
    subs_matrix: str = "BLOSUM62",
    open_penalty: float = -10,
    extend_penalty: float = -0.5,
) -> list[SharedResidue]:
    """Get all shared residues and their atoms from two chains.

    Args:
        chain_1: First chain used in the comparison.
        chain_2: Second chain used in the comparison.
        chain_1_exclude_region: tuple of start and end indexes
            indicating the excluded region in chain 1.
        chain_2_exclude_region: tuple of start and end indexes
            indicating the excluded region in chain 2.
        atom_filter: Filter to be applied to each atom
        subs_matrix: Substitution matrix used in the alignment.
        open_penalty: Penalty to apply to each open "-" residue. PyMol default.
        extend_penalty: Penalty to apply to each "-" appended. PyMol default.

    Information about alignment can be found inside:
        https://biopython.org/docs/1.75/api/Bio.pairwise2.html
    Information about PyMol alignment can be found inside:
        https://pymolwiki.org/index.php/Align

    Returns:
        List of SharedResidue objects containing information about the shared residues.

    Raises:
        ValueError if chain 1 and 2 do not have the same excluded region.
    """
    # Define the custom sequence aligner
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.open_gap_score = open_penalty
    aligner.extend_gap_score = extend_penalty
    aligner.substitution_matrix = Align.substitution_matrices.load(subs_matrix)

    chain_1_sequence = get_chain_sequence(chain_1)
    chain_2_sequence = get_chain_sequence(chain_2)
    alignment = aligner.align(chain_1_sequence, chain_2_sequence)[0]
    seq_1, seq_2 = alignment[0], alignment[1]

    residues_1 = list(chain_1.get_residues())
    residues_2 = list(chain_2.get_residues())

    if chain_1_exclude_region is None:
        chain_1_exclude_region = (-1, -1)
    if chain_2_exclude_region is None:
        chain_2_exclude_region = (-1, -1)
    chain_1_start, chain_1_end = chain_1_exclude_region
    chain_2_start, chain_2_end = chain_2_exclude_region
    # Maintain the correct indexing of each residue
    res_idx_1, res_idx_2 = 0, 0

    # Construct the paired chain
    shared_residue_info = []
    for res_1, res_2 in zip(seq_1, seq_2):
        # If the chains do not contain the same residues, ignore
        if res_1 == "-" or res_2 == "-":
            res_idx_1 += res_1 != "-"
            res_idx_2 += res_2 != "-"
            continue

        # If residue index 1 is in diffused region 1
        # (between chain_1_start and chain_1_end),
        # it should be skipped for alignment.
        if chain_1_start <= res_idx_1 <= chain_1_end:
            # Check also the residue index 2 is in diffused region 2.
            if not chain_2_start <= res_idx_2 <= chain_2_end:
                raise ValueError("Chain 1 and 2 should have the same excluded region.")
            res_idx_1 += 1
            res_idx_2 += 1
            continue

        # Get the atoms of the current residue
        atoms_1 = list(filter(atom_filter, residues_1[res_idx_1].get_atoms()))
        atoms_2 = list(filter(atom_filter, residues_2[res_idx_2].get_atoms()))

        # Get all keys with the same names
        name_atom_dict_1 = {atom.name: atom for atom in atoms_1}
        name_atom_dict_2 = {atom.name: atom for atom in atoms_2}
        comparable_atoms = sorted(name_atom_dict_1.keys() & name_atom_dict_2.keys())

        shared_atoms_1 = [name_atom_dict_1[name] for name in comparable_atoms]
        shared_atoms_2 = [name_atom_dict_2[name] for name in comparable_atoms]

        res_idx_1 += 1
        res_idx_2 += 1

        shared_residue_info.append(
            SharedResidue(
                name=res_1,
                res_idx_1=res_idx_1,
                res_idx_2=res_idx_2,
                atoms_1=shared_atoms_1,
                atoms_2=shared_atoms_2,
            )
        )

    return shared_residue_info


def unroll_shared_residues(
    shared_residues_info: list[SharedResidue],
) -> tuple[list[Atom], list[Atom]]:
    """Unroll a list of shared residues into a tuple of independent list of atoms.

    Args:
        shared_residues_info: List of SharedResidue objects containing all shared
        residues and their atoms.

    Returns:
        Tuple containing a list of Atoms for each chain in SharedResidue.
    """
    chain_atoms_1, chain_atoms_2 = [], []
    for shared_info in shared_residues_info:
        chain_atoms_1.extend(shared_info.atoms_1)
        chain_atoms_2.extend(shared_info.atoms_2)
    return chain_atoms_1, chain_atoms_2
