"""Module containing some atom filter functions."""
from Bio.PDB import Atom

from framedipt.protein import residue_constants


def is_backbone(atom: Atom) -> bool:
    """Check if a given atom belongs to the backbone.

    Args:
        atom: Atom object to be checked.

    Returns:
        True if the atom belongs to the backbone.
    """
    return atom.id in residue_constants.BACKBONE_ATOMS


def is_carbon_alpha(atom: Atom) -> bool:
    """Check if a given atom is a CA (carbon alpha).

    Args:
        atom: Atom object to be checked.

    Returns:
        True if the atom is a CA atom.
    """
    return atom.id == "CA"


def is_all_no_hydrogen(atom: Atom) -> bool:
    """Check if a given atom is not a hydrogen element.

    Args:
        atom: Atom object to be checked.

    Returns:
        True if the atom is not a hydrogen element.
    """
    return atom.element != "H"


def is_all_no_special(atom: Atom) -> bool:
    """Check if a given atom is not a special atom: HET, OXT.

    Args:
        atom: Atom object to be checked.

    Returns:
        True if the atom is not a special atom.
    """
    return atom.id not in ("OXT", "HET")


def is_any_atom(_: Atom) -> bool:
    """Identity function for atom filters.

    Args:
        _: Atom object to be checked.

    Returns:
        True.
    """
    return True
