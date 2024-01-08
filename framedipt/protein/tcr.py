"""Utils for processing TCR structures."""
from __future__ import annotations

import anarci
import numpy as np

from framedipt.data import utils as data_utils

# Copied from TCR repo.
# -- CDR loop residue limits in a given TCR chain.
#    Check this page for more information:
#        https://www.imgt.org/IMGTScientificChart/Numbering/IMGTIGVLsuperfamily.html
#    And this article for CDR2.5:
#        https://www.nature.com/articles/nature22383
CDR_RES_LIMITS: dict[str, tuple[int, int]] = {
    "CDR1": (27, 38),
    "CDR2": (56, 65),
    "CDR2.5": (81, 86),
    "CDR3": (105, 117),
}


def get_cdr_loop_seq(seq: str, cdr_loop_id: str, clean: bool = True) -> str:
    """Retrieve a desired CDR loop sequence from a TCR sequence based.

    CDR loops can be numbered with the IMGT scheme and are placed within the limits:
    CDR1 - (27,38), CDR2 - (56,65), CDR2.5 - (81-86), CDR3 - (105-117).
    Check this page for more information:
        https://www.imgt.org/IMGTScientificChart/Numbering/IMGTIGVLsuperfamily.html
    And this article for CDR2.5:
        https://www.nature.com/articles/nature22383

    Args:
        seq: TCR sequence.
        cdr_loop_id: string identifying the loop: ["CDR1", "CDR2", "CDR2.5", "CDR3"].
        clean: whether to clean the CDR sequence by removing "-" and " " characters.

    Returns:
        Desired CDR loop sequence.
    """
    if cdr_loop_id not in CDR_RES_LIMITS:
        raise ValueError(f"{cdr_loop_id=} must be one of {list(CDR_RES_LIMITS.keys())}")
    # Returned numbering is a list containing lists of domains for each input sequence.
    # For each input sequence, a list of domains may be found.
    # A domain is represented by a list of numbering, start and end indexes.
    # Example:
    # For sequence "EVQLQQSGAEVVRSGASVKLSCTASGFNIKDYYIHWVKQRPEKGLEWIGWI"
    # "DPEIGDTEYVPKFQGKATMTADTSSNTAYLQLSSLTSEDTAVYYCNAGHDYDRGRFPYWGQGTL"
    # "VTVSAAKTTPPSVYPLAPGSAAQTNSMVTLGCLVKGYFPEPVTVTWNSGSLSSGVHTFPAVLQS"
    # "DLYTLSSSVTVPSSTWPSETVTCNVAHPASSTKVDKKIVPRD",
    # The result is
    # [[([((1, ' '), 'E'), ((2, ' '), 'V'), ((3, ' '), 'Q'), ...], 0, 119)]].
    numbering, _, _ = anarci.anarci([("seq1", seq)], scheme="imgt", output=False)

    # seq_cut_n is the numbering of the first domain found in seq.
    # For the above example, seq_cut_n is
    # [((1, ' '), 'E'), ((2, ' '), 'V'), ((3, ' '), 'Q'), ...].
    # It contains the residue indexes and names of the first domain.
    seq_cut_n = numbering[0][0][0]
    llim, ulim = CDR_RES_LIMITS[cdr_loop_id]
    cdr = "".join([t[1] for t in seq_cut_n if t[0][0] >= llim and t[0][0] <= ulim])
    return cdr.replace(" ", "").replace("-", "") if clean else cdr


def create_diffusion_mask(
    chain_indexes: np.ndarray,
    aatype: np.ndarray,
    tcr_chains: list[str],
    cdr_loops: list[str],
    shifted_region: str | None = None,
) -> np.ndarray:
    """Create diffusion mask for TCR complexes.

    Args:
        chain_indexes: array of chain indexes in the structure, shape [N_res,].
        aatype: AA types in the structure, shape [N_res,].
        tcr_chains: list of chain ids for TCR alpha and beta chain, e.g. ["A", "B"].
        cdr_loops: list of strings identifying the loop.
            Should be in ["CDR1", "CDR2", "CDR2.5", "CDR3"].
        shifted_region: optional string in ["before", "after"], which takes
                the region before or after CDR loop, only applied for CDR3 loops.

    Returns:
        Array of diffusion mask which masks CDR3 loops
            in TCR alpha and beta chain, shape [N_res,].

    Raises:
        ValueError if any of cdr_loops not in ["CDR1", "CDR2", "CDR2.5", "CDR3"].
            or shifted_region is not None and not in ["before", "after"].
    """
    if any(cdr_loop_id not in CDR_RES_LIMITS for cdr_loop_id in cdr_loops):
        raise ValueError(
            f"CDR loops should be in {CDR_RES_LIMITS.keys()}, got {cdr_loops}."
        )

    if shifted_region is not None and shifted_region not in ["before", "after"]:
        raise ValueError(
            f"Shifted region should be either before or after, got {shifted_region}."
        )

    diffused_mask = np.zeros_like(chain_indexes)
    sorted_chain_ids = [chr(ord("A") + i) for i in range(len(tcr_chains))]
    for i, _ in enumerate(tcr_chains):
        tcr_chain_id = data_utils.chain_str_to_int(sorted_chain_ids[i])
        chain_mask = (chain_indexes == tcr_chain_id).astype(bool)
        chain_start_idx = np.where(chain_mask)[0][0]

        tcr_seq = data_utils.aatype_to_seq(aatype[chain_mask])
        for cdr_loop_id in cdr_loops:
            cdr_seq = get_cdr_loop_seq(tcr_seq, cdr_loop_id=cdr_loop_id)
            cdr_start_idx = tcr_seq.index(cdr_seq)
            if cdr_loop_id == "CDR3":
                if shifted_region == "before":
                    cdr_start_idx = cdr_start_idx - len(cdr_seq)
                elif shifted_region == "after":
                    cdr_start_idx = cdr_start_idx + len(cdr_seq)
            diffused_mask[
                chain_start_idx
                + cdr_start_idx : chain_start_idx
                + cdr_start_idx
                + len(cdr_seq)
            ] = 1

    return diffused_mask


def cut_tcr_sequence(seq: str) -> str:
    """Get variable domain from the full sequence.

    Example:
        For input sequence
            EVQLQQSGAEVVRSGASVKLSCTASGFNIKDYYIHWVKQRPEKGLEWIGWI
            DPEIGDTEYVPKFQGKATMTADTSSNTAYLQLSSLTSEDTAVYYCNAGHDYDRGRFPYWGQGTL
            VTVSAAKTTPPSVYPLAPGSAAQTNSMVTLGCLVKGYFPEPVTVTWNSGSLSSGVHTFPAVLQS
            DLYTLSSSVTVPSSTWPSETVTCNVAHPASSTKVDKKIVPRD
        the truncated sequence is
            EVQLQQSGAEVVRSGASVKLSCTASGFNIKDYYIHWVKQRPEKGLEWIGWI
            DPEIGDTEYVPKFQGKATMTADTSSNTAYLQLSSLTSEDTAVYYCNAGHDYDRGRFPYWGQGTL
            VTVSA

    For more details of anarci results, refer to comments in
        `framedipt.protein.tcr.get_cdr_loop_seq`.

    Args:
        seq: Query sequence.

    Returns:
        Sequence corresponding to a variable domain.

    Raises:
        ValueError if truncated sequence not found in query sequence.
    """
    # Identify antibody and TCR domains, number them
    # and annotate their germline and species.
    # For each input tuple it returns three lists.
    # Numbered, Alignment_details and Hit_tables.
    numbering, _, _ = anarci.anarci([("seq1", seq)], scheme="imgt", output=False)

    cut_n = numbering[0][0][0]
    seq_cut = "".join([t[1] for t in cut_n]).replace(" ", "").replace("-", "")

    if seq.find(seq_cut) == -1:
        raise ValueError(
            "Truncated sequence not found in query sequence, "
            "Please check the input query sequence."
        )

    return seq_cut
