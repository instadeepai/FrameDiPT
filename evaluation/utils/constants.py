"""Constants used in the evaluation pipeline."""

# Constants for TCR evaluation.
TCR_CHAINS = ["alpha", "beta"]
DIHEDRAL_ANGLES = ["phi", "psi", "omega"]

EVAL_METRICS = {
    "bb_rmsd": "Backbone RMSD per residue",
    "angle_error_phi": "Angle error phi",
    "angle_error_psi": "Angle error psi",
    "angle_error_omega": "Angle error omega",
    "signed_angle_error_phi": "Signed angle error phi",
    "signed_angle_error_psi": "Signed angle error psi",
    "signed_angle_error_omega": "Signed angle error omega",
    "asa_abs_error": "ASA absolute error",
    "asa_square_error": "ASA square error",
    "rsa_abs_error": "RSA absolute error",
    "rsa_square_error": "RSA square error",
    "sample_phi": "Sample angles phi",
    "sample_psi": "Sample angles psi",
    "sample_omega": "Sample angles omega",
    "gt_phi": "Ground truth angles phi",
    "gt_psi": "Ground truth angles psi",
    "gt_omega": "Ground truth angles omega",
}

# xticks for plotting metrics per residue.
# We consider the left side and right side 4 residues,
# and all middle residues' metrics are averaged, represented by 5.
XTICKS = [str(idx) for idx in [1, 2, 3, 4, 5, -4, -3, -2, -1]]

SAMPLE_SELECTION_STRATEGY = ["mean", "median", "mode", "mean_closest", "median_closest"]
