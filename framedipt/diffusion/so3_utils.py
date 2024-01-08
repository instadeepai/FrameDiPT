"""Module of SO3 utils."""
import torch


def skew_symmetric_matrix_from_axis_angle(axis_angle: torch.Tensor) -> torch.Tensor:
    """Skew-symmetric matrix from axis-angle representation,
     i.e. hat map from vector space R^3 to Lie algebra so(3).

    Source: https://en.wikipedia.org/wiki/Rotation_matrix.

    Args:
        axis_angle: axis-angle vector, shape [..., 3].

    Returns:
        skew_mat: skew-symmetric matrix, shape [..., 3, 3].
    """
    skew_mat = torch.zeros([*axis_angle.shape[:-1], 3, 3])
    skew_mat[..., 0, 1], skew_mat[..., 0, 2], skew_mat[..., 1, 2] = (
        -axis_angle[..., 2],
        axis_angle[..., 1],
        -axis_angle[..., 0],
    )
    return skew_mat + -skew_mat.transpose(-1, -2)


def axis_angle_from_skew_symmetric_matrix(skew_mat: torch.Tensor) -> torch.Tensor:
    """Axis-angle representation from skew-symmetric matrix,
     i.e. vee map from Lie algebra so(3) to the vector space R^3.

    Args:
        skew_mat: skew-symmetric matrix, shape [..., 3, 3].

    Returns:
        axis_angle: axis-angle vector, shape [..., 3].
    """
    if not torch.allclose(skew_mat, -skew_mat.transpose(-1, -2)):
        raise ValueError("Input mat must be skew symmetric.")
    axis_angle = torch.stack(
        [-skew_mat[..., 1, 2], skew_mat[..., 0, 2], -skew_mat[..., 0, 1]], dim=-1
    )
    return axis_angle


def axis_angle_from_rot_mat_by_log_map(rot_mat: torch.Tensor) -> torch.Tensor:
    """Logarithmic map from SO(3) to R^3 (i.e. rotation vector).

    Args:
        rot_mat: rotation matrix, shape [..., 3, 3].

    Returns:
        rotation vector (axis-angle representation), shape [..., 3].
    """
    shape = list(rot_mat.shape[:-2])
    rot_mat_ = rot_mat.reshape([-1, 3, 3])
    axis_angle = rotation_vector_from_matrix(rot_mat_)
    return axis_angle.reshape([*shape, 3])


def log_matrix(rot_mat: torch.Tensor) -> torch.Tensor:
    """logarithmic map from SO(3) to so(3), this is the matrix logarithm.

    Source: https://en.wikipedia.org/wiki/Logarithm_of_a_matrix.

    Args:
        rot_mat: rotation matrix, shape [..., 3, 3].

    Returns:
        logarithm matrix, shape [..., 3, 3].
    """
    return skew_symmetric_matrix_from_axis_angle(
        axis_angle_from_rot_mat_by_log_map(rot_mat)
    )


def exp_matrix(rot_mat: torch.Tensor) -> torch.Tensor:
    """Exponential map from so(3) to SO(3), this is the matrix exponential.

    Source: https://en.wikipedia.org/wiki/Matrix_exponential.

    Args:
        rot_mat: rotation matrix, shape [..., 3, 3].

    Returns:
        exponential matrix, shape [..., 3, 3].
    """
    return torch.linalg.matrix_exp(rot_mat)


def rot_mat_from_axis_angle_by_exp_map(axis_angle: torch.Tensor) -> torch.Tensor:
    """Exponential map from R^3 to SO(3).

    Args:
        axis_angle: axis-angle vector, shape [..., 3].

    Returns:
        exponential matrix, shape [..., 3, 3].
    """
    return exp_matrix(skew_symmetric_matrix_from_axis_angle(axis_angle))


def omega(rot_mat: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Angle of rotation SO(3) to R^+, this is the norm in our chosen orthonormal basis.

    Source: https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation.

    Args:
        rot_mat: rotation matrix, shape [..., 3, 3].
        eps: multiplying by (1 - eps) to prevent instability of arccos
         when provided with -1 or 1. Default to `1e-4`.

    Returns:
        rotation angle, shape [...].
    """
    trace = torch.diagonal(rot_mat, dim1=-2, dim2=-1).sum(dim=-1) * (1 - eps)
    return torch.arccos((trace - 1) / 2)


### New Log map adapted from geomstats
def rotation_vector_from_matrix(rot_mat: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix (in 3D) to rotation vector (axis-angle).

    # Adapted from geomstats
    # https://github.com/geomstats/geomstats/blob/master/geomstats/geometry/special_orthogonal.py#L884

    Get the angle through the trace of the rotation matrix:
    The eigenvalues are:
    :math:`\\{1, \\cos(angle) + i \\sin(angle), \\cos(angle) - i \\sin(angle)\\}`
    so that:
    :math:`trace = 1 + 2 \\cos(angle), \\{-1 \\leq trace \\leq 3\\}`
    The rotation vector is the vector associated to the skew-symmetric
    matrix
    :math:`S_r = \frac{angle}{(2 * \\sin(angle) ) (R - R^T)}`

    For the edge case where the angle is close to pi,
    the rotation vector (up to sign) is derived by using the following
    equality (see the Axis-angle representation on Wikipedia):
    :math:`outer(r, r) = \frac{1}{2} (R + I_3)`
    In nD, the rotation vector stores the :math:`n(n-1)/2` values
    of the skew-symmetric matrix representing the rotation.

    Parameters
    ----------
    rot_mat : array-like, shape=[..., n, n]
        Rotation matrix.

    Returns
    -------
    regularized_rot_vec : array-like, shape=[..., 3]
        Rotation vector.
    """
    angle = omega(rot_mat)
    if len(angle.shape) != 1:
        raise ValueError("cannot handle vectorized Log map here.")
    n_rot_mats = len(angle)
    rot_mat_transpose = torch.transpose(rot_mat, -2, -1)
    rot_vec_not_pi = axis_angle_from_skew_symmetric_matrix(rot_mat - rot_mat_transpose)
    mask_0 = torch.isclose(angle, torch.tensor(0.0)).to(angle.dtype)
    mask_pi = torch.isclose(angle, torch.tensor(torch.pi), atol=1e-2).to(angle.dtype)
    mask_else = (1 - mask_0) * (1 - mask_pi)

    numerator = 0.5 * mask_0 + angle * mask_else
    denominator = (
        (1 - angle**2 / 6) * mask_0 + 2 * torch.sin(angle) * mask_else + mask_pi
    )

    rot_vec_not_pi = rot_vec_not_pi * numerator[..., None] / denominator[..., None]

    vector_outer = 0.5 * (torch.eye(3) + rot_mat)
    vector_outer = vector_outer + (
        torch.maximum(torch.tensor(0.0), vector_outer) - vector_outer
    ) * torch.eye(3)
    squared_diag_comp = torch.diagonal(vector_outer, dim1=-2, dim2=-1)
    diag_comp = torch.sqrt(squared_diag_comp)
    norm_line = torch.linalg.norm(vector_outer, dim=-1)
    max_line_index = torch.argmax(norm_line, dim=-1)
    selected_line = vector_outer[range(n_rot_mats), max_line_index]
    # want
    signs = torch.sign(selected_line)
    rot_vec_pi = angle[..., None] * signs * diag_comp

    rot_vec = rot_vec_not_pi + mask_pi[..., None] * rot_vec_pi
    return regularize(rot_vec)


def regularize(point: torch.Tensor) -> torch.Tensor:
    """Regularize a point to be in accordance with convention.
    In 3D, regularize the norm of the rotation vector,
    to be between 0 and pi, following the axis-angle
    representation's convention.
    If the angle is between pi and 2pi,
    the function computes its complementary in 2pi and
    inverts the direction of the rotation axis.
    Parameters

    # Adapted from geomstats
    # https://github.com/geomstats/geomstats/blob/master/geomstats/geometry/special_orthogonal.py#L884
    ----------
    point : array-like, shape=[...,3]
        Point.
    Returns
    -------
    regularized_point : array-like, shape=[..., 3]
        Regularized point.
    """
    theta = torch.linalg.norm(point, axis=-1)
    k = torch.floor(theta / 2.0 / torch.pi)

    # angle in [0;2pi)
    angle = theta - 2 * k * torch.pi

    # this avoids dividing by 0
    theta_eps = torch.where(torch.isclose(theta, torch.tensor(0.0)), 1.0, theta)

    # angle in [0, pi]
    normalized_angle = torch.where(angle <= torch.pi, angle, 2 * torch.pi - angle)
    norm_ratio = torch.where(
        torch.isclose(theta, torch.tensor(0.0)), 1.0, normalized_angle / theta_eps
    )

    # reverse sign if angle was greater than pi
    norm_ratio = torch.where(angle > torch.pi, -norm_ratio, norm_ratio)
    return torch.einsum("...,...i->...i", norm_ratio, point)
