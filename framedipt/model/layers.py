"""Module of model layers."""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Callable

import numpy as np
import torch
from scipy.stats import truncnorm


def _standardize(kernel: torch.Tensor) -> torch.Tensor:
    """
    Makes sure that Var(W) = 1 and E[W] = 0
    """
    eps = 1e-6

    axis = [0, 1] if len(kernel.shape) == 3 else 1

    var, mean = torch.var_mean(kernel, dim=axis, unbiased=True, keepdim=True)
    kernel = (kernel - mean) / (var + eps) ** 0.5
    return kernel


def he_orthogonal_init(tensor: torch.Tensor) -> torch.Tensor:
    """
    Generate a weight matrix with variance according to He initialization.
    Based on a random (semi-)orthogonal matrix neural networks
    are expected to learn better when features are decorrelated
    (stated by eg. "Reducing overfitting in deep networks
     by decorrelating representations",
    "Dropout: a simple way to prevent neural networks from overfitting",
    "Exact solutions to the nonlinear dynamics of learning
     in deep linear neural networks")
    """
    tensor = torch.nn.init.orthogonal_(tensor)

    fan_in = tensor.shape[:-1].numel() if len(tensor.shape) == 3 else tensor.shape[1]

    with torch.no_grad():
        tensor.data = _standardize(tensor.data)
        tensor.data *= (1 / fan_in) ** 0.5

    return tensor


class Dense(torch.nn.Module):
    """
    Combines dense layer and scaling for swish activation.

    Parameters
    ----------
        units: int
            Output embedding size.
        activation: str
            Name of the activation function to use.
        bias: bool
            True if use bias.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        activation: str | None = None,
    ) -> None:
        super().__init__()
        self._activation: torch.nn.Module

        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)
        self.reset_parameters()
        self.weight = self.linear.weight
        self.bias = self.linear.bias

        if isinstance(activation, str):
            activation = activation.lower()
        if activation in ["swish", "silu"]:
            self._activation = ScaledSiLU()
        elif activation is None:
            self._activation = torch.nn.Identity()
        else:
            raise NotImplementedError(
                "Activation function not implemented for GemNet (yet)."
            )

    def reset_parameters(self) -> None:
        he_orthogonal_init(self.linear.weight)
        if self.linear.bias is not None:
            self.linear.bias.data.fill_(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self._activation(x)
        return x


class ScaledSiLU(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale_factor = 1 / 0.6
        self._activation = torch.nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._activation(x) * self.scale_factor


class ResidualLayer(torch.nn.Module):
    """
    Residual block with output scaled by 1/sqrt(2).

    Parameters
    ----------
        units: int
            Output embedding size.
        nLayers: int
            Number of dense layers.
        activation: str
            Name of the activation function to use.
    """

    def __init__(
        self, units: int, n_layers: int = 2, activation: str | None = None
    ) -> None:
        super().__init__()
        self.dense_mlp = torch.nn.Sequential(
            *[
                Dense(units, units, activation=activation, bias=False)
                for i in range(n_layers)
            ]
        )
        self.inv_sqrt_2 = 1 / (2.0**0.5)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.dense_mlp(inputs)
        x = inputs + x
        x = x * self.inv_sqrt_2
        return x


class EfficientInteractionDownProjection(torch.nn.Module):
    """
    Down projection in the efficient reformulation.

    Parameters
    ----------
        num_spherical: int
            Same as the setting in the basis layers.
        num_radial: int
            Same as the setting in the basis layers.
        emb_size_interm: int
            Intermediate embedding size (down-projection size).
    """

    def __init__(
        self,
        num_spherical: int,
        num_radial: int,
        emb_size_interm: int,
    ) -> None:
        super().__init__()

        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.emb_size_interm = emb_size_interm

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.weight = torch.nn.Parameter(
            torch.empty((self.num_spherical, self.num_radial, self.emb_size_interm)),
            requires_grad=True,
        )
        he_orthogonal_init(self.weight)

    def forward(self, tbf: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
            (rbf_W1, sph): tuple
            - rbf_W1: Tensor, shape=(nEdges, emb_size_interm, num_spherical)
            - sph: Tensor, shape=(nEdges, Kmax, num_spherical)
        """
        rbf_env, sph = tbf
        # (num_spherical, nEdges, num_radial), (nEdges, Kmax, num_spherical);
        # Kmax = maximum number of neighbors of the edges

        # MatMul: mul + sum over num_radial
        rbf_w1 = torch.matmul(
            rbf_env, self.weight
        )  # (num_spherical, nEdges , emb_size_interm)
        rbf_w1 = rbf_w1.permute(1, 2, 0)  # (nEdges, emb_size_interm, num_spherical)

        sph = torch.transpose(sph, 1, 2)  # (nEdges, num_spherical, Kmax)
        return rbf_w1, sph


def permute_final_dims(tensor: torch.Tensor, inds: Sequence[int]) -> torch.Tensor:
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def flatten_final_dims(t: torch.Tensor, no_dims: int) -> torch.Tensor:
    return t.reshape(t.shape[:-no_dims] + (-1,))


def ipa_point_weights_init_(weights: torch.Tensor) -> None:
    with torch.no_grad():
        softplus_inverse_1 = 0.541324854612918
        weights.fill_(softplus_inverse_1)


def _calculate_fan(linear_weight_shape: torch.Size, fan: str = "fan_in") -> float | int:
    fan_out, fan_in, *_ = linear_weight_shape

    f: float | int
    if fan == "fan_in":
        f = fan_in
    elif fan == "fan_out":
        f = fan_out
    elif fan == "fan_avg":
        f = (fan_in + fan_out) / 2
    else:
        raise ValueError("Invalid fan option")

    return f


def trunc_normal_init_(
    weights: torch.Tensor, scale: float = 1.0, fan: str = "fan_in"
) -> None:
    shape = weights.shape
    f = _calculate_fan(shape, fan)
    scale = scale / max(1, f)
    a = -2
    b = 2
    std = math.sqrt(scale) / truncnorm.std(a=a, b=b, loc=0, scale=1)
    size = math.prod(shape)
    samples = truncnorm.rvs(a=a, b=b, loc=0, scale=std, size=size)
    samples = np.reshape(samples, shape)
    with torch.no_grad():
        weights.copy_(torch.tensor(samples, device=weights.device))


def lecun_normal_init_(weights: torch.Tensor) -> None:
    trunc_normal_init_(weights, scale=1.0)


def he_normal_init_(weights: torch.Tensor) -> None:
    trunc_normal_init_(weights, scale=2.0)


def glorot_uniform_init_(weights: torch.Tensor) -> None:
    torch.nn.init.xavier_uniform_(weights, gain=1)


def final_init_(weights: torch.Tensor) -> None:
    with torch.no_grad():
        weights.fill_(0.0)


def gating_init_(weights: torch.Tensor) -> None:
    with torch.no_grad():
        weights.fill_(0.0)


def normal_init_(weights: torch.Tensor) -> None:
    torch.nn.init.kaiming_normal_(weights, nonlinearity="linear")


class Linear(torch.nn.Linear):
    """
    A Linear layer with built-in nonstandard initializations. Called just
    like torch.nn.Linear.

    Implements the initializers in 1.11.4, plus some additional ones found
    in the code.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        init: str = "default",
        init_fn: Callable[[torch.Tensor, torch.Tensor], None] | None = None,
    ) -> None:
        """
        Args:
            in_dim:
                The final dimension of inputs to the layer
            out_dim:
                The final dimension of layer outputs
            bias:
                Whether to learn an additive bias. True by default
            init:
                The initializer to use. Choose from:

                "default": LeCun fan-in truncated normal initialization
                "relu": He initialization w/ truncated normal distribution
                "glorot": Fan-average Glorot uniform initialization
                "gating": Weights=0, Bias=1
                "normal": Normal initialization with std=1/sqrt(fan_in)
                "final": Weights=0, Bias=0

                Overridden by init_fn if the latter is not None.
            init_fn:
                A custom initializer taking weight and bias as inputs.
                Overrides init if not None.
        """
        super().__init__(in_dim, out_dim, bias=bias)

        if bias:
            with torch.no_grad():
                self.bias.fill_(0)

        if init_fn is not None:
            init_fn(self.weight, self.bias)
        elif init == "default":
            lecun_normal_init_(self.weight)
        elif init == "relu":
            he_normal_init_(self.weight)
        elif init == "glorot":
            glorot_uniform_init_(self.weight)
        elif init == "gating":
            gating_init_(self.weight)
            if bias:
                with torch.no_grad():
                    self.bias.fill_(1.0)
        elif init == "normal":
            normal_init_(self.weight)
        elif init == "final":
            final_init_(self.weight)
        else:
            raise ValueError("Invalid init string.")
