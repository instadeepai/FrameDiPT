"""Module of custom types."""
from typing import TypeVar

import numpy as np
import torch

TensorNDArrayFloat = TypeVar("TensorNDArrayFloat", torch.Tensor, np.ndarray, float)
NDArrayFloat = TypeVar("NDArrayFloat", np.ndarray, float)
TensorNDArray = TypeVar("TensorNDArray", torch.Tensor, np.ndarray)
