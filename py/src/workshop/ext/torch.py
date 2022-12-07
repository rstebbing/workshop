##########################################
# File: torch.py                         #
# Copyright Richard Stebbing 2022.       #
# Distributed under the MIT License.     #
# (See accompany file LICENSE or copy at #
#  http://opensource.org/licenses/MIT)   #
##########################################

import numbers
import textwrap
import typing as t

import torch
from torch import nn
from torch import types as torch_types


_size_like = t.Union[int, torch_types._size]


def as_size(shape: _size_like):
    if isinstance(shape, torch.Size):
        return shape

    if isinstance(shape, (int, numbers.Integral)):
        shape = (shape,)

    shape = torch.Size(shape)

    return shape


def validate_tensor(tensor, shape: _size_like, dtype: torch_types._dtype, *, name: str = "tensor"):
    tensor = torch.as_tensor(tensor, dtype=dtype)
    shape = as_size(shape)

    if tensor.ndim < len(shape):
        raise ValueError(
            textwrap.dedent(
                f"""\
                {name}.ndim < len(shape)
                {name}.ndim = {tensor.ndim!r}
                {len(shape) = }"""
            )
        )

    last_dimensions = tensor.shape[-len(shape) :]
    if last_dimensions != shape:
        raise ValueError(
            textwrap.dedent(
                f"""\
                {name}.shape[-len(shape):] != shape
                {name}.shape[-len(shape):] = {last_dimensions!r}
                {shape = }"""
            )
        )

    return tensor


_dtype_like = t.Union[str, torch.dtype]


# (Originally retrieved via `torch.testing.get_all_dtypes()`.)
_DTYPES = [
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float32,
    torch.float64,
    torch.float16,
    torch.bfloat16,
    torch.bool,
    torch.complex64,
    torch.complex128,
]


_STR_TO_DTYPE = {str(dtype): dtype for dtype in _DTYPES}


def as_dtype(dtype: _dtype_like) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype

    dtype_: t.Optional[torch.dtype] = None
    if isinstance(dtype, str):
        dtype_ = _STR_TO_DTYPE.get(dtype)

    if dtype_ is None:
        raise ValueError(f"unsupported dtype\n{dtype = }")

    return dtype_


class RootMeanSquareLayerNorm(nn.Module):
    def __init__(self, normalized_shape: _size_like, eps=1e-6):
        super().__init__()

        normalized_shape = as_size(normalized_shape)

        self.normalized_shape = normalized_shape
        self.weight = nn.Parameter(torch.ones(normalized_shape))  # pyright: ignore[reportPrivateImportUsage]
        self.eps = eps

    def forward(self, input_):
        input_ = validate_tensor(input_, self.normalized_shape, torch.float32, name="input_")

        zero_mean_variance = input_.pow(2).mean(-1, keepdim=True)
        input_ = input_ * torch.rsqrt(zero_mean_variance + self.eps)
        input_ = self.weight * input_

        return input_
