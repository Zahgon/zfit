
from __future__ import annotations

import typing
from typing import TypeVar

import numpy as np
import tensorflow as tf

if typing.TYPE_CHECKING:
    import zfit  # noqa: F401

__all__ = ["BoolTensor", "ComplexTensor", "IntTensor", "RealTensor", "StringTensor"]

tensor_like = (np.ndarray, tf.Tensor, tf.TensorSpec)

BoolTensor = TypeVar("BoolTensor", *tensor_like)

IntTensor = TypeVar("IntTensor", *tensor_like)

RealTensor = TypeVar("RealTensor", *tensor_like)

FloatTensor = TypeVar("FloatTensor", *tensor_like)

DoubleTensor = TypeVar("DoubleTensor", *tensor_like)

ComplexTensor = TypeVar("ComplexTensor", *tensor_like)

StringTensor = TypeVar("StringTensor", *tensor_like)
