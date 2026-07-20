from __future__ import annotations

import typing

from .core.constraint import (
    GaussianConstraint,
    LogNormalConstraint,
    PoissonConstraint,
    SimpleConstraint,
)
from .util import ztyping
from .util.deprecation import deprecated

if typing.TYPE_CHECKING:
    import zfit  # noqa: F401

__all__ = [
    "GaussianConstraint",
    "LogNormalConstraint",
    "PoissonConstraint",
    "SimpleConstraint",
]


@deprecated(None, "Use `GaussianConstraint` directly.")
def nll_gaussian(
    params: ztyping.ParamTypeInput,
    observation: ztyping.NumericalScalarType,
    uncertainty: ztyping.NumericalScalarType,
) -> GaussianConstraint:
    pass
