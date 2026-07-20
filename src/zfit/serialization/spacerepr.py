from __future__ import annotations

import typing
from typing import Literal

from pydantic.v1 import Field, root_validator, validator

from ..core.space import Space
from ..util.exception import WorkInProgressError
from .serializer import BaseRepr

if typing.TYPE_CHECKING:
    import zfit  # noqa: F401

NumericTyped = float | int

NameObsTyped = tuple[str] | str | None


class SpaceRepr(BaseRepr):
    _implementation = Space
    hs3_type: Literal["Space"] = Field("Space", alias="type")
    name: str
    lower: NumericTyped | None = Field(alias="min")
    upper: NumericTyped | None = Field(alias="max")
    binning: float | None = None  # TODO: binning





    def _to_orm(self, init) -> SpaceRepr._implementation:
        init["limits"] = init.pop("lower", None), init.pop("upper", None)
        init["obs"] = init.pop("name")
        return super()._to_orm(init)
