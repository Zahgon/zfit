
from __future__ import annotations

import typing
from typing import Any

if typing.TYPE_CHECKING:
    import zfit  # noqa: F401


import attr

__all__ = ["dataclass"]


def dataclass(cls: type[Any]) -> type[Any]:
    pass
