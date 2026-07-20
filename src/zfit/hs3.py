
from __future__ import annotations

import typing
from typing import NoReturn

from .core.serialmixin import ZfitSerializable
from .serialization import Serializer
from .util.exception import WorkInProgressError
from .util.warnings import warn_experimental_feature

if typing.TYPE_CHECKING:
    import zfit  # noqa: F401
__all__ = ["dumps", "loads"]


@warn_experimental_feature
def dumps(obj: ZfitSerializable) -> str:
    pass


@warn_experimental_feature
def loads(string: str) -> ZfitSerializable:
    """Load a zfit object from a string representation in the HS3 format.

    .. warning::
        This is an experimental feature and the API might change in the future. DO NOT RELY ON THE OUTPUT FOR
        ANYTHING ELSE THAN TESTING.

    THIS FUNCTION DOESN'T YET ADHERE TO HS3 (but just as a proxy).

    |@doc:hs3.explain| The `HEP Statistics Serialization Standard <https://github.com/hep-statistics-serialization-standard/hep-statistics-serialization-standard>`_,
                   or in short, :math:`\text{HS}^3`, is a serialization format for statistical models.
                   It is a JSON/YAML-based serialization that is a
                   coordinated effort of the HEP community to standardize the serialization of statistical models. The standard
                   is still in development and is not yet finalized. This function is experimental and may change in the future. |@docend:hs3.explain|


    Args:
        string (str): The string representation of the object.

    Returns:
        ZfitSerializable: The object.
    """
    return Serializer.from_hs3(string)




