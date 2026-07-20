
from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    import zfit  # noqa: F401


def make_param_constructor(constructor):
    """Create a constructor for a parameter class avoiding the `NameAlreadyTakenError`.

    If the parameter already exists, it is returned instead of creating a new one.

    Args:
        constructor: Callable that creates the parameter.

    Returns:
        Callable that creates the parameter.
    """


    return param_constructor
