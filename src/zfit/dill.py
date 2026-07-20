from __future__ import annotations

import functools as _functools
import io
import typing

import dill as __dill

from dill import *  # noqa: F403

if typing.TYPE_CHECKING:
    import zfit  # noqa: F401

_NONE = object()


class ZfitDillDumpError(Exception):
    pass


class ZfitDillLoadError(Exception):
    pass


def __retry_with_gc(func: typing.Callable, kwargs: dict, *, max_retries: int | None = None) -> typing.Any:
    pass


@_functools.wraps(__dill.dumps)
def dumps(
    obj: typing.Any,
    protocol: typing.Any = None,
    byref: bool = False,
    fmode: typing.Any = None,
    recurse: typing.Any = None,
    *,
    max_retries: int | bool | None = None,
    verify: bool | None = None,
    **kwds,
) -> bytes:
    pass


dumps.__doc__ = dumps.__doc__.format(docstring=__dill.dumps.__doc__)


@_functools.wraps(__dill.dump)
def dump(
    obj: typing.Any,
    file: io.IOBase,
    protocol: typing.Any = None,
    byref: typing.Any = None,
    fmode: typing.Any = None,
    recurse: typing.Any = None,
    *,
    max_retries: int | bool | None = None,
    verify: bool | None = None,
    **kwds,
) -> typing.Any:
    pass


def __retry_with_graphclear(
    func: typing.Callable, kwargs: dict, max_retries: int, _file_to_reset: io.IOBase | None = None
) -> typing.Any:
    original_error = None

    initial_position = None
    if _file_to_reset is not None:
        initial_position = _file_to_reset.tell()
    for i in range(max_retries + 1):
        if _file_to_reset is not None:
            _file_to_reset.seek(initial_position)
        try:
            out = func(**kwargs)
        except io.UnsupportedOperation as error:
            if "read" in str(error):
                msg = (
                    "Tried to verify (use `verify=False` to not verify) dumping by loading but failed (most likely because the file was opened in write mode only. "
                    "Try to open it in write and read mode, for example by changing `wb` to `w+b`.)"
                )
                raise io.UnsupportedOperation(msg) from error
        except Exception as error:
            if original_error is None:
                original_error = error
            if i == max_retries:
                msg = (
                    f"Max retries reached when loading {kwargs}, error still occurred. Original error {original_error}"
                )
                raise ZfitDillLoadError(msg) from error
            from zfit import run  # noqa: PLC0415

            run.clear_graph_cache(call_gc=True)
        else:
            break
    return out


@_functools.wraps(__dill.loads)
def loads(str: bytes, *, max_retries: int | bool | None = None, **kwds) -> typing.Any:
    """Wrapper around :py:func`dill.loads`that helps loading zfit objects as it retries with graph clearing if
    necessary.

    .. warning ::

        This function *may* clears the cached graph/traced functions. This should not have an effect on the
        results but may significantly slow down subsequent calls to the same zfit fits, as the graph needs to be
        recompiled.

    Additional argument max_retries: Maximum number of retries if it fails (can occur due to garbage collector required to run first).
        If None, defaults to 2.

    Original docstring:
    {docstring}
    """
    if max_retries is None:
        max_retries = 2
    elif max_retries < 0:
        msg = "max_retries has to be >= 0"
        raise ValueError(msg)
    kwargs = dict(str=str, **kwds)
    return __retry_with_graphclear(func=__dill.loads, kwargs=kwargs, max_retries=max_retries)


@_functools.wraps(__dill.load)
def load(file: io.IOBase, *, max_retries: int | bool | None = None, **kwds) -> typing.Any:
    pass


dump.__doc__ = dump.__doc__.format(docstring=__dill.dump.__doc__)
