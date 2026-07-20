
from __future__ import annotations

import contextlib
import functools
import math as _mt
import typing
import warnings
from collections import Counter, defaultdict, deque
from collections.abc import Callable
from typing import Any
from weakref import WeakSet

import numpy as np
import tensorflow as tf

import zfit.z.numpy as znp

from ..settings import run, ztypes
from ..util.warnings import warn_advanced_feature

if typing.TYPE_CHECKING:
    import zfit  # noqa: F401


def constant(value, dtype=ztypes.float, shape=None, name=None, verify_shape=None):
    """Create a constant tensor.

    Args:
        value: The constant value.
        dtype: The data type. Defaults to ztypes.float.
        shape: Shape of the tensor.
        name: Name for the operation. Defaults to "Const" if None.
        verify_shape: Shape verification (deprecated).
    """
    del verify_shape
    if name is None:
        name = "Const"
    return tf.constant(value, dtype=dtype, shape=shape, name=name)


pi = np.float64(_mt.pi)


def to_complex(number, dtype=None):
    """Convert number to complex tensor.

    Args:
        number: Number to convert.
        dtype: Complex data type. Defaults to ztypes.complex if None.
    """
    if dtype is None:
        dtype = ztypes.complex
    return znp.asarray(number, dtype=dtype)


def to_real(x, dtype=None):
    """Convert to real tensor.

    Args:
        x: Value to convert.
        dtype: Real data type. Defaults to ztypes.float if None.
    """
    if dtype is None:
        dtype = ztypes.float
    return znp.asarray(x, dtype=dtype)




def nth_pow(x, n):
    pass


def unstack_x(
    value: Any,
    num: Any = None,
    axis: int = -1,
    always_list: bool = False,
    name: str = "unstack_x",
):
    """Unstack a Data object and return a list of (or a single) tensors in the right order.

    Args:
        value:
        num:
        axis:
        always_list: If True, also return a list if only one element.
        name:

    Returns:
        Union[List[tensorflow.python.framework.ops.Tensor], tensorflow.python.framework.ops.Tensor, None]:
    """
    if isinstance(value, list):
        if len(value) == 1 and not always_list:
            value = value[0]
        return value
    try:
        return value.unstack_x(always_list=always_list)
    except AttributeError:
        unstacked_x = tf.unstack(value=value, num=num, axis=axis, name=name)
    if len(unstacked_x) == 1 and not always_list:
        assert isinstance(unstacked_x, list), (
            "unstacked_x has to be a list, otherwise this is a bug. Please report on github: "
            "https://github.com/zfit/zfit/issues/new?assignees=&labels=bug&projects=&template="
            "bug_report.md&title=[ASSERT]%20unstack_x%20does%20not%20provide%20a%20list,%20internal%20error"
        )
        unstacked_x = unstacked_x[0]
    return unstacked_x


def stack_x(values, axis: int = -1, name: str | None = None):
    pass




def convert_to_tensor(value, dtype=None, name=None, preferred_dtype=None):
    return tf.convert_to_tensor(value=value, dtype=dtype, name=name, dtype_hint=preferred_dtype)


def safe_where(
    condition: tf.Tensor,
    func: Callable,
    safe_func: Callable,
    values: tf.Tensor,
    value_safer: Callable = tf.ones_like,
) -> tf.Tensor:
    """Like :py:func:`tf.where` but fixes gradient `NaN` if func produces `NaN` with certain `values`.

    Args:
        condition: Same argument as to :py:func:`tf.where`, a boolean :py:class:`tf.Tensor`
        func: Function taking `values` as argument and returning the tensor _in case
            condition is True_. Equivalent `x` of :py:func:`tf.where` but as function.
        safe_func: Function taking `values` as argument and returning the tensor
            _in case the condition is False_, Equivalent `y` of :py:func:`tf.where` but as function.
        values: Values to be evaluated either by `func` or `safe_func` depending on
            `condition`.
        value_safer: Function taking `values` as arguments and returns "safe" values
            that won't cause troubles when given to`func` or by taking the gradient with respect
            to `func(value_safer(values))`.

    Returns:
        :py:class:`tf.Tensor`:
    """
    safe_x = tf.where(condition=condition, x=values, y=value_safer(values))
    return tf.where(condition=condition, x=func(safe_x), y=safe_func(values))




class DoNotCompile(Exception):
    pass


DEFAULT_XLAJIT_KWARGS = {"autograph": False, "reduce_retracing": True, "jit_compile": True}
DEFAULT_NOXLAJIT_KWARGS = {"autograph": False, "reduce_retracing": True, "jit_compile": False}


class FunctionWrapperRegistry:
    registries = WeakSet()
    allow_jit = True
    DEFAULT_CACHE_SIZE = 40
    _DEFAULT_DO_JIT_TYPES: typing.ClassVar = defaultdict(lambda: True)
    _DEFAULT_DO_JIT_TYPES.update(
        {
            None: True,
            "model": False,
            "loss": True,
            "sample": True,
            "model_sampling": True,
            "zfit_tensor": True,
            "tensor": True,
        }
    )
    DEFAULT_TF_FUNCTION_KWARGS: typing.ClassVar = {
        "model": DEFAULT_NOXLAJIT_KWARGS.copy(),
        "loss": DEFAULT_NOXLAJIT_KWARGS.copy(),
        "sample": DEFAULT_NOXLAJIT_KWARGS.copy(),
        "model_sampling": DEFAULT_NOXLAJIT_KWARGS.copy(),
        "zfit_tensor": DEFAULT_NOXLAJIT_KWARGS.copy(),
        "tensor": DEFAULT_NOXLAJIT_KWARGS.copy(),
    }

    do_jit_types = _DEFAULT_DO_JIT_TYPES.copy()

    def __init__(
        self,
        wraps=None,
        *,
        stateless_args=None,
        cachesize=None,
        keepalive=None,
        force_eager=None,
        **kwargs_user,
    ) -> None:
        """`tf.function`-like decorator with additional cache-invalidation functionality.

        Args:
            wraps: name of the function/type that is wrapped. Can be used to toggle specific types to be jitted or not.
            stateless_args: If true, assume that all arguments are stateless, i.e. aren't `zfit.Parameters` or similar.
            force_eager: Forces the execution of the function (and functions that have called this function) to be
                executed eagerly, that is, Python-like (and not jitted).
                This can significantly reduce the performance, however, normal
                Python syntax and control flow can be used also on values.
            cachesize: OLD
            keepalive: OLD
            **kwargs_user: arguments to `tf.function`
        """
        super().__init__()
        if cachesize is None:
            cachesize = self.DEFAULT_CACHE_SIZE
        if stateless_args is None:
            stateless_args = False
        if keepalive is None:
            keepalive = True
        self._initial_user_kwargs = kwargs_user
        self._deleted_cachers = Counter()

        self.registries.add(self)  # TODO: remove?
        self.python_func = None
        self.wrapped_func = None
        self.force_eager = force_eager if force_eager is not None else False

        if wraps not in self.do_jit_types:
            from ..settings import run  # noqa: PLC0415

            self.do_jit_types[wraps] = bool(run.get_graph_mode())
        self.wraps = wraps
        self.stateless_args = stateless_args

        self.function_cache = deque()
        self.reset()
        self.currently_traced = set()
        self.cachesize = cachesize
        self.keepalive = keepalive


    def reset(self, **_):
        self.function_cache.clear()

    def set_graph_cache_size(self, cachesize: int | None = None):
        pass


    def __call__(self, func):
        keepalive = self.keepalive
        wrapped_func = self.tf_function(func)
        cache = self.function_cache
        deleted_cachers = self._deleted_cachers
        from ..util.cache import FunctionCacheHolder  # noqa: PLC0415


        return concrete_func


def function(func=None, *, stateless_args=None, cachesize=None, **kwargs):
    """JIT/Graph compilation of functions, `tf.function`-like with additional cache-invalidation functionality.

    Args:
        func: Function to be compiled.
        stateless_args: If True, the function is assumed to be stateless and *does not depend on the name of tf.Variables*
            but only needs the values of the variables. This is not the case for taking gradients, for example.
        cachesize: Size of the cache. If None, the default size is used.
        **kwargs: arguments to `tf.function`

    Returns:
    """

    if stateless_args is None:
        stateless_args = False
    if callable(func):
        wrapper = FunctionWrapperRegistry(cachesize=cachesize, stateless_args=stateless_args, **kwargs)
        return wrapper(func)
    if func:
        msg = "All argument have to be key-word only. `func` must not be used"
        raise ValueError(msg)

    return FunctionWrapperRegistry(**kwargs, cachesize=cachesize, stateless_args=stateless_args)


