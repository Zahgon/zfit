
from __future__ import annotations

import typing
from collections.abc import Callable, Iterable
from typing import NoReturn

import numdifftools
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from ..util.container import convert_to_container
from ..util.deprecation import deprecated
from ..util.exception import BreakingAPIChangeError
from . import numpy as znp
from .tools import _auto_upcast
from .zextension import convert_to_tensor

if typing.TYPE_CHECKING:
    import zfit


def poly_complex(*args, real_x=False) -> tf.Tensor:
    pass


def numerical_gradient(func: Callable, params: Iterable[zfit.Parameter]) -> tf.Tensor:
    """Calculate numerically the gradient of func() with respect to ``params``.

    Args:
        func: Function without arguments that depends on ``params``
        params: Parameters that ``func`` implicitly depends on and with respect to which the
            derivatives will be taken.

    Returns:
        Gradients
    """
    from ..core.parameter import assign_values  # noqa: PLC0415

    params = convert_to_container(params)


    param_vals = znp.array(params)
    param_vals = znp.atleast_1d(param_vals)


    grad_func = numdifftools.Gradient(wrapped_func, order=2, base_step=1e-1)
    gradient = tf.numpy_function(grad_func, inp=[param_vals], Tout=tf.float64)
    gradient = znp.atleast_1d(gradient)
    gradient.set_shape(param_vals.shape)
    with tf.control_dependencies([gradient]):
        assign_values(params, param_vals)
        return gradient


def numerical_value_gradient(func: Callable, params: Iterable[zfit.Parameter]) -> tuple[tf.Tensor, tf.Tensor]:
    """Calculate numerically the gradients of ``func()`` with respect to ``params``, also returns the value of
    ``func()``.

    Args:
        func: Function without arguments that depends on ``params``
        params: Parameters that ``func`` implicitly depends on and with respect to which the
            derivatives will be taken.

    Returns:
        Value, gradient
    """
    return func(), numerical_gradient(func, params)


deprecated(None, "Use `numerical_value_gradient` instead.")




def numerical_hessian(func: Callable | None, params: Iterable[zfit.Parameter], hessian=None) -> tf.Tensor:
    """Calculate numerically the hessian matrix of func with respect to ``params``.

    Args:
        func: Function without arguments that depends on ``params``
        params: Parameters that ``func`` implicitly depends on and with respect to which the
            derivatives will be taken.

    Returns:
        Hessian matrix
    """
    from ..core.parameter import assign_values  # noqa: PLC0415

    params = convert_to_container(params)


    nparams = len(params)
    param_vals = znp.array(params)

    if hessian == "diag":
        hesse_func = numdifftools.Hessdiag(
            wrapped_func,
            order=2,
            base_step=1e-1,
        )
    else:
        hesse_func = numdifftools.Hessian(
            wrapped_func,
            order=2,
            base_step=1e-1,
        )
    if tf.executing_eagerly():
        computed_hessian = convert_to_tensor(hesse_func(param_vals))
    else:
        computed_hessian = tf.numpy_function(hesse_func, inp=[param_vals], Tout=tf.float64)
    if hessian == "diag":
        computed_hessian.set_shape((nparams,))
    else:
        computed_hessian.set_shape((nparams, nparams))

    with tf.control_dependencies([computed_hessian]):
        assign_values(params, param_vals)
        return computed_hessian


def numerical_value_gradient_hessian(
    func: Callable | None,
    params: Iterable[zfit.Parameter],
    gradient: Callable | None = None,
    hessian: str | None = None,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    pass




def autodiff_gradient(func: Callable, params: Iterable[zfit.Parameter]) -> tf.Tensor:
    """Calculate using autodiff the gradients of ``func()`` wrt ``params``.

    Automatic differentiation (autodiff) is a way of retreiving the derivative of x wrt y. It works by consecutively
    applying the chain rule. All that is needed is that every operation knows its own derivative.
    TensorFlow implements this and anything using ``tf.*`` operations only can use this technique.

        Args:
            func: Function without arguments that depends on ``params``
            params: Parameters that ``func`` implicitly depends on and with respect to which the
                derivatives will be taken.

        Returns:
            Gradient
    """
    return autodiff_value_gradient(func, params)[1]


def _extract_tfparams(
    params: Iterable[zfit.Parameter] | zfit.Parameter,
) -> list[tf.Variable]:
    pass


def autodiff_value_gradient(func: Callable, params: Iterable[zfit.Parameter]) -> tuple[tf.Tensor, tf.Tensor]:
    """Calculate using autodiff the gradients of ``func()`` wrt ``params``; also return ``func()``.

    Automatic differentiation (autodiff) is a way of retreiving the derivative of x wrt y. It works by consecutively
    applying the chain rule. All that is needed is that every operation knows its own derivative.
    TensorFlow implements this and anything using ``tf.*`` operations only can use this technique.

        Args:
            func: Function without arguments that depends on ``params``
            params: Parameters that ``func`` implicitly depends on and with respect to which the
                derivatives will be taken.

        Returns:
            Value and gradient
    """
    with tf.GradientTape(
        persistent=False,  # needs to be persistent for a call from hessian.
        watch_accessed_variables=False,
    ) as tape:
        tape.watch(params)
        value = func()
    gradients = tape.gradient(value, sources=params)
    gradients = znp.asarray(gradients)
    return value, gradients




def autodiff_hessian(func: Callable, params: Iterable[zfit.Parameter], hessian=None) -> tf.Tensor:
    """Calculate using autodiff the hessian matrix of ``func()`` wrt ``params``.

    Automatic differentiation (autodiff) is a way of retrieving the derivative of x wrt y. It works by consecutively
    applying the chain rule. All that is needed is that every operation knows its own derivative.
    TensorFlow implements this and anything using ``tf.*`` operations only can use this technique.

        Args:
            func: Function without arguments that depends on ``params``
            params: Parameters that ``func`` implicitly depends on and with respect to which the
                derivatives will be taken.

        Returns:
            Hessian matrix
    """

    return automatic_value_gradient_hessian(func, params, hessian=hessian)[2]


def automatic_value_gradient_hessian(
    func: Callable | None = None,
    params: Iterable[zfit.Parameter] | None = None,
    value_grad_func=None,
    hessian=None,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Calculate using autodiff the gradients and hessian matrix of ``func()`` wrt ``params``; also return ``func()``.

    Automatic differentiation (autodiff) is a way of retrieving the derivative of x wrt y. It works by consecutively
    applying the chain rule. All that is needed is that every operation knows its own derivative.
    TensorFlow implements this and anything using ``tf.*`` operations only can use this technique.

        Args:
            func: Function without arguments that depends on ``params``
            params: Parameters that ``func`` implicitly depends on and with respect to which the
                derivatives will be taken.

        Returns:
            Value, gradient and hessian matrix
    """
    if params is None:
        msg = "Parameters have to be specified, are currently None."
        raise ValueError(msg)
    if func is None and value_grad_func is None:
        msg = "Either `func` or `value_grad_func` has to be specified."
        raise ValueError(msg)

    from .. import z  # noqa: PLC0415

    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
        tape.watch(params)
        if callable(value_grad_func):
            loss, gradients = value_grad_func(params)
        else:
            loss, gradients = autodiff_value_gradient(func=func, params=params)
        if hessian == "diag":
            gradients = tf.unstack(gradients)
    if hessian == "diag":
        computed_hessian = znp.stack(
            [tape.gradient(grad, sources=param) for param, grad in zip(params, gradients, strict=True)]
        )
    else:
        for usepfor in [True, False]:
            try:
                computed_hessian = z.convert_to_tensor(
                    tape.jacobian(
                        gradients,
                        sources=params,
                        experimental_use_pfor=usepfor,  # causes TF bug? Slow..
                    )
                )
            except ValueError as error:
                if "Encountered an exception while vectorizing the jacobian computation." in str(error):
                    continue
                raise
            else:
                break
    del tape
    return loss, gradients, computed_hessian






def log(x) -> tf.Tensor:
    x = _auto_upcast(x)
    return _auto_upcast(znp.log(x=x))


def weighted_quantile(x, quantiles, weights=None, side="middle") -> tf.Tensor:
    pass
