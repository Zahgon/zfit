

from __future__ import annotations

import collections
import typing
from collections.abc import Callable, Iterable, Mapping

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import zfit.z.numpy as znp
from zfit import z
from zfit._interfaces import ZfitModel, ZfitSpace

from ..settings import ztypes
from ..util import ztyping
from ..util.exception import AnalyticIntegralNotImplemented, WorkInProgressError
from .space import MultiSpace, convert_to_space, supports

if typing.TYPE_CHECKING:
    import zfit  # noqa: F401




def numeric_integrate():
    pass


def simpson(func, lower, upper, num_points=1001, dtype=None):
    pass




def mc_integrate(
    func: Callable,
    limits: ztyping.LimitsType,
    axes: ztyping.AxesTypeInput | None = None,
    x: ztyping.XType | None = None,
    n_axes: int | None = None,
    draws_per_dim: int = 40000,
    max_draws=800_000,
    tol: float = 1e-6,
    method: str | None = None,  # noqa: ARG001
    xfixed: Mapping | None = None,
    dtype: type = ztypes.float,
    mc_sampler: Callable = tfp.mcmc.sample_halton_sequence,  # noqa: ARG001
    importance_sampling: Callable | None = None,  # noqa: ARG001
    vectorizable=None,
) -> tf.Tensor:
    pass


def normalization_nograd(func, n_axes, batch_size, num_batches, dtype, space, x=None, shape_after=()):
    del x  # unused
    upper, lower = space.v1.limits
    lower = z.convert_to_tensor(lower, dtype=dtype)
    upper = z.convert_to_tensor(upper, dtype=dtype)


    def cond(batch_num, _):
        return batch_num < num_batches

    initial_mean = tf.constant(0, shape=shape_after, dtype=dtype)
    initial_body_args = (0, initial_mean)
    _, final_mean = tf.while_loop(
        cond=cond,
        body=body,
        loop_vars=initial_body_args,
        parallel_iterations=1,
        swap_memory=False,
        back_prop=True,
    )
    return final_mean






class AnalyticIntegral:
    def __init__(self, *args, **kwargs):
        """Hold analytic integrals and manage their dimensions, limits etc."""
        super().__init__(*args, **kwargs)
        self._integrals = collections.defaultdict(dict)

    def get_max_axes(self, limits: ztyping.LimitsType) -> tuple[int]:
        """Return the maximal available axes to integrate over analytically for given limits.

        Args:
            limits: The integral function will be able to integrate over this limits
            axes: The axes over which (or over a subset) it will integrate

        Returns:
            Tuple[int]:
        """
        if not isinstance(limits, ZfitSpace):
            msg = "`limits` have to be a `ZfitSpace`"
            raise TypeError(msg)

        return self._get_max_axes_limits(limits, out_of_axes=limits.axes)[0]  # only axes

    def _get_max_axes_limits(self, limits, out_of_axes):  # TODO: automatic caching? but most probably not relevant
        if out_of_axes:
            out_of_axes = frozenset(out_of_axes)
            implemented_axes = frozenset(d for d in self._integrals if d <= out_of_axes)
        else:
            implemented_axes = set(self._integrals.keys())
        implemented_axes = sorted(implemented_axes, key=len, reverse=True)  # iter through biggest first
        for axes in implemented_axes:
            limits_matched = [lim for lim, integ in self._integrals[axes].items() if integ.limits >= limits]

            if limits_matched:  # one or more integrals available
                return tuple(sorted(axes)), limits_matched
        return (), ()  # no integral available for this axes

    def get_max_integral(self, limits: ztyping.LimitsType, axes: ztyping.AxesTypeInput = None) -> None | Integral:
        """Return the integral over the ``limits`` with ``axes`` (or a subset of them).

        Args:
            limits:
            axes:

        Returns:
            Return a callable that integrated over the given limits.
        """
        limits = convert_to_space(limits=limits, axes=axes)

        axes, limits = self._get_max_axes_limits(limits=limits, out_of_axes=axes)
        axes = frozenset(axes)
        integrals = [self._integrals[axes][lim] for lim in limits]
        return max(integrals, key=lambda lim: lim.priority, default=None)

    def register(
        self,
        func: Callable,
        limits: ztyping.LimitsType,
        priority: int = 50,
        *,
        supports_norm: bool = False,
        supports_multiple_limits: bool = False,
    ) -> None:
        """Register an analytic integral.

        Args:
            func: The integral function. Takes 1 argument.
            axes: The axes over which to integrate
            limits: |@doc:pdf.integrate.limits| Limits of the integration. |@docend:pdf.integrate.limits|
                ``Limits`` can be None if ``func`` works for any
            possible limits
            priority: If two or more integrals can integrate over certain limits, the one with the higher
                priority is taken (usually around 0-100).
            supports_norm: If True, norm_range will (if needed) be given to ``func`` as an argument.
            supports_multiple_limits: If True, multiple limits may be given as an argument to ``func``.
        """

        if not isinstance(limits, ZfitSpace):
            msg = "Limits for registering an integral have to be `ZfitSpace`"
            raise TypeError(msg)
        axes = frozenset(limits.axes)

        func = supports(norm=supports_norm, multiple_limits=supports_multiple_limits)(func)
        limits = limits.with_axes(axes=tuple(sorted(limits.axes)))
        self._integrals[axes][limits] = Integral(func=func, limits=limits, priority=priority)  # TODO improve with

    def integrate(
        self,
        x: ztyping.XType | None,
        limits: ztyping.LimitsType,
        axes: ztyping.AxesTypeInput = None,
        norm: ztyping.LimitsType = None,
        model: ZfitModel = None,
        params: dict | None = None,
    ) -> ztyping.XType:
        """Integrate analytically over the axes if available.

        Args:
            x: If a partial integration is made, x are the value to be evaluated for the partial
                integrated function. If a full integration is performed, this should be `None`.
            limits: The limits to integrate
            axes: The dimensions to integrate over
            norm: |@doc:pdf.integrate.norm| Normalization of the integration.
               By default, this is the same as the default space of the PDF.
               ``False`` means no normalization and returns the unnormed integral. |@docend:pdf.integrate.norm|
            params: The parameters of the function


        Returns:
            Union[tf.Tensor, float]:

        Raises:
            AnalyticIntegralNotImplementedError: If the requested integral is not available.
        """
        if axes is None:
            axes = limits.axes
        axes = frozenset(axes)
        integral_holder = self._integrals.get(axes)
        if integral_holder is None:
            msg = f"Analytic integral is not available for axes {axes}"
            raise AnalyticIntegralNotImplemented(msg)
        integral_fn = self.get_max_integral(limits=limits)
        if integral_fn is None:
            msg = f"Integral is available for axes {axes}, but not for limits {limits}"
            raise AnalyticIntegralNotImplemented(msg)

        try:
            return integral_fn(x=x, limits=limits, norm=norm, params=params, model=model)
        except TypeError as error1:
            error1msg = str(error1)

        try:
            return integral_fn(limits=limits, norm=norm, params=params, model=model)
        except TypeError as error2:
            error2msg = str(error2)

        try:
            return integral_fn(x=x, limits=limits, norm_range=norm, params=params, model=model)
        except TypeError as error3:
            error3msg = str(error3) + "\n Also, make sure to change `norm_range` to `norm`."
        try:
            return integral_fn(limits=limits, norm_range=norm, params=params, model=model)
        except TypeError as error4:
            error4msg = str(error4) + "\n Also, make sure to change `norm_range` to `norm`."
        msg = (
            f"Could not integrate, maybe an TypeError in a custom integral function? One of these errors could help:"
            f" {error1msg}, {error2msg}, {error3msg}, {error4msg}. "
            f"Otherwise, or if it's hard to debug, lease fill a bug report."
        )
        raise AssertionError(msg)


class Integral:  # TODO analytic integral
    def __init__(self, func: Callable, limits: ZfitSpace, priority: int | float):
        """A lightweight holder for the integral function."""
        self.limits = limits
        self.integrate = func
        self.axes = limits.axes
        self.priority = priority

    def __call__(self, *args, **kwargs):
        return self.integrate(*args, **kwargs)


class Integration:
    def __init__(self, mc_sampler, draws_per_dim, tol, max_draws, draws_simpson):
        self.tol = tol
        self.max_draws = max_draws
        self.mc_sampler = mc_sampler
        self.draws_per_dim = draws_per_dim
        self.draws_simpson = draws_simpson
