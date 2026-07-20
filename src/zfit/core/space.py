
from __future__ import annotations

import functools
import inspect
import itertools
import typing
import warnings
from abc import abstractmethod
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from contextlib import suppress

import numpy as np
import pandas as pd
import tensorflow as tf
from numpy import ndarray
from pandas import DataFrame
from tensorflow import Tensor

import zfit
import zfit.z.numpy as znp
from zfit._interfaces import (
    ZfitData,
    ZfitLimit,
    ZfitOrderableDimensional,
    ZfitPDF,
    ZfitSpace,
)

from .. import z
from ..settings import ztypes
from ..util import ztyping
from ..util.container import convert_to_container
from ..util.deprecation import deprecated, deprecated_args, deprecated_norm_range
from ..util.exception import (
    AxesIncompatibleError,
    AxesNotSpecifiedError,
    BreakingAPIChangeError,
    CannotConvertToNumpyError,
    CoordinatesIncompatibleError,
    CoordinatesUnderdefinedError,
    IllegalInGraphModeError,
    IntentionAmbiguousError,
    InvalidLimitSubspaceError,
    LimitsIncompatibleError,
    LimitsNotSpecifiedError,
    LimitsUnderdefinedError,
    MultipleLimitsNotImplemented,
    NormNotImplemented,
    NumberOfEventsIncompatibleError,
    ObsIncompatibleError,
    ObsNotSpecifiedError,
    OverdefinedError,
    ShapeIncompatibleError,
    SpaceIncompatibleError,
)
from .baseobject import BaseObject
from .coordinates import (
    Coordinates,
    _convert_obs_to_str,
    convert_to_axes,
    convert_to_obs_str,
)
from .dimension import common_axes, common_obs, limits_overlap
from .serialmixin import SerializableMixin

if typing.TYPE_CHECKING:
    import zfit


class LimitRangeDefinition:
    pass


class Any(LimitRangeDefinition):
    _singleton_instance = None

    def __new__(cls, *_, **__):
        instance = cls._singleton_instance
        if instance is None:
            instance = super().__new__(cls)
            cls._singleton_instance = instance

        return instance

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._singleton_instance = None  # each subclass is a singleton of "itself"

    def __repr__(self):
        return "<Any>"

    def __lt__(self, other):
        return True

    def __le__(self, other):
        return True

    def __ge__(self, other):
        return True

    def __gt__(self, other):
        return True


class AnyLower(Any):
    def __repr__(self):
        return "<Any Lower Limit>"


class AnyUpper(Any):
    def __repr__(self):
        return "<Any Upper Limit>"


ANY = Any()
ANY_LOWER = AnyLower()
ANY_UPPER = AnyUpper()


class V1Space:
    def __init__(self, space: ZfitSpace):
        self.space = space

    @property
    def lower(self) -> np.ndarray:
        return znp.atleast_1d(self.space.lower[0])






class V0Space:
    def __init__(self, space: ZfitSpace):
        self.space = space

    @property
    def lower(self):
        return self.space.lower






class VectorizeLimits:
    def __init__(self, space):
        self.space = space

    @property
    def lower(self):
        return self.space.v1.lower[None, :]









@z.function(wraps="tensor", keepalive=True)
def inside_rect_limits(x, rect_limits):
    if (ndims := x.get_shape().ndims) is not None and ndims <= 1:
        msg = (
            "x has ndims <= 1, which is most probably not wanted. The default shape for array-like"
            " structures is (nevents, n_obs)."
        )
        raise ValueError(msg)
    lower, upper = z.unstack_x(rect_limits, axis=0)
    lower = z.convert_to_tensor(lower)
    upper = z.convert_to_tensor(upper)
    below_upper = znp.all(znp.less_equal(x, upper), axis=-1)  # if all obs inside
    above_lower = znp.all(znp.greater_equal(x, lower), axis=-1)
    return znp.logical_and(above_lower, below_upper)




def convert_to_tensor_or_numpy(obj, dtype=ztypes.float):
    if contains_tensor(obj):
        return znp.asarray(obj, dtype=dtype)
    else:
        with suppress(AttributeError):
            dtype = dtype.as_numpy_dtype
        return np.array(obj, dtype=dtype)


def _sanitize_x_input(x, n_obs):
    if isinstance(x, ZfitData):
        x = x.value()
    x = z.convert_to_tensor(x)
    if not x.shape.ndims > 1 and n_obs > 1:
        msg = (
            "x has ndims <= 1, which is most probably not wanted. The default shape for array-like"
            " structures is (nevents, n_obs)."
        )
        raise ValueError(msg)
    if x.shape.ndims <= 1 and n_obs == 1:
        x = tf.broadcast_to(x, (1, 1)) if x.shape.ndims == 0 else znp.expand_dims(x, axis=-1)
    if tf.get_static_value(x.shape[-1]) != n_obs:
        msg = (
            f"n_obs ({n_obs}) and the last dim of x (shape: {x.shape}) do not agree. Assuming x has shape (..., n_obs)"
        )
        raise ShapeIncompatibleError(msg)
    return x


def is_range_definition(limit):
    if isinstance(limit, LimitRangeDefinition | ZfitSpace):
        return True
    elif (isinstance(limit, np.ndarray) and limit.dtype != object) or tf.is_tensor(limit):
        return False
    try:
        return any(is_range_definition(lim) for lim in limit)
    except TypeError:
        return False  # not iterable and was not a LimitRangeDefinition in the beginning


class Limit(
    ZfitLimit,
):
    _experimental_allow_vectors = False

    def __init__(
        self,
        limit_fn: ztyping.LimitsFuncTypeInput = None,
        rect_limits: ztyping.LimitsTypeInput = None,
        n_obs: int | None = None,
    ):
        """Specify a limit with rectangular limits (and possiblty an arbitrary function).

        Args:
            limit_fn: Function that works as ``inside``: return true if a point is inside of the limits.
                The function should take one tensor-like argument with shape (..., n_obs) and should return
                a shape without the last dimension.
            rect_limits: Rectangular limits, a tuple of tensor-like objects with shape (typically) (1, n_obs) or similar
                such as only a tuple/list of values that will be interpreted as the last dimension. They should cover an
                area that includes ``limit_fn`` fully.
            n_obs: dimensionality of the Limits, the last dimension.
        """
        super().__init__()
        (
            limit_fn,
            rect_limits,
            n_obs,
            is_rect,
            sublimits,
        ) = self._check_convert_input_limits(limit_fn=limit_fn, rect_limits=rect_limits, n_obs=n_obs)
        self._limit_fn = limit_fn
        self._rect_limits = rect_limits
        self._n_obs = n_obs
        self._is_rect = is_rect
        self._sublimits = sublimits

    def _check_convert_input_limits(self, limit_fn, rect_limits, n_obs):
        if isinstance(limit_fn, ZfitLimit):
            if not isinstance(limit_fn, Limit):  # because of the limit_fn, that is private. Maybe use `inside` instead?
                msg = "If limits_fn is an instance of ZfitLimit, it has to be an instance of Limit (currently)"
                raise TypeError(msg)
            if rect_limits is not None or n_obs != limit_fn.n_obs:
                msg = "limits_fn is a ZfitLimit. rect_limits and n_obs must not be specified(or n_obs coincide)."
                raise OverdefinedError(msg)
            limit = limit_fn

            limit_fn = limit.limit_fn
            rect_limits = limit.rect_limits
            n_obs = limit.n_obs
        limits_are_rect = True

        return_limits_short = False
        if limit_fn is False:
            if rect_limits in (False, None):
                limits_short = False
                return_limits_short = True
        elif limit_fn is None:
            if rect_limits is False:
                limits_short = False
                return_limits_short = True
            elif rect_limits is None:
                limits_short = None
                return_limits_short = True
            else:  # start from limits are anything, rect is None
                limit_fn = rect_limits
                rect_limits = None
        if return_limits_short:
            sublimits = [type(self)(limit_fn=limits_short, n_obs=1) for _ in range(n_obs)] if n_obs > 1 else (self,)
            return limits_short, limits_short, n_obs, limits_short, sublimits

        if not callable(limit_fn):  # limits_fn is actually rect_limits
            rect_limits = limit_fn
            limit_fn = None

        else:
            limits_are_rect = False
            if rect_limits in (None, False):
                msg = "Limits given as a function need also rect_limits, cannot be None or False"
                raise ValueError(msg)
        try:
            lower, upper = rect_limits
        except TypeError as err:
            msg = "The outermost shape of `rect_limits` has to be 2 to represent (lower, upper)."
            raise TypeError(msg) from err

        lower = self._sanitize_rect_limit(lower)
        upper = self._sanitize_rect_limit(upper)

        if not self._experimental_allow_vectors:
            lower_nevents = tf.get_static_value(lower.shape[0])
            upper_nevents = tf.get_static_value(upper.shape[0])
            if lower_nevents != 1 or upper_nevents != 1:
                msg = (
                    "Vectors (limits with n_events != 1) are not allowed. Experimental"
                    " flag (_experimental_allow_vectors) can be switched on if desired."
                    " This happened most likely due to the new Space limits layout:"
                    " To create multiple limits, use the addition operator of simple spaces."
                )
                raise LimitsIncompatibleError(msg)

        lower_nobs = tf.get_static_value(lower.shape[-1])

        if lower_nobs != (upper_nobs := tf.get_static_value(upper.shape[-1])):
            msg = f"Last dimension of lower ({lower_nobs}) and upper ({upper_nobs}) have to coincide."
            raise ShapeIncompatibleError(msg)
        if n_obs is not None and lower_nobs != n_obs:
            msg = f"Inferred last dimension ({lower_nobs}) does not coincide with given n_obs ({n_obs})"
            raise ShapeIncompatibleError(msg)

        if not any(is_range_definition(limit) for limit in (lower, upper)):
            tf.assert_greater(
                upper,
                lower,
                message="All upper limits have to be larger than the lower limits and are"
                " given as (lower, upper). Maybe (upper, lower) was entered?",
            )

        n_obs = lower_nobs  # in case it was None
        rect_limits = (lower, upper)


        sublimits = []
        if limits_are_rect and n_obs > 1:
            for i in range(n_obs):
                low = z.unstable.gather(lower, (i,), axis=-1)
                up = z.unstable.gather(upper, (i,), axis=-1)
                sublimits.append(type(self)(rect_limits=(low, up), n_obs=1))
        else:
            sublimits.append(self)

        sublimits = tuple(sublimits)

        return limit_fn, rect_limits, n_obs, limits_are_rect, sublimits

    @staticmethod
    def _sanitize_rect_limit(limit) -> ztyping.RectLowerReturnType:
        """Sanitize the input limit and return if it is numerical or not.

        Args:
            limit:

        Returns:
        """
        dtype = object if is_range_definition(limit) else ztypes.float  # as the above ANY
        limit = convert_to_tensor_or_numpy(limit, dtype=dtype)
        if len(limit.shape) == 0:
            limit = z.unstable.broadcast_to(limit, shape=(1, 1))
        if len(limit.shape) == 1:
            limit = z.unstable.expand_dims(limit, axis=0)
        return limit

    @property
    def has_rect_limits(self) -> bool:
        pass

    @property
    def limits(self) -> ztyping.LimitsReturnType:
        pass

    @property  # todo: remove, legacy object
    def rect_limits(self) -> ztyping.RectLimitsReturnType:
        pass


    @property
    def rect_limits_np(self) -> ztyping.RectLimitsNPReturnType:
        pass

    @property
    def rect_lower(self) -> ztyping.RectLowerReturnType:
        pass

    @property
    def rect_upper(self) -> ztyping.RectUpperReturnType:
        pass

    def rect_area(self) -> float | np.ndarray | znp.array:
        pass

    def inside(self, x: ztyping.XTypeInput, guarantee_limits: bool = False) -> ztyping.XTypeReturnNoData:
        """Test if `x` is inside the limits.

        This function should be used to test if values are inside the limits. If the given x is already inside
        the rectangular limits, e.g. because it was sampled from within them

        Args:
            x: Values to be checked whether they are inside of the limits. The shape is expected to have the last
                dimension equal to n_obs.
            guarantee_limits: Guarantee that the values are already inside the rectangular limits.

        Returns:
            Return a boolean tensor-like object with the same shape as the input `x` except of the
                last dimension removed.
        """
        x = _sanitize_x_input(x, n_obs=self.n_obs)
        if not self.has_limits:
            msg = "Cannot call `inside` without limits defined."
            raise LimitsNotSpecifiedError(msg)
        if guarantee_limits and self.has_rect_limits:
            return tf.broadcast_to(True, x.shape)
        else:
            return self._inside(x, guarantee_limits)

    def _inside(self, x, guarantee_limits):
        del guarantee_limits
        if self.has_rect_limits:
            return inside_rect_limits(x, rect_limits=self._rect_limits_tf)
        else:
            return self._limit_fn(x)

    def filter(
        self,
        x: ztyping.XTypeInput,
        guarantee_limits: bool = False,
        axis: int | None = None,
    ) -> ztyping.XTypeReturnNoData:
        pass



    @property
    def rect_limits_are_tensors(self) -> bool:
        pass

    @property
    def limits_are_set(self) -> bool:
        pass

    @property
    def limits_are_false(self) -> bool:
        pass

    @property
    def has_limits(self) -> bool:
        pass

    @property
    def n_obs(self) -> int:
        pass

    @property
    def n_events(self) -> int | None:
        pass

    def equal(self, other: object, allow_graph: bool = True) -> znp.array:
        """Compare the limits on equality. For ANY objects, this also returns true.

        If called inside a graph context *and* the limits are tensors, this will return a symbolic `tf.Tensor`.

        Args:
            other: Any other object to compare with
            allow_graph: If False and the function returns a symbolic tensor, raise IllegalInGraphModeError instead.

        Returns:
            A ``znp.array`` with the result of the comparison.
         Raises:
             IllegalInGraphModeError: if `allow_graph`
        """
        if not isinstance(other, ZfitLimit):
            return np.array(False)
        return equal_limits(self, other, allow_graph=allow_graph)

    def __eq__(self, other: object) -> bool:
        """Compares two Limits for equality without graph mode allowed.

        Returns:

        Raises:
             IllegalInGraphModeError: it the comparison happens with tensors in a graph context.
        """
        if not isinstance(other, ZfitLimit):
            return NotImplemented
        return self.equal(other, allow_graph=False)

    def less_equal(self, other: object, allow_graph: bool = True) -> znp.array:
        """Set-like comparison for compatibility. If an object is less_equal to another, the limits are combatible.

        This can be used to determine whether a fitting range specification can handle another limit.

        If called inside a graph context *and* the limits are tensors, this will return a symbolic `tf.Tensor`.


        Args:
            other: Any other object to compare with
            allow_graph: If False and the function returns a symbolic tensor, raise IllegalInGraphModeError instead.


        Returns:
            Result of the comparison
        Raises:
             IllegalInGraphModeError: it the comparison happens with tensors in a graph context.
        """
        if not isinstance(other, ZfitLimit):
            return np.array(False)
        return less_equal_limits(self, other, allow_graph=allow_graph)

    def __le__(self, other: object) -> bool:
        """Set-like comparison for compatibility. If an object is less_equal to another, the limits are combatible.

        This can be used to determine whether a fitting range specification can handle another limit.

        Returns:
            Result of the comparison
        Raises:
             IllegalInGraphModeError: it the comparison happens with tensors in a graph context.
        """
        if not isinstance(other, ZfitLimit):
            return NotImplemented
        return self.less_equal(other, allow_graph=False)

    def get_sublimits(self) -> Iterable[ZfitLimit]:
        """Splits itself into multiple sublimits with smaller n_obs.

        If this is not possible, if the limits are not rectangular, just returns itself.

        Returns:
            The sublimits if it was able to split.
        """
        return self._sublimits

    def __hash__(self) -> int:
        objects = (
            self._limit_fn,
            self.n_obs,
        )  # not rect limits, not hashable and unprecise
        return hash(tuple(objects))

    def __repr__(self) -> str:
        class_name = str(self.__class__).split(".")[-1].split("'")[0]
        if not self.limits_are_set:
            limits = None
        elif self.limits_are_false:
            limits = False
        elif self.n_obs < 5 and not self.n_events > 1:
            limits = self.v1.limits
        else:
            limits = "rectangular"

        return f"<zfit {class_name} rect_limits={limits}, limit_fn={not self.has_rect_limits}>"


def rect_limits_are_any(limit: ZfitLimit) -> bool:
    """True if all limits in limit are ANY objects."""
    if limit.rect_limits_are_tensors:
        return False
    return bool(all(isinstance(ele, Any) for lim in limit.rect_limits_np for ele in lim.flatten()))


def less_equal_limits(limit1: Limit, limit2: Limit, allow_graph=True) -> znp.array:
    if rect_limits_are_any(limit1) or rect_limits_are_any(limit2):
        return np.array(True)

    try:
        lower1, upper1 = limit1.rect_limits_np
        lower2, upper2 = limit2.rect_limits_np
    except CannotConvertToNumpyError as error:
        if not allow_graph:
            msg = (
                "Cannot use equality in graph mode, e.g. inside a `tf.function` decorated "
                "function. To retrieve a symbolic Tensor, use `.equal(..., allow_graph=True)`"
            )
            raise IllegalInGraphModeError(msg) from error

        lower1, upper1 = limit1.rect_limits
        lower2, upper2 = limit2.rect_limits

    lower_le = z.unstable.reduce_all(z.unstable.less_equal(lower1, lower2), axis=-1)
    upper_le = z.unstable.reduce_all(z.unstable.less_equal(upper1, upper2), axis=-1)
    rect_limits_le = z.unstable.logical_and(lower_le, upper_le)
    if not (limit1.has_rect_limits or limit2.has_rect_limits):
        funcs_equal = limit1.limit_fn == limit2.limit_fn

    elif not limit1.has_rect_limits and limit2.has_rect_limits:
        funcs_equal = np.array(True)
    else:
        funcs_equal = limit1.limit_fn == limit2.limit_fn
    return z.unstable.logical_and(rect_limits_le, funcs_equal)


def equal_limits(limit1: Limit, limit2: Limit, allow_graph=True) -> bool:
    if not (limit1.has_rect_limits or limit2.has_rect_limits):
        return np.array(limit1.limit_fn == limit2.limit_fn)

    elif limit1.has_rect_limits ^ limit2.has_rect_limits:
        return np.array(False)

    try:
        lower, upper = limit1.rect_limits_np
        lower_other, upper_other = limit2.rect_limits_np
    except CannotConvertToNumpyError as error:
        if not allow_graph:
            msg = (
                "Cannot use equality in graph mode, e.g. inside a `tf.function` decorated "
                "function. To retrieve a symbolic Tensor, use `.equal(..., allow_graph=True)`"
            )
            raise IllegalInGraphModeError(msg) from error

        lower, upper = limit1.rect_limits
        lower_other, upper_other = limit2.rect_limits

    lower_limits_equal = z.unstable.reduce_all(z.unstable.allclose_anyaware(lower, lower_other))
    upper_limits_equal = z.unstable.reduce_all(z.unstable.allclose_anyaware(upper, upper_other))
    rect_limits_equal = z.unstable.logical_and(lower_limits_equal, upper_limits_equal)
    funcs_equal = limit1.limit_fn == limit2.limit_fn
    return z.unstable.logical_and(rect_limits_equal, funcs_equal)




warned_new_limits = False




class BaseSpace(ZfitSpace, BaseObject):
    def __init__(self, obs, axes, name, **kwargs):
        super().__init__(name, **kwargs)
        coords = Coordinates(obs, axes)
        self.coords = coords


    def inside(self, x: ztyping.XTypeInput, guarantee_limits: bool = False) -> ztyping.XTypeReturn:
        """Test if `x` is inside the limits.

        This function should be used to test if values are inside the limits. If the given x is already inside
        the rectangular limits, e.g. because it was sampled from within them

        Args:
            x: Values to be checked whether they are inside of the limits. The shape is expected to have the last
                dimension equal to n_obs.
            guarantee_limits: Guarantee that the values are already inside the rectangular limits.

        Returns:
            Return a boolean tensor-like object with the same shape as the input `x` except of the
                last dimension removed.
        """
        x = _sanitize_x_input(x, n_obs=self.n_obs)
        if self.has_rect_limits and guarantee_limits:
            return tf.broadcast_to(True, x.shape)
        return self._inside(x, guarantee_limits)



    @abstractmethod
    def _inside(self, x, guarantee_limits):
        raise NotImplementedError

    def filter(
        self,
        x: ztyping.XTypeInput,
        guarantee_limits: bool = False,
        axis: int | None = None,
    ) -> ndarray | Tensor | DataFrame | Any:
        pass


    @property
    def n_obs(self) -> int:
        pass

    @property
    def obs(self) -> ztyping.ObsTypeReturn:
        pass

    @property
    def axes(self) -> ztyping.AxesTypeReturn:
        pass



    def __iter__(self) -> Iterable[ZfitSpace]:
        yield self

    def get_reorder_indices(self, obs: ztyping.ObsTypeInput = None, axes: ztyping.AxesTypeInput = None) -> tuple[int]:
        """Indices that would order the instances obs as `obs` respectively the instances axes as `axes`.

        Args:
            obs: Observables that the instances obs should be ordered to. Does not reorder, but just
                return the indices that could be used to reorder.
            axes: Axes that the instances obs should be ordered to. Does not reorder, but just
                return the indices that could be used to reorder.

        Returns:
            New indices that would reorder the instances obs to be obs respectively axes.

        Raises:
            CoordinatesUnderdefinedError: If neither `obs` nor `axes` is given
        """
        return self.coords.get_reorder_indices(obs=obs, axes=axes)

    def _check_convert_input_axes(
        self, axes: ztyping.AxesTypeInput, allow_none: bool = False
    ) -> ztyping.AxesTypeReturn:
        if axes is None:
            if allow_none:
                return None
            else:
                msg = "TODO: Cannot be None"
                raise AxesNotSpecifiedError(msg)
        return axes.axes if isinstance(axes, ZfitSpace) else convert_to_container(value=axes, container=tuple)

    def _check_convert_input_obs(self, obs: ztyping.ObsTypeInput, allow_none: bool = False) -> ztyping.ObsTypeReturn:
        """Input check: Convert `NOT_SPECIFIED` to None or check if obs are all strings.

        Args:
            obs:

        Returns:
        """
        if obs is None:
            if allow_none:
                return None
            else:
                msg = "TODO: Cannot be None"
                raise ObsNotSpecifiedError(msg)

        if isinstance(obs, ZfitSpace):
            obs = obs.obs
        else:
            obs = convert_to_container(obs, container=tuple)
            obs_not_str = tuple(o for o in obs if not isinstance(o, str))
            if obs_not_str:
                msg = f"The following observables are not strings: {obs_not_str}"
                raise ValueError(msg)
        return obs


    def __repr__(self):
        class_name = str(self.__class__).split(".")[-1].split("'")[0]
        if not self.limits_are_set:
            limits = None
        elif self.limits_are_false:
            limits = False
        elif self.has_rect_limits:
            limits = self.rect_limits if self.n_obs < 3 and not self.n_events > 1 else "rectangular"
        else:
            limits = "functional"
        return f"<zfit {class_name} obs={self.obs}, axes={self.axes}, limits={limits}, binned={self.is_binned}>"

    @deprecated(
        None,
        "Multiple limits won't be supported anymore in the future. For alternatives, see the announcement: https://github.com/zfit/zfit/discussions/533",
    )
    def __add__(self, other):
        if not isinstance(other, ZfitSpace):
            msg = f"Cannot add a {type(self)} and a {type(other)}"
            raise TypeError(msg)
        return add_spaces(self, other)

    def get_sublimits(self):
        limits = self.extract_limits()
        return list(limits.values())

    @deprecate_multispace
    def add(self, *other: ztyping.SpaceOrSpacesTypeInput):
        """Add the limits of the spaces. Only works for the same obs.

        In case the observables are different, the order of the first space is taken.

        Args:
            other:

        Returns:
            :py:class:`~zfit.Space`:
        """
        return add_spaces(self, *other)

    def combine(self, *other: ztyping.SpaceOrSpacesTypeInput) -> ZfitSpace:
        """Combine spaces with different obs (but consistent limits).

        Args:
            other:

        Returns:
            :py:class:`~zfit.Space`:
        """
        return combine_spaces(self, *other)

    def __mul__(self, other):
        return self.combine(other)

    def __ge__(self, other):
        return NotImplemented

    def equal(self, other: object, allow_graph: bool) -> znp.array:
        """Compare the limits on equality. For ANY objects, this also returns true.

        If called inside a graph context *and* the limits are tensors, this will return a symbolic `tf.Tensor`.

        Returns:
            Result of the comparison
        Raises:
             IllegalInGraphModeError: it the comparison happens with tensors in a graph context.
        """
        if not isinstance(other, ZfitSpace):
            return False
        return equal_space(self, other, allow_graph=allow_graph)

    def __eq__(self, other: object) -> bool:
        """Compares two Limits for equality without graph mode allowed.

        Returns:

        Raises:
             IllegalInGraphModeError: it the comparison happens with tensors in a graph context.
        """
        if not isinstance(other, ZfitSpace):
            return NotImplemented
        return self.equal(other=other, allow_graph=False)

    def less_equal(self, other, allow_graph):
        """Set-like comparison for compatibility. If an object is less_equal to another, the limits are combatible.

        This can be used to determine whether a fitting range specification can handle another limit.

        If called inside a graph context *and* the limits are tensors, this will return a symbolic `tf.Tensor`.


        Args:
            other: Any other object to compare with
            allow_graph: If False and the function returns a symbolic tensor, raise IllegalInGraphModeError instead.

        Returns:
            Result of the comparison
        Raises:
             IllegalInGraphModeError: it the comparison happens with tensors in a graph context.
        """
        if not isinstance(other, ZfitSpace):
            return False
        return less_equal_space(self, other, allow_graph=allow_graph)

    def __le__(self, other: object) -> bool:
        """Set-like comparison for compatibility. If an object is less_equal to another, the limits are combatible.

        This can be used to determine whether a fitting range specification can handle another limit.

        Returns:
            Result of the comparison
        Raises:
             IllegalInGraphModeError: it the comparison happens with tensors in a graph context.
        """
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.less_equal(other, allow_graph=False)

    def __hash__(self):
        limits_frozen = tuple(((key, tuple(ldict.items())) for key, ldict in self._limits_dict.items()))
        return hash((limits_frozen, hash(self.coords), hash(self.binning)))

    def reorder_x(
        self,
        x,
        *,
        x_obs=None,
        x_axes=None,
        func_obs=None,
        func_axes=None,
    ):
        return self.coords.reorder_x(x, x_obs=x_obs, x_axes=x_axes, func_obs=func_obs, func_axes=func_axes)

    @deprecated(
        None,
        "Multiple limits won't be supported anymore in the future. For alternatives, see the announcement: https://github.com/zfit/zfit/discussions/533",
    )
    def __len__(self):
        if not self:
            return 0
        else:
            return sum(1 for _ in self)

    def __bool__(self):
        return self.has_limits


def _legacy_get_arguments_space(obs, args, limits, binning, axes, rect_limits, lower, upper):  # noqa: ARG001
    """Legacy function to get the arguments of a `Space`, i.e. making the transition to allow "lower" and "upper"
    smooth."""

    if len(args) > 2:  # catch legacy behavior
        msg = "The API for `Space` has changed and takes now lower, upper as the first two arguments. Cannot take more than 3 positional arguments"
        raise BreakingAPIChangeError(msg)

    if len(args) == 2:
        if all(isinstance(lim, int | float | Any) for lim in args):
            limits = args
        else:
            limits, binning = args
    elif len(args) == 1:
        limits = args[0]
    elif lower is not None and upper is not None:
        limits = [lower, upper]

    return limits, binning


class Space(
    BaseSpace,
    SerializableMixin,
):
    AUTO_FILL = object()
    ANY = ANY
    ANY_LOWER = ANY_LOWER  # TODO: needed? or move everything inside?
    ANY_UPPER = ANY_UPPER

    @deprecated_args(None, "Use `lower` and `upper` instead", ("rect_limits", "limits"))
    def __init__(
        self,
        obs: ztyping.ObsTypeInput | None = None,
        *args,
        limits: ztyping.LimitsTypeInput | None = None,
        binning: ztyping.BinningTypeInput = None,
        axes=None,
        rect_limits=None,
        name: str | None = None,
        label: str | Iterable[str] | None = None,
        lower: ztyping.LimitsTypeInputV1 | None = None,
        upper: ztyping.LimitsTypeInputV1 | None = None,
    ):
        """Define a space with the name (`obs`) of the axes (and it's number) and possibly it's limits.

        A space can be thought of as coordinates, possibly with the definition of a range (limits). For most use-cases,
        it is sufficient to specify a `Space` via observables; simple string identifiers. They can be multidimensional.

        Observables are like the columns of a spreadsheet/dataframe, and are therefore needed for any object that does
        numerical operations or holds data in order to match the right axes. On object creation, the observables are
        assigned using a `Space`. This is often used as the default space of an object and can be used as the
        default `norm`, sampling limits etc.

        Axes are the same concept as observables, but numbers, indexes, and are used *inside* an object. There,
        axes 0 corresponds to the 0th data column we get (which corresponds to a certain observable).

        Args:
            obs: |@doc:space.init.obs| Observable of the space.
               Serves as the "variable". |@docend:space.init.obs|
            lower, upper: |@doc:space.init.lowerupper| Lower and upper limits of the space, respectively.
               Each of them should be a scalar-like object. |@docend:space.init.lowerupper|
            limits: |@doc:space.init.limits| A tuple-like object of the limits of the space.
               These are the lower and upper limits. |@docend:space.init.limits|
            binning: |@doc:space.init.binning| Binning of the space.
               Currently, only regular and variable binning *with a name* is supported.
               If an integer or a list of integers is given with
               lengths equal to the number of observables,
               it is interpreted as the number of bins and
               a regular binning is automatically created using the limits as the
               start and end points. |@docend:space.init.binning|
            name: |@doc:space.init.name| Name of the space.
               Maybe has implications on the serialization and deserialization of the space.
               For a human-readable name, use the label. |@docend:space.init.name|

        Raises
            TypeError: If the axes in the binning do not have a name.
            ObsIncompatibleError: If the obs do not agree with the name of the binning.
            ShapeIncompatibleError: If the shape of the limits or the binnings do not match the shape of the obs.
        """

        self.v1 = V1Space(self)
        self.v0 = V0Space(self)
        self.vec = VectorizeLimits(self)

        from .._variables.axis import Binnings, RegularBinning, histaxes_to_binning  # noqa: PLC0415

        limits, binning = _legacy_get_arguments_space(obs, args, limits, binning, axes, rect_limits, lower, upper)
        if binning is None:
            binning = False
        if name is None:
            name = "Space"
        if binning is not False:
            integer_autobinning = isinstance(binning, int) or (
                isinstance(binning, list | tuple) and all(isinstance(b, int) for b in binning)
            )
            if not integer_autobinning:
                if not isinstance(binning, Binnings):
                    binning = convert_to_container(binning)
                    if binning is not None:
                        binning = histaxes_to_binning(binning)
                if not all(binning.name):
                    msg = f"Axes must have a name. Missing: {[axis for axis in binning if not hasattr(axis, 'name')]}"
                    raise TypeError(msg)
                if obs is None and axes is None:
                    obs = [axis.name for axis in binning]
        super().__init__(obs=obs, axes=axes, name=name)

        label = convert_to_container(label, container=tuple)
        if label is not None and len(label) != self.n_obs:
            msg = f"Number of labels ({len(label)}) does not match the number of observables ({self.n_obs})"
            raise ValueError(msg)

        if self.obs is not None:  # if None, the obs are not given, only axes -> no labels
            if label is None:
                label = self.obs
            elif len(label) != self.n_obs:
                msg = f"Number of labels ({label}) does not match the number of observables ({self.obs})"
                raise ValueError(msg)
            label = dict(zip(self.obs, label, strict=True))
        self._labels = label

        if binning is not False and not isinstance(binning, int) and limits is None and rect_limits is None:
            limits = [[], []]
            for axis in binning:
                limits[0].append(axis.edges[0])
                limits[1].append(axis.edges[-1])

        limits_dict = self._check_convert_input_limits(
            limit=limits,
            rect_limits=rect_limits,
            obs=self.obs,
            axes=self.axes,
            n_obs=self.n_obs,
        )
        self._limits_dict = limits_dict

        if binning is not False and isinstance(binning, int):
            binning = [binning]
        if binning is not False and integer_autobinning:
            if len(binning) != self.n_obs:
                msg = (
                    f"Wrong number ({len(binning)}) of integers given for regular binning"
                    f" ({binning}) with {self.n_obs} observables ({self.obs})."
                    f" Numbers have to match the number of observables."
                )
                raise ShapeIncompatibleError(msg)
            regular_binnings = []
            for i, nbins in enumerate(binning):
                if nbins < 1:
                    msg = "If binning is an integer, it must be > 0"
                    raise ValueError(msg)

                lower = self.lower[0][i]
                upper = self.upper[0][i]
                regular_binnings.append(RegularBinning(bins=nbins, start=lower, stop=upper, name=self.obs[i]))

            binning = Binnings(regular_binnings)
        if binning is not False:
            bining_names = set(binning.name)
            obs = set(self.obs)
            wrong_names = bining_names - obs
            if wrong_names:
                msg = f"Binning names ({wrong_names}) do not match observables ({obs}), {wrong_names} not in space."
                raise ObsIncompatibleError(msg)
            missing_obs = obs - bining_names
            if missing_obs:
                msg = f"Binning names ({missing_obs}) do not match observables ({obs}), missing {missing_obs}."
                raise ObsIncompatibleError(msg)
            binning = Binnings([binning[ob] for ob in self.obs])
        self._binning = None if binning is False else binning





    def _check_convert_input_limits(
        self,
        limit: ztyping.LowerTypeInput | ztyping.UpperTypeInput,
        rect_limits,
        obs,
        axes,
        n_obs,
    ) -> ztyping.LowerTypeReturn | ztyping.UpperTypeReturn:
        """Check and sanitize the input limits as well as the rectangular limits.

        Args:
            limit:

        Returns:
            Limits dictionary containing the observables and/or the axes as a key matching
                `ZfitLimits` objects.
        """
        limits_dict: defaultdict = defaultdict(dict)
        input_limits = limit
        if isinstance(input_limits, Space):
            space = input_limits

            if obs and not axes:
                space = space.with_obs(obs, allow_subset=True, allow_superset=True)
                space = space.with_axes(None)
            elif not obs and axes:
                space = space.with_axes(axes, allow_subset=True, allow_superset=True)
                space = space.with_obs(None)
            elif obs and axes:
                coords = Coordinates(obs=obs, axes=axes)
                space = space.with_coords(coords, allow_superset=True, allow_subset=True)
            input_limits = space.get_limits()

            obs = space.obs
            axes = space.axes
        if not isinstance(input_limits, dict) and not isinstance(rect_limits, dict):
            limit = Limit(limit_fn=limit, rect_limits=rect_limits, n_obs=n_obs)
            i_old = 0
            for lim in limit.get_sublimits():  # split into smaller ones if possible
                i = i_old + lim.n_obs
                if obs is not None:
                    limits_dict["obs"][obs[i_old:i]] = lim
                if axes is not None:
                    limits_dict["axes"][axes[i_old:i]] = lim
                i_old = i
            input_limits = limits_dict

        if isinstance(input_limits, dict):
            input_limits = input_limits.copy()
        elif isinstance(rect_limits, dict):
            input_limits = rect_limits.copy()

        if "axes" not in input_limits and "obs" not in input_limits:
            msg = "Probably internal error: wrong format of limits_dict"
            raise ValueError(msg)

        if obs:
            if "obs" in input_limits:
                obs_limit_dict = input_limits["obs"]
                obs_limit_dict = {ob: lim for ob, lim in obs_limit_dict.items() if ob[0] in obs}
            else:
                obs_limit_dict = {}
                for axes_lim, lim in input_limits["axes"].items():
                    obs_coords = tuple(obs[axes.index(ax)] for ax in axes_lim)
                    if isinstance(lim, ZfitOrderableDimensional):
                        lim = lim.with_coords(self.space)
                    obs_limit_dict[obs_coords] = lim
            limits_dict["obs"] = obs_limit_dict

        if axes:
            if "axes" in input_limits:
                axes_limit_dict = input_limits["axes"]
                axes_limit_dict = {axis: lim for axis, lim in axes_limit_dict.items() if axis[0] in axes}
            else:
                axes_limit_dict = {}
                for obs_lim, lim in input_limits["obs"].items():
                    axes_coords = tuple(axes[obs.index(ob)] for ob in obs_lim)

                    if isinstance(lim, ZfitOrderableDimensional):
                        lim = lim.with_coords(self.space)
                    axes_limit_dict[axes_coords] = lim
            limits_dict["axes"] = axes_limit_dict

        if not axes and "axes" in limits_dict:
            limits_dict.pop("axes")

        if not obs and "obs" in limits_dict:
            limits_dict.pop("obs")

        return limits_dict

    def get_limits(
        self, obs: ztyping.ObsTypeInput = None, axes: ztyping.AxesTypeInput = None
    ) -> ztyping.LimitsDictWithCoords | ztyping.LimitsDictNoCoords:
        return_dict = {}
        both_none_or_true = (obs is None and axes is None) or (obs is True and axes is True)

        if (obs is True and axes is None) or both_none_or_true:
            if not self.obs:
                if obs is True:
                    msg = "Obs are not defined for this instance, no limits set for obs."
                    raise ObsIncompatibleError(msg)
            else:
                return_dict["obs"] = self._limits_dict["obs"].copy()
        if (axes is True and obs is None) or both_none_or_true:
            if not self.axes:
                if axes is True:
                    msg = "Axes are not defined for this instance, no limits set for axes."
                    raise AxesIncompatibleError(msg)
            else:
                return_dict["axes"] = self._limits_dict["axes"].copy()
        else:
            if obs:
                return_dict["obs"] = extract_limits_from_dict(self._limits_dict, obs=obs)
            if axes:
                return_dict["axes"] = extract_limits_from_dict(self._limits_dict, axes=axes)
        return return_dict

    @property
    @fail_not_rect
    def limits(self) -> ztyping.LimitsTypeReturn:
        pass

    @property
    @deprecated(
        None,
        "Use for compatibility the v1.limits, v1.lower, v1.upper, as described here: https://github.com/zfit/zfit/discussions/533",
    )
    def rect_limits(self) -> ztyping.RectLimitsReturnType:
        pass


    @property
    def _rect_limits_tf(self) -> ztyping.LimitsTypeReturn:
        pass

    @property
    @deprecated(
        None,
        "Use for compatibility the v1.limits, v1.lower, v1.upper, as described here: https://github.com/zfit/zfit/discussions/533",
    )
    def rect_limits_np(self) -> ztyping.RectLimitsNPReturnType:
        pass

    @property
    @deprecated(
        None,
        "Use for compatibility the v1.limits, v1.lower, v1.upper, as described here: https://github.com/zfit/zfit/discussions/533",
    )
    def rect_upper(self) -> ztyping.UpperTypeReturn:
        pass

    @property
    @deprecated(
        None,
        "Use for compatibility the v1.limits, v1.lower, v1.upper, as described here: https://github.com/zfit/zfit/discussions/533",
    )
    def rect_lower(self) -> ztyping.RectLowerReturnType:
        pass


    @deprecated(
        None,
        "Use for compatibility the v1.limits, v1.lower, v1.upper, as described here: https://github.com/zfit/zfit/discussions/533",
    )
    def rect_area(self) -> float | np.ndarray | znp.array:
        pass

    @property
    def rect_limits_are_tensors(self) -> bool:
        pass

    @property
    def has_rect_limits(self) -> bool:
        pass

    @property
    def limits_are_false(self) -> bool:
        pass

    @property
    def has_limits(self) -> bool:
        pass


    @property
    def n_events(self) -> int | None:
        pass

    @property
    @fail_not_rect
    def lower(self) -> ztyping.LowerTypeReturn:
        """Return the lower limits.

        Returns:
        """
        return self.rect_lower

    @property
    @fail_not_rect
    def upper(self) -> ztyping.UpperTypeReturn:
        pass

    @property
    @deprecated(
        None,
        "Multiple limits won't be supported anymore in the future. For alternatives, see the announcement:"
        "https://github.com/zfit/zfit/discussions/533",
    )
    def n_limits(self) -> int:
        pass


    def with_limits(
        self,
        limits: ztyping.LimitsTypeInput = None,
        rect_limits: ztyping.RectLimitsInputType | None = None,
        name: str | None = None,
    ) -> ZfitSpace:
        """Return a copy of the space with the new `limits` (and the new `name`).

        Args:
            limits: Limits to use. Can be rectangular, a function (requires to also specify `rect_limits`
                or an instance of ZfitLimit.
            rect_limits: Rectangular limits that will be assigned with the instance
            name: Human readable name

        Returns:
            Copy of the current object with the new limits.
        """
        return type(self)(
            obs=self.coords, limits=limits, rect_limits=rect_limits, binning=self.binning, name=name, label=self.labels
        )

    def reorder_x(
        self,
        x: tf.Tensor | np.ndarray,
        *,
        x_obs: ztyping.ObsTypeInput = None,
        x_axes: ztyping.AxesTypeInput = None,
        func_obs: ztyping.ObsTypeInput = None,
        func_axes: ztyping.AxesTypeInput = None,
    ) -> ztyping.XTypeReturnNoData:
        """Reorder x in the last dimension either according to its own obs or assuming a function ordered with func_obs.

        There are two obs or axes around: the one associated with this Coordinate object and the one associated with x.
        If x_obs or x_axes is given, then this is assumed to be the obs resp. the axes of x and x will be reordered
        according to `self.obs` resp. `self.axes`.

        If func_obs resp. func_axes is given, then x is assumed to have `self.obs` resp. `self.axes` and will be
        reordered to align with a function ordered with `func_obs` resp. `func_axes`.

        Switching `func_obs` for `x_obs` resp. `func_axes` for `x_axes` inverts the reordering of x.

        Args:
            x: Tensor to be reordered, last dimension should be n_obs resp. n_axes
            x_obs: Observables associated with x. If both, x_obs and x_axes are given, this has precedency over the
                latter.
            x_axes: Axes associated with x.
            func_obs: Observables associated with a function that x will be given to. Reorders x accordingly and assumes
                self.obs to be the obs of x. If both, `func_obs` and `func_axes` are given, this has precedency over the
                latter.
            func_axes: Axe associated with a function that x will be given to. Reorders x accordingly and assumes
                self.axes to be the axes of x.

        Returns:
            The reordered array-like object
        """
        return self.coords.reorder_x(x=x, x_obs=x_obs, x_axes=x_axes, func_obs=func_obs, func_axes=func_axes)

    def with_obs(
        self,
        obs: ztyping.ObsTypeInput | None,
        allow_superset: bool = True,
        allow_subset: bool = True,
    ) -> ZfitSpace:
        """Create a new Space that has `obs`; sorted by or set or dropped.

        The behavior is as follows:

         * obs are already set:

           * input obs are None: the observables will be dropped. If no axes are set, an error
             will be raised, as no coordinates will be assigned to this instance anymore.
           * input obs are not None: the instance will be sorted by the incoming obs. If axes or other
             objects have an associated order (e.g. data, limits,...), they will be reordered as well.
             If a strict subset is given (and allow_subset is True), only a subset will be returned.
             This can be used to take a subspace of limits, data etc.
             If a strict superset is given (and allow_superset is True), the obs will be sorted accordingly as
             if the obs not contained in the instances obs were not in the input obs.
         * obs are not set:

           * if the input obs are None, the same object is returned.
           * if the input obs are not None, they will be set as-is and now correspond to the already
             existing axes in the object.

        Args:
            obs: Observables to sort/associate this instance with
            allow_superset: if False and a strict superset of the own observables is given, an error
            is raised.
            allow_subset:if False and a strict subset of the own observables is given, an error
            is raised.

        Returns:
            A copy of the object with the new ordering/observables

        Raises:
            CoordinatesUnderdefinedError: if obs is None and the instance does not have axes
            ObsIncompatibleError: if `obs` is a superset and allow_superset is False or a subset and
                allow_allow_subset is False
        """
        if obs is None:  # drop obs, check if there are axes
            if self.obs is None:
                return self
            if self.axes is None:
                msg = "Cannot remove obs (using None) for a Space without axes"
                raise AxesIncompatibleError(msg)
            new_limits = self._limits_dict.copy()
            new_space = self.copy(obs=obs, limits=new_limits)
        else:
            obs = _convert_obs_to_str(obs)
            coords = self.coords.with_obs(obs, allow_superset=allow_superset, allow_subset=allow_subset)
            binning = self.binning
            if binning is not None:
                binning = [binning[ob] for ob in obs if ob in self.obs]
            if (newlabels := self.labels) is not None:
                newlabels = tuple(self._labels[ob] for ob in coords.obs)  # use just the obs that are available
            new_space = type(self)(coords, limits=self._limits_dict, binning=binning, label=newlabels)
        return new_space

    def with_axes(
        self,
        axes: ztyping.AxesTypeInput | None,
        allow_superset: bool = True,
        allow_subset: bool = True,
    ) -> ZfitSpace:
        """Create a new instance that has `axes`; sorted by or set or dropped.

        The behavior is as follows:

         * axes are already set:

           * input axes are None: the axes will be dropped. If no observables are set, an error
             will be raised, as no coordinates will be assigned to this instance anymore.
           * input axes are not None: the instance will be sorted by the incoming axes. If obs or other
             objects have an associated order (e.g. data, limits,...), they will be reordered as well.
             If a strict subset is given (and allow_subset is True), only a subset will be returned. This can
             be used to retrieve a subspace of limits, data etc.
             If a strict superset is given (and allow_superset is True), the axes will be sorted accordingly as
             if the axes not contained in the instances axes were not present in the input axes.
         * axes are not set:

           * if the input axes are None, the same object is returned.
           * if the input axes are not None, they will be set as-is and now correspond to the already
             existing obs in the object.

        Args:
            axes: Axes to sort/associate this instance with
            allow_superset: if False and a strict superset of the own axeservables is given, an error
            is raised.
            allow_subset:if False and a strict subset of the own axeservables is given, an error
            is raised.

        Returns:
            A copy of the object with the new ordering/axes
        Raises:
            CoordinatesUnderdefinedError: if obs is None and the instance does not have axes
            AxesIncompatibleError: if `axes` is a superset and allow_superset is False or a subset and
                allow_allow_subset is False
        """
        if axes is None:  # drop axes
            if self.axes is None:
                return self
            if self.obs is None:
                msg = "Cannot remove axes (using None) for a Space without obs"
                raise ObsIncompatibleError(msg)
            new_limits = self._limits_dict.copy()
            if (newlabels := self.labels) is not None and (obs := self.obs) is not None:
                newlabels = tuple(self._labels[ob] for ob in obs)
            new_space = self.copy(axes=axes, limits=new_limits, label=newlabels)
        else:
            axes = convert_to_axes(axes)
            if self.axes is None:
                if len(axes) != len(self.obs):
                    msg = f"Trying to set axes {axes} to object with obs {self.obs}"
                    raise AxesIncompatibleError(msg)
                if (newlabels := self.labels) is not None and (obs := self.obs) is not None:
                    newlabels = tuple(self._labels[ob] for ob in obs)
                new_space = self.copy(axes=axes, limits=self._limits_dict, label=newlabels)
            else:
                coords = self.coords.with_axes(axes=axes, allow_superset=allow_superset, allow_subset=allow_subset)
                if (newlabels := self.labels) is not None and (obs := coords.obs) is not None:
                    newlabels = tuple(self._labels[ob] for ob in obs)
                new_space = type(self)(coords, limits=self._limits_dict, binning=self.binning, label=newlabels)

        return new_space

    def with_coords(
        self,
        coords: ZfitOrderableDimensional,
        allow_superset: bool = True,
        allow_subset: bool = True,
    ) -> Space:
        """Create a new :py:class:`~zfit.Space` with reordered observables and/or axes.

        The behavior is that _at least one coordinate (obs or axes) has to be set in both instances
        (the space itself or in `coords`). If both match, observables is taken as the defining coordinate.
        The space is sorted according to the defining coordinate and the other coordinate is sorted as well.
        If either the space did not have the "weaker coordinate" (e.g. both have observables, but only coords
        has axes), then the resulting Space will have both.
        If both have both coordinates, obs and axes, and sorting for obs results in non-matchin axes results
        in axes being dropped.

        Args:
            coords: An instance of :py:class:`Coordinates`
            allow_superset: If `False` and a strict superset is given, an error is raised
            allow_subset: If `False` and a strict subset is given, an error is raised

        Returns:
            :py:class:`~zfit.Space`:
        Raises:
            CoordinatesUnderdefinedError: if neither both obs or axes are specified.
            CoordinatesIncompatibleError: if `coords` is a superset and allow_superset is False or a subset and
                allow_allow_subset is False
        """
        if self.obs is not None and coords.obs is not None:
            new_space_obs = self.with_obs(coords.obs, allow_superset=allow_superset, allow_subset=allow_subset)

            if coords.axes is not None:  # use this axes: first drop the other one
                if new_space_obs.axes is not None:
                    new_space_obs = new_space_obs.with_axes(None)
                coords_axes = coords.with_obs(
                    new_space_obs.obs,
                    allow_superset=allow_superset,
                    allow_subset=allow_subset,
                )
                new_space_obs = new_space_obs.with_axes(coords_axes.axes)  # are the same or self.axes is None
            new_space = new_space_obs

        elif self.axes is not None and coords.axes is not None:
            new_space_axes = self.with_axes(coords.axes, allow_superset=allow_superset, allow_subset=allow_subset)
            if coords.obs is not None:
                coords_obs = coords.with_axes(
                    new_space_axes.axes,
                    allow_superset=allow_superset,
                    allow_subset=allow_subset,
                )
                new_space_axes = new_space_axes.with_obs(coords_obs.obs)
            new_space = new_space_axes
        else:
            msg = f"Neither the axes nor the obs are specified in both objects {self} and {coords}"
            raise CoordinatesUnderdefinedError(msg)

        return new_space

    def with_autofill_axes(self, overwrite: bool = False) -> zfit.Space:
        """Overwrite the axes of the current object with axes corresponding to range(len(n_obs)).

        This effectively fills with (0, 1, 2,...) and can be used mostly when an object enters a PDF or
        similar. `overwrite` allows to remove the axis first in case there are already some set.

        .. code-block::

            object.obs -> ('x', 'z', 'y')
            object.axes -> None

            object.with_autofill_axes()

            object.obs -> ('x', 'z', 'y')
            object.axes -> (0, 1, 2)


        Args:
            overwrite: If axes are already set, replace the axes with the autofilled ones.
                If axes is already set and `overwrite` is False, raise an error.

        Returns:
            The object with the new axes

        Raises:
            AxesIncompatibleError: if the axes are already set and `overwrite` is False.
        """
        new_coords = self.coords.with_autofill_axes(overwrite=overwrite)
        return self.copy(axes=new_coords.axes) if self.axes is None or overwrite else self

    def get_subspace(
        self,
        obs: ztyping.ObsTypeInput = None,
        axes: ztyping.AxesTypeInput = None,
        name: str | None = None,
    ) -> Space:
        """Create a :py:class:`~zfit.Space` consisting of only a subset of the `obs`/`axes` (only one allowed).

        Args:
            obs: Observables of the subspace to return.
            axes: Axes of the subspace to return.
            name: Human readable names

        Returns:
            A space containing only a subspace (and sublimits etc.)
        """
        del name  # should be label
        if obs is not None and axes is not None:
            msg = "Cannot specify `obs` *and* `axes` to get subspace."
            raise ValueError(msg)
        if axes is None and obs is None:
            msg = "Either `obs` or `axes` has to be specified and not None"
            raise ValueError(msg)

        obs = self._check_convert_input_obs(obs=obs, allow_none=True)
        axes = self._check_convert_input_axes(axes=axes, allow_none=True)
        if obs is not None:
            limits_dict = self.get_limits(obs=obs)
            new_coords = self.coords.with_obs(obs, allow_subset=True, allow_superset=True)
        else:
            limits_dict = self.get_limits(axes=axes)
            new_coords = self.coords.with_axes(axes=axes, allow_subset=True, allow_superset=True)
        if (newlabels := self.labels) is not None and (obs := new_coords.obs) is not None:
            newlabels = tuple(self._labels[ob] for ob in obs)
        return type(self)(obs=new_coords, limits=limits_dict, label=newlabels)

    @fail_not_rect
    def _legacy_area(self) -> float:
        pass

    def with_binning(self, binning):
        return self.copy(binning=binning)


    def copy(self, **overwrite_kwargs) -> zfit.Space:
        """Create a new :py:class:`~zfit.Space` using the current attributes and overwriting with
        `overwrite_overwrite_kwargs`.

        Args:
            name: The new name. If not given, the new instance will be named the same as the
                current one.
            **overwrite_kwargs:

        Returns:
            :py:class:`~zfit.Space`
        """
        kwargs = {
            "name": self.name,
            "limits": self._limits_dict,
            "binning": self.binning,
            "axes": self.axes,
            "obs": self.obs,
            "label": self.labels,
        }
        kwargs.update(overwrite_kwargs)
        if set(overwrite_kwargs) - set(kwargs):
            msg = f"Not usable keys in `overwrite_kwargs`: {set(overwrite_kwargs) - set(kwargs)}"
            raise KeyError(msg)
        kwargs.get("binning")
        return self.__class__(**kwargs)

    def _inside(self, x, guarantee_limits):
        del guarantee_limits  # not used
        xs_inside = []
        obs_in_use = self.obs is not None
        limits_dict = self._limits_dict["obs" if obs_in_use else "axes"]
        for coords, limit in limits_dict.items():
            reorder_kwargs = {"func_obs" if obs_in_use else "func_axes": coords}
            x_sub = self.reorder_x(x, **reorder_kwargs)
            x_inside = limit.inside(x_sub)
            xs_inside.append(x_inside)
        return znp.all(xs_inside, axis=0)

    @property  # TODO(1.0): deprecate limits, see also https://github.com/zfit/zfit/discussions/533
    @fail_not_rect
    def limit1d(self) -> tuple[float, float]:
        pass

    @classmethod
    @deprecated(
        date=None,
        instructions="Use directly the class to create a Space. E.g. zfit.Space(axes=(0, 1), ...)",
    )
    def from_axes(
        cls,
        axes: ztyping.AxesTypeInput,  # noqa: ARG003
        limits: ztyping.LimitsTypeInput | None = None,  # noqa: ARG003
        rect_limits=None,  # noqa: ARG003
        name: str | None = None,  # noqa: ARG003
    ) -> zfit.Space:
        pass


def extract_limits_from_dict(limits_dict, obs=None, axes=None):
    if (obs is None) and (axes is None):
        msg = "Need to specify at least one, obs or axes."
        raise ValueError(msg)
    if (obs is not None) and (axes is not None):
        axes = None  # obs has precedency
    if obs is None:
        obs_in_use = False
        coords_to_extract = axes
    else:
        obs_in_use = True
        coords_to_extract = obs
    coords_to_extract = convert_to_container(coords_to_extract)
    coords_to_extract = set(coords_to_extract)

    limits_to_eval = {}
    limit_dict = limits_dict["obs" if obs_in_use else "axes"].items()
    keys_sorted = sorted(limit_dict, key=lambda x: len(x[0]), reverse=True)
    for key_coords, limit in keys_sorted:
        coord_intersec = frozenset(key_coords).intersection(coords_to_extract)
        if not coord_intersec:  # this limit does not contain any requested obs
            continue
        if coord_intersec == frozenset(key_coords):
            if isinstance(limit, ZfitOrderableDimensional):  # drop coordinates if given
                limit = limit.with_axes(None) if obs_in_use else limit.with_obs(None)
            limits_to_eval[key_coords] = limit
        else:
            coord_limit = [coord for coord in key_coords if coord in coord_intersec]
            kwargs = {"obs" if obs_in_use else "axes": coord_limit}
            try:
                sublimit = limit.get_subspace(**kwargs)
            except InvalidLimitSubspaceError:
                msg = f"Cannot extract {coord_intersec} from limit {limit}."
                raise InvalidLimitSubspaceError(msg) from None
            sublimit_coord = limit.obs if obs_in_use else limit.axes
            if isinstance(sublimit, ZfitOrderableDimensional):  # drop coordinates if given
                sublimit = sublimit.with_axes(None) if obs_in_use else sublimit.with_obs(None)
            limits_to_eval[sublimit_coord] = sublimit
            coords_to_extract -= coord_intersec
    return limits_to_eval


def add_spaces(*spaces: Iterable[ZfitSpace], name=None):
    """Add two spaces and merge their limits if possible or return False.

    Args:
        spaces:

    Returns:
        Union[None, :py:class:`~zfit.Space`, bool]:

    Raises:
        LimitsIncompatibleError: if limits of the `spaces` cannot be merged because they overlap
    """
    if not all(isinstance(space, ZfitSpace) for space in spaces):
        msg = f"Can only add type ZfitSpace, not {spaces}"
        raise TypeError(msg)
    return MultiSpace(spaces, name=name)


def get_coord(space, obs_in_use=True):
    if obs_in_use:
        return space.obs
    else:
        return space.axes


def combine_spaces(*spaces: Iterable[Space]):
    """Combine spaces with different `obs` and `limits` to one `space`.

    Checks if the limits in each obs coincide *exactly*. If this is not the case, the combination
    is not unambiguous and `False` is returned

    Args:
        spaces:

    Returns:
        Returns False if the limits don't coincide in one or more obs. Otherwise
            return the :py:class:`~zfit.Space` with all obs from `spaces` sorted by the order of `spaces` and with the
            combined limits.
    Raises:
        ValueError: if only one space is given
        LimitsIncompatibleError: If the limits of one or more spaces (or within a space) overlap
        LimitsNotSpecifiedError: If the limits for one or more obs but not all are None or False.
    """
    from .._variables.axis import Binnings  # noqa: PLC0415

    spaces = convert_to_container(spaces, container=tuple)

    common_obs_ordered = common_obs(spaces=spaces)
    common_axes_ordered = common_axes(spaces=spaces)
    all_spaces_binned = all(space.is_binned for space in spaces)
    all_spaces_unbinned = not any(space.is_binned for space in spaces)
    if not (all_spaces_binned or all_spaces_unbinned):
        msg = (
            f"Some spaces are binned {[s for s in spaces if s.is_binned]}"
            f" while others are not {[s for s in spaces if not s.is_binned]}. Cannot mix."
        )
        raise ValueError(msg)

    all_labels_map = {}
    for space in spaces:
        if (lab := space._labels) is not None:
            all_labels_map.update(lab)
    if all_spaces_binned:
        binnings_ordererd = []
        for ob in common_obs_ordered:
            for space in spaces:
                if ob in space.obs:
                    binning = space.binning[ob]
                    if binning not in binnings_ordererd:
                        binnings_ordererd.append(binning)

        binning = Binnings(binnings_ordererd)
    else:
        binning = None
    using_obs = bool(common_obs_ordered)
    common_coords_ordered = common_obs_ordered if using_obs else common_axes_ordered

    if using_obs:
        spaces = tuple(space.with_obs(common_obs_ordered, allow_superset=True) for space in spaces)
        all_coords = [space.obs for space in spaces]
    elif common_axes_ordered:
        spaces = tuple(space.with_axes(common_axes_ordered, allow_superset=True) for space in spaces)
        all_coords = [space.axes for space in spaces]
    else:
        msg = "Neither `obs` nor `axes` exist in all spaces."
        raise CoordinatesUnderdefinedError(msg)

    all_limits_false = all(space.limits_are_false for space in spaces)
    all_limits_not_set = all(not space.limits_are_set for space in spaces)
    has_limits = [space.has_limits for space in spaces]
    if all_limits_false:
        limits = False
    elif all_limits_not_set:
        limits = None
    elif not all(has_limits):
        msg = "Limits either have to be set, not set, or False for all spaces to be combined."
        raise LimitsNotSpecifiedError(msg)
    else:
        space_combinations = tuple(itertools.product(*spaces))
        if len(space_combinations) > 1:  # there are MultiSpaces in there
            all_combinations = []
            for spa in space_combinations:
                with suppress(LimitsIncompatibleError):
                    all_combinations.append(combine_spaces(*spa))
                if not all_combinations:
                    msg = f"The limits of {spaces} are all not compatible to be combined."
                    raise LimitsIncompatibleError(msg)
            [space for space in all_combinations if space is not False]

            return MultiSpace(
                spaces=all_combinations,
                obs=common_obs_ordered if common_obs_ordered else None,
                axes=common_axes_ordered if common_axes_ordered else None,
            )
        limits_dict = {}

        non_unique_coords = set()
        unique_coords = set()
        for coord in common_coords_ordered:
            if sum(coord in coords for coords in all_coords) > 1:
                non_unique_coords.add(coord)
            else:
                unique_coords.add(coord)

        for coord in common_coords_ordered:
            if coord in unique_coords:
                space = next(space for space in spaces if coord in get_coord(space, using_obs))
                space = space.get_subspace(
                    obs=unique_coords if using_obs else None,
                    axes=None if using_obs else unique_coords,
                )
                limits_dict.update(space.get_limits()["obs" if using_obs else "axes"])
                for coord in get_coord(space, using_obs):
                    unique_coords.remove(coord)
            elif coord in non_unique_coords:
                non_unique_spaces = [space for space in spaces if coord in get_coord(space, using_obs)]
                common_coords_non_unique = list(
                    set.intersection(*(set(get_coord(space, using_obs)) for space in non_unique_spaces))
                )
                non_unique_subspaces = [
                    space.get_subspace(
                        obs=common_coords_non_unique if using_obs else None,
                        axes=None if using_obs else common_coords_non_unique,
                    )
                    for space in non_unique_spaces
                ]

                any_non_equal = any(non_unique_subspaces[0] != space for space in non_unique_subspaces[1:])
                if any_non_equal:
                    msg = f"Limits in coord {common_coords_non_unique} do not match for spaces {non_unique_subspaces}"
                    raise LimitsIncompatibleError(msg)

                non_unique_subspace = non_unique_subspaces[0]
                limits_dict.update(non_unique_subspace.get_limits()["obs" if using_obs else "axes"])
                for coord in get_coord(non_unique_subspace, using_obs):
                    non_unique_coords.remove(coord)
            else:
                pass  # fine, since it is already satisfied by a space

        limits = {"obs" if using_obs else "axes": limits_dict}



    return Space(
        obs=common_obs_ordered if using_obs else None,
        axes=None if using_obs else common_axes_ordered,
        binning=binning,
        limits=limits,
        label=tuple(all_labels_map[ob] for ob in common_coords_ordered) if using_obs else None,
    )


def less_equal_space(space1, space2, allow_graph=True):
    return compare_multispace(
        space1=space1,
        space2=space2,
        comparator=lambda limit1, limit2: limit1.less_equal(limit2, allow_graph=allow_graph),
    )


def equal_space(space1, space2, allow_graph=True):
    has_any = True
    with suppress(Exception):
        has_any = np.any(space1.ANY in np.array(space1.v1.limits)) or np.any(space2.ANY in np.array(space2.v1.limits))

    if has_any or allow_graph or isinstance(space1, MultiSpace) or isinstance(space2, MultiSpace):
        return compare_multispace(
            space1=space1,
            space2=space2,
            comparator=lambda limit1, limit2: limit1.equal(limit2, allow_graph=allow_graph),
        )
    else:
        return compare_spaces_equal_static(space1, space2)


def compare_spaces_equal_static(space1: Space, space2: Space):
    """Compare two spaces if they have the same obs, axes, and, if a comparator is given, limits.

    It is automatically checked if the limits are set resp. are False

    Args:
        space1:
        space2:
        comparator:

    Returns:
    """
    try:
        space2obs1 = space2.with_coords(space1, allow_superset=False, allow_subset=False)
    except CoordinatesIncompatibleError:
        return False

    try:
        limits1 = np.array(space1.v1.limits)
        limits2 = np.array(space2obs1.v1.limits)
        diff = np.abs(limits1 - limits2)
    except Exception:
        return False

    return np.all(diff < 1e-8)


def compare_multispace(space1: ZfitSpace, space2: ZfitSpace, comparator: Callable):
    """Compare multiple spaces if they have the same obs, axes, and, if a comparator is given, limits.

    It is automatically checked if the limits are set resp. are False

    Args:
        space1:
        space2:
        comparator:

    Returns:
    """
    axes_not_none = space1.axes is not None and space2.axes is not None
    obs_not_none = space1.obs is not None and space2.obs is not None
    if not (axes_not_none or obs_not_none):  # if both are None
        return False

    if obs_not_none:
        if set(space1.obs) != set(space2.obs):
            return False
    elif axes_not_none and set(space1.axes) != set(space2.axes):  # axes only matter if there are no obs
        return False
    if space1.binning != space2.binning:
        return False
    if not space1.limits_are_set:
        return bool(not space2.limits_are_set)

    elif space1.limits_are_false:
        return bool(space2.limits_are_false)

    return compare_limits_multispace(space1, space2, comparator=comparator)


def compare_limits_multispace(space1: ZfitSpace, space2: ZfitSpace, comparator: Callable) -> bool:
    if len(space1) != len(space2):
        return False
    if not (space1.has_limits and space2.has_limits):
        return False
    space2_reordered = space2.with_coords(space1)

    comparison = []
    for space11 in space1:
        compare_spaces2 = []
        for space22 in space2_reordered:
            compare_spaces2.append(
                compare_limits_coords_dict(space11.get_limits(), space22.get_limits(), comparator=comparator)
            )
        comparison.append(compare_spaces2)
    comparison = convert_to_tensor_or_numpy(comparison, dtype=tf.bool)
    space1_matches = z.unstable.reduce_any(comparison, axis=1)  # reduce over axis containing space2, has to match with
    space2_matches = z.unstable.reduce_any(comparison, axis=0)
    all_space1_match = z.unstable.reduce_all(space1_matches, axis=0)
    all_space2_match = z.unstable.reduce_all(space2_matches, axis=0)

    return z.unstable.logical_and(all_space1_match, all_space2_match)


def compare_limits_coords_dict(
    limits1: Mapping[str, Mapping[Iterable, ZfitLimit]],
    limits2: Mapping[str, Mapping[Iterable, ZfitLimit]],
    comparator: Callable,
    require_all_coord_types: bool = False,
) -> bool:
    if limits1.keys() != limits2.keys() and require_all_coord_types:
        return False
    equal = []
    for coord_type, limit1_dict in limits1.items():
        limit2_dict = limits2.get(coord_type)
        if limit2_dict is None:
            continue
        equal.append(compare_limits_dict(limit1_dict, limit2_dict, comparator=comparator))
    return z.unstable.reduce_all(equal)


def compare_limits_dict(dict1: Mapping, dict2: Mapping, comparator: Callable) -> bool:
    comparison = []
    limits2_to_check = dict2.copy()

    for coord, limit1 in dict1.items():
        for limit2cord, limit2 in limits2_to_check.items():
            if set(limit2cord) == set(coord):
                limit2 = limits2_to_check.pop(limit2cord)
                comparison.append(comparator(limit1, limit2))
                break

        else:  # no break, nothing matched
            return False
    return z.unstable.reduce_all(comparison)




class MultiSpace(BaseSpace):
    def __new__(
        cls,
        spaces: Iterable[ZfitSpace],
        obs: ztyping.ObsTypeInput = None,
        binning: ztyping.BinningTypeInput = None,
        axes: ztyping.AxesTypeInput = None,
        name: str | None = None,
    ) -> Space | MultiSpace:
        del name
        if binning is not None:
            msg = (
                "Binning not yet implemented for MultiSpace, won't ever be. Use the new truncated, multidim PDF instead"
            )
            raise RuntimeError(msg)
        spaces, obs, axes = cls._check_convert_input_spaces_obs_axes(spaces, obs, axes)
        if len(spaces) == 1:
            return spaces[0]
        space = super().__new__(cls)

        space._tmp_store_spaces_obs_axes = spaces, obs, axes

        return space

    @deprecated(
        date=None,
        instructions="MultiSpace is deprecated, see announcement here: https://github.com/zfit/zfit/discussions/533"
        "If you're use-case is not covered by this or you have questions, please open an issue "
        " at https://github.com/zfit/zfit/issues/new?assignees=mayou36&labels=discussion&projects=&template=behaviorunderdiscussion.md&title=%5BMultiSpace%20deprecation%5D",
    )
    def __init__(self, spaces: Iterable[ZfitSpace], obs=None, axes=None, name: str | None = None) -> None:
        del spaces, obs, axes  # not needed, we take the already preprocessed.
        spaces, obs, axes = self._tmp_store_spaces_obs_axes
        del self._tmp_store_spaces_obs_axes
        if name is None:
            name = "MultiSpace"
        super().__init__(obs, axes, name)

        self._labels = {ob: ob for ob in self.obs} if self.obs is not None else None

        self.spaces = spaces
        self.v0 = V0Space(self)








    @property
    @fail_not_rect
    def lower(self) -> None:
        self._raise_limits_not_implemented()





    def rect_area(self) -> float | np.ndarray | tf.Tensor:
        pass

    @property
    def rect_limits_are_tensors(self) -> bool:
        pass

    @property
    def has_rect_limits(self) -> bool:
        pass


    @property
    def limits_are_false(self) -> bool:
        pass


    @property
    def has_limits(self) -> bool:
        pass

    @property
    def limits_are_set(self) -> bool:
        pass

    @property
    def n_events(self) -> int | None:
        pass

    def with_limits(
        self,
        limits: ztyping.LimitsTypeInput = None,
        rect_limits: ztyping.RectLimitsInputType | None = None,
        name: str | None = None,
    ) -> ZfitSpace:
        """Return a copy of the space with the new `limits` (and the new `name`).

        Args:
            limits: Limits to use. Can be rectangular, a function (requires to also specify `rect_limits`
                or an instance of ZfitLimit.
            rect_limits: Rectangular limits that will be assigned with the instance
            name: Human readable name

        Returns:
            Copy of the current object with the new limits.
        """
        return self.copy(
            spaces=[space.with_limits(limits=limits, rect_limits=rect_limits) for space in self],
            name=name,
        )


    def with_obs(
        self,
        obs: ztyping.ObsTypeInput | None,
        allow_superset: bool = True,
        allow_subset: bool = True,
    ) -> MultiSpace:
        """Create a new Space that has `obs`; sorted by or set or dropped.

        The behavior is as follows:

         * obs are already set:
           * input obs are None: the observables will be dropped. If no axes are set, an error
             will be raised, as no coordinates will be assigned to this instance anymore.
           * input obs are not None: the instance will be sorted by the incoming obs. If axes or other
             objects have an associated order (e.g. data, limits,...), they will be reordered as well.
             If a strict subset is given (and allow_subset is True), only a subset will be returned.
             This can be used to take a subspace of limits, data etc.
             If a strict superset is given (and allow_superset is True), the obs will be sorted accordingly as
             if the obs not contained in the instances obs were not in the input obs.
         * obs are not set:
           * if the input obs are None, the same object is returned.
           * if the input obs are not None, they will be set as-is and now correspond to the already
             existing axes in the object.

        Args:
            obs: Observables to sort/associate this instance with
            allow_superset: if False and a strict superset of the own observables is given, an error
            is raised.
            allow_subset:if False and a strict subset of the own observables is given, an error
            is raised.

        Returns:
            A copy of the object with the new ordering/observables

        Raises:
            CoordinatesUnderdefinedError: if obs is None and the instance does not have axes
            ObsIncompatibleError: if `obs` is a superset and allow_superset is False or a subset and
                allow_allow_subset is False
        """
        spaces = [
            space.with_obs(obs, allow_superset=allow_superset, allow_subset=allow_subset) for space in self.spaces
        ]
        coords = self.coords.with_obs(obs, allow_subset=allow_subset, allow_superset=allow_superset)
        return self.copy(spaces=spaces, obs=coords.obs, axes=coords.axes)

    def with_axes(
        self,
        axes: ztyping.AxesTypeInput | None,
        allow_superset: bool = True,
        allow_subset: bool = True,
    ) -> MultiSpace:
        """Create a new instance that has `axes`; sorted by or set or dropped.

        The behavior is as follows:

         * axes are already set:
           * input axes are None: the axes will be dropped. If no observables are set, an error
             will be raised, as no coordinates will be assigned to this instance anymore.
           * input axes are not None: the instance will be sorted by the incoming axes. If obs or other
             objects have an associated order (e.g. data, limits,...), they will be reordered as well.
             If a strict subset is given (and allow_subset is True), only a subset will be returned. This can
             be used to retrieve a subspace of limits, data etc.
             If a strict superset is given (and allow_superset is True), the axes will be sorted accordingly as
             if the axes not contained in the instances axes were not present in the input axes.
         * axes are not set:
           * if the input axes are None, the same object is returned.
           * if the input axes are not None, they will be set as-is and now correspond to the already
             existing obs in the object.

        Args:
            axes: Axes to sort/associate this instance with
            allow_superset: if False and a strict superset of the own axeservables is given, an error
            is raised.
            allow_subset:if False and a strict subset of the own axeservables is given, an error
            is raised.

        Returns:
            A copy of the object with the new ordering/axes
        Raises:
            CoordinatesUnderdefinedError: if obs is None and the instance does not have axes
            AxesIncompatibleError: if `axes` is a superset and allow_superset is False or a subset and
                allow_allow_subset is False
        """
        spaces = [
            space.with_axes(axes, allow_superset=allow_superset, allow_subset=allow_subset) for space in self.spaces
        ]
        coords = self.coords.with_axes(axes, allow_subset=allow_subset, allow_superset=allow_superset)
        return self.copy(spaces=spaces, obs=coords.obs, axes=coords.axes)

    def with_coords(
        self,
        coords: ZfitOrderableDimensional,
        allow_superset: bool = True,
        allow_subset: bool = True,
    ) -> MultiSpace:
        """Create a new :py:class:`~zfit.Space` with reordered observables and/or axes.

        The behavior is that _at least one coordinate (obs or axes) has to be set in both instances
        (the space itself or in `coords`). If both match, observables is taken as the defining coordinate.
        The space is sorted according to the defining coordinate and the other coordinate is sorted as well.
        If either the space did not have the "weaker coordinate" (e.g. both have observables, but only coords
        has axes), then the resulting Space will have both.
        If both have both coordinates, obs and axes, and sorting for obs results in non-matchin axes results
        in axes being dropped.

        Args:
            coords: An instance of :py:class:`Coordinates`
            allow_superset: If false and a strict superset is given, an error is raised
            allow_subset: If false and a strict subset is given, an error is raised

        Returns:
            :py:class:`~zfit.Space`:
        Raises:
            CoordinatesUnderdefinedError: if neither both obs or axes are specified.
            CoordinatesIncompatibleError: if `coords` is a superset and allow_superset is False or a subset and
                allow_allow_subset is False
        """
        new_spaces = [
            space.with_coords(coords, allow_superset=allow_superset, allow_subset=allow_subset) for space in self
        ]
        return type(self)(spaces=new_spaces)

    def with_autofill_axes(self, overwrite: bool = False) -> MultiSpace:
        """Overwrite the axes of the current object with axes corresponding to range(len(n_obs)).

        This effectively fills with (0, 1, 2,...) and can be used mostly when an object enters a PDF or
        similar. `overwrite` allows to remove the axis first in case there are already some set.

        .. code-block::

            object.obs -> ('x', 'z', 'y')
            object.axes -> None

            object.with_autofill_axes()

            object.obs -> ('x', 'z', 'y')
            object.axes -> (0, 1, 2)


        Args:
            overwrite: If axes are already set, replace the axes with the autofilled ones.
                If axes is already set and `overwrite` is False, raise an error.

        Returns:
            The object with the new axes

        Raises:
            AxesIncompatibleError: if the axes are already set and `overwrite` is False.
        """
        spaces = [space.with_autofill_axes(overwrite=overwrite) for space in self]
        return self.copy(spaces=spaces)

    def get_subspace(
        self,
        obs: ztyping.ObsTypeInput = None,
        axes: ztyping.AxesTypeInput = None,
        name: str | None = None,
    ) -> MultiSpace:
        """Create a :py:class:`~zfit.Space` consisting of only a subset of the `obs`/`axes` (only one allowed).

        Args:
            obs: Observables of the subspace to return.
            axes: Axes of the subspace to return.
            name: Human readable names

        Returns:
            A space containing only a subspace (and sublimits etc.)
        """
        spaces = [space.get_subspace(obs=obs, axes=axes) for space in self.spaces]
        return self.copy(spaces=spaces, name=name)

    def copy(self, *, deep: bool = False, name: str | None = None, **overwrite_params) -> MultiSpace:
        assert not deep, "deep not explicitly implemented, should not be needed for immutable objects"
        kwargs = {
            "spaces": tuple(self),
            "obs": self.obs,
            "axes": self.axes,
        }
        kwargs.update(overwrite_params)
        kwargs["name"] = self.name if name is None else name
        return type(self)(**kwargs)

    def _raise_limits_not_implemented(self):
        msg = (
            "Limits/lower/upper not implemented for MultiSpace. This error is either caught"
            " automatically as part of the codes logic or the MultiLimit case should"
            " be considered. To do that, simply iterate through the MultiSpace, which returns"
            " a simple space. Iterating through a Spaces also works"
            "for simple spaces."
        )
        raise MultipleLimitsNotImplemented(msg)

    def _inside(self, x, guarantee_limits):
        inside_limits = [space.inside(x, guarantee_limits=guarantee_limits) for space in self]
        return znp.any(inside_limits, axis=0)  # has to be inside one limit

    def __iter__(self) -> ZfitSpace:
        yield from self.spaces

    def __repr__(self):
        class_name = str(self.__class__).split(".")[-1].split("'")[0]
        if not self.limits_are_set:
            limits = None
        elif self.limits_are_false:
            limits = False
        elif self.has_rect_limits:
            if self.n_obs < 3 and not self.n_events > 1 and self._depr_n_limits <= 3:
                limits = [lim.v0.limits for lim in self]
            else:
                limits = "rectangular"
        else:
            limits = "functional"
        return f"<zfit {class_name} obs={self.obs}, axes={self.axes}, limits={limits}>"

    def __eq__(self, other):
        if not isinstance(other, MultiSpace):
            warnings.warn("Multispace limits compare never equal to Space.", stacklevel=2)
        return equal_space(self, other, allow_graph=False)

    def __le__(self, other):
        if not isinstance(other, MultiSpace):
            warnings.warn("Multispace limits compare never equal to Space.", stacklevel=2)
        return less_equal_space(self, other, allow_graph=False)

    def __hash__(self):
        return hash(self.spaces)


def convert_to_space(
    obs: ztyping.ObsTypeInput | None = None,
    axes: ztyping.AxesTypeInput | None = None,
    limits: ztyping.LimitsTypeInput | None = None,
    *,
    overwrite_limits: bool = False,
    one_dim_limits_only: bool = True,
    simple_limits_only: bool = True,
) -> None | ZfitSpace | bool:
    """Convert *limits* to a :py:class:`~zfit.Space` object if not already None or False.

    Args:
        obs:
        limits:
        axes:
        overwrite_limits: If `obs` or `axes` is a :py:class:`~zfit.Space` _and_ `limits` are given, return an instance
            of :py:class:`~zfit.Space` with the new limits. If the flag is `False`, the `limits` argument will be
            ignored if
        one_dim_limits_only:
        simple_limits_only:

    Returns:
        Union[:py:class:`~zfit.Space`, False, None]:

    Raises:
        OverdefinedError: if `obs` or `axes` is a :py:class:`~zfit.Space` and `axes` respectively `obs` is not `None`.
    """
    space = None

    if isinstance(obs, ZfitSpace):
        if axes is not None:
            msg = "if `obs` is a `Space`, `axes` cannot be defined."
            raise OverdefinedError(msg)
        space = obs
    elif isinstance(axes, ZfitSpace):
        if obs is not None:
            msg = "if `axes` is a `Space`, `obs` cannot be defined."
            raise OverdefinedError(msg)
        space = axes
    elif isinstance(limits, ZfitSpace):
        return limits
    if space is not None:
        if limits is not None and (overwrite_limits or not space.limits_are_set):
            if isinstance(limits, ZfitSpace):  # figure out if compatible if limits is `Space`
                if not (
                    limits.obs == space.obs or (limits.axes == space.axes and limits.obs is None and space.obs is None)
                ):
                    msg = "`obs`/`axes` is a `Space` as well as the `limits`, but the obs/axes of them do not match"
                    raise IntentionAmbiguousError(msg)
                limits = False if limits.limits_are_false else limits.limits

            space = space.with_limits(limits=limits)
        return space

    if not (obs is None and axes is None):
        space = Space(obs=obs, axes=axes, limits=limits)  # create and test if valid
        if one_dim_limits_only and space.n_obs > 1 and space.has_limits:
            msg = "Limits more sophisticated than 1-dim cannot be auto-created from tuples. Use `Space` instead."
            raise LimitsUnderdefinedError(msg)
        if simple_limits_only and space.has_limits and space._depr_n_limits > 1:
            msg = "Limits with multiple limits cannot be auto-created from tuples. Use `Space` instead."
            raise LimitsUnderdefinedError(msg)
    return space


def check_norm(supports=None):
    if supports is None:
        supports = False
    supports = convert_to_container(supports, convert_none=True)

    def no_norm_range(func):
        pass

    return no_norm_range


def param_args_supported(func):
    pass


def no_multiple_limits(func):
    pass


@deprecated_norm_range
def supports(
    *,
    norm: bool | str | Iterable[str] | None = None,
    multiple_limits: bool | None = None,
) -> Callable:
    """Decorator: Add (mandatory for some methods) on a method to control what it can handle.

    If any of the flags is set to False, it will check the arguments and, in case they match a flag
    (say if a *norm* is passed while the *norm* flag is set to `False`), it will
    raise a corresponding exception (in this example a `NormRangeNotImplementedError`) that will
    be catched by an earlier function that knows how to handle things.

    Args:
        norm: If False, no norm_range argument will be passed through resp. will be `None`.
            Other options include `'space'` or `'norm'`, which will check if the norm is equal to
            the space or norm of the PDF. If they are, it is assumed to be supported.
        multiple_limits: If False, only simple limits are to be expected and no iteration is
            therefore required.
    """
    if norm is None:
        norm = False
    if multiple_limits is None:
        multiple_limits = False

    decorator_stack = []
    if not multiple_limits:
        decorator_stack.append(no_multiple_limits)

    if norm is not None:  # check True. Could also be a str
        decorator_stack.append(check_norm(norm))

    decorator_stack.append(param_args_supported)


    return create_deco_stack


def contains_tensor(objects):
    tensor_found = tf.is_tensor(objects)
    with suppress(TypeError):
        for obj in objects:
            if tensor_found:
                break
            tensor_found += contains_tensor(obj)
    return tensor_found




def limits_consistent(spaces: Iterable[zfit.Space]):
    """Check if space limits are the *exact* same in each obs they are defined and therefore are compatible.

    In this case, if a space has several limits, e.g. from -1 to 1 and from 2 to 3 (all in the same observable),
    to be consistent with this limits, other limits have to have (in this obs) also the limits
    from -1 to 1 and from 2 to 3. Only having the limit -1 to 1 _or_ 2 to 3 is considered _not_ consistent.

    This function is useful to check if several spaces with *different* observables can be _combined_.

    Args:
        spaces:

    Returns:
    """
    try:
        _ = combine_spaces(*spaces)
    except LimitsIncompatibleError:
        return False
    else:
        return True


def add_spaces_old(spaces: Iterable[zfit.Space]):
    pass
