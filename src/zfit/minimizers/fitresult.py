
from __future__ import annotations

import collections
import contextlib
import itertools
import typing
import warnings
from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING

import colored
import iminuit
import numpy as np
import scipy.optimize
import scipy.stats
from colorama import Style, init
from ordered_set import OrderedSet
from scipy.optimize import LbfgsInvHessProduct
from tabulate import tabulate

from ..util.exception import BreakingAPIChangeError

if typing.TYPE_CHECKING:
    import zfit

    from .evaluation import LossEval

if TYPE_CHECKING:
    with contextlib.suppress(ImportError):  # for type checking
        import ipyopt

from zfit._interfaces import (
    ZfitData,
    ZfitIndependentParameter,
    ZfitLoss,
    ZfitParameter,
    ZfitUnbinnedData,
)

from ..core.parameter import set_values
from ..util.container import convert_to_container
from ..util.deprecation import deprecated, deprecated_args
from ..util.warnings import ExperimentalFeatureWarning
from ..util.ztyping import ParamsTypeOpt
from ..z import numpy as znp
from .errors import (
    WeightCorr,
    compute_errors,
    covariance_with_weights,
    dict_to_matrix,
    matrix_to_dict,
)
from .interface import ZfitMinimizer, ZfitResult
from .termination import ConvergenceCriterion

init(autoreset=True)


class Approximations:
    def __init__(
        self,
        params: list[ZfitParameter] | tuple[ZfitParameter],
        gradient: np.ndarray | None = None,
        hessian: np.ndarray | None = None,
        inv_hessian: np.ndarray | None = None,
    ) -> None:
        """Holds different approximations after the minimisation and/or calculates them.

        Args:
            params: List of parameters the approximations (gradient, hessian, ...) were calculated with.
            gradient: Gradient
            hessian: Hessian Matrix
            inv_hessian: Inverse of the Hessian Matrix
        """
        if isinstance(params, dict):
            params = list(params)
        if not isinstance(params, list | tuple):
            msg = f"params has to be a list or tuple, not {type(params)}"
            raise TypeError(msg)
        if gradient is not None:
            if gradient.ndim > 1:
                msg = f"Gradient has to be a 1D array, not {gradient.shape}"
                raise ValueError(msg)
            if gradient.shape[0] != len(params):
                msg = f"Gradient has to have the same length as params, {gradient.shape[0]} != {len(params)}"
                raise ValueError(msg)

        if hessian is not None:
            if hessian.ndim != 2:
                msg = f"Hessian has to be a 2D array, not {hessian.shape}"
                raise ValueError(msg)
            if hessian.shape[0] != len(params) or hessian.shape[1] != len(params):
                msg = f"Hessian has to have the same length as params, {hessian.shape} != {len(params)}"
                raise ValueError(msg)

        if inv_hessian is not None:
            if inv_hessian.ndim != 2:
                msg = f"Inverse Hessian has to be a 2D array, not {inv_hessian.shape}"
                raise ValueError(msg)
            if inv_hessian.shape[0] != len(params) or inv_hessian.shape[1] != len(params):
                msg = f"Inverse Hessian has to have the same length as params, {inv_hessian.shape} != {len(params)}"
                raise ValueError(msg)

        self._params = params
        self._gradient = gradient
        self._hessian = hessian
        self._inv_hessian = inv_hessian
        super().__init__()


    def gradient(self, params: ZfitParameter | Iterable[ZfitParameter] | None = None) -> np.ndarray | None:
        """Return an approximation of the gradient _if available_.

        Args:
            params: Parameters to which the gradients should be returned

        Returns:
            Array with gradients or ``None``
        """
        grad = self._gradient
        if grad is None:
            return None

        if params is not None:
            params = convert_to_container(params, container=tuple)
            params_mapped = {i: params.index(param) for i, param in enumerate(self.params) if param in params}
            indices = sorted(params_mapped, key=lambda x: params_mapped[x])
            grad = grad[indices]
        return grad

    def hessian(self, invert: bool = True) -> np.ndarray | None:
        """Return an approximation of the hessian _if available_.

        Args:
            invert: If a _hessian approximation_ is not available but an inverse hessian is, invert the latter to
                obtain the hessian approximation.

        Returns:
            Array with hessian matrix or ``None``
        """
        hess = self._hessian
        if hess is None and invert:
            inv_hess = self._inv_hessian
            if inv_hess is not None:
                hess = np.linalg.inv(inv_hess)
                self._hessian = hess
        return hess

    def inv_hessian(self, invert: bool = True) -> None | np.ndarray:
        """Return an approximation of the inverse hessian _if available_.

        Args:
            invert: If an _inverse hessian approximation_ is not available but a hessian is, invert the latter to
                obtain the inverse hessian approximation.

        Returns:
            Array with the inverse of the hessian matrix or ``None``
        """
        inv_hess = self._inv_hessian
        if inv_hess is None and invert:
            hess = self._hessian
            if hess is not None:
                inv_hess = np.linalg.inv(hess)
                self._inv_hessian = inv_hess
        return inv_hess











class ParamToNameGetitem:
    __slots__ = ()

    def __getitem__(self, item):
        if isinstance(item, ZfitParameter):
            item = item.name
        return super().__getitem__(item)


class NameToParamGetitem:
    __slots__ = ()

    def __getitem__(self, item):
        if isinstance(item, ZfitParameter):
            item = item.name
        for param in self.keys():
            name = param.name if isinstance(param, ZfitParameter) else param
            if name == item:
                item = param
                break
        return super().__getitem__(item)  # raises key error if not there, which is good

    def __contains__(self, item):
        try:
            self[item]
        except KeyError:
            return False
        except Exception as error:
            msg = "Unknown exception occurred! This should not happen."
            raise RuntimeError(msg) from error
        else:
            return True


class OptimizeResultMixin:

    @property
    def success(self) -> bool:
        pass

    @property
    def fun(self) -> float:
        pass

    @property
    def jac(self) -> np.ndarray | None:
        pass

    @property
    def hess(self) -> np.ndarray | None:
        pass

    @property
    def hess_inv(self) -> np.ndarray | None:
        pass

    @property
    def nfev(self) -> int | None:
        pass

    @property
    def njev(self) -> int | None:
        pass

    @property
    def nhev(self) -> int | None:
        pass

    @property
    def nit(self) -> int | None:
        pass

    @property
    def maxcv(self) -> float | None:
        pass


class FitResult(OptimizeResultMixin, ZfitResult):

    _default_hesse = "minuit_hesse"
    _hesse_methods: typing.ClassVar = {
        "minuit_hesse": _covariance_minuit,
        "hesse_np": _covariance_np,
        "approx": _covariance_approx,
    }
    _default_error = "minuit_minos"
    _error_methods: typing.ClassVar = {
        "minuit_minos": _minos_minuit,
        "zfit_error": compute_errors,
        "zfit_errors": compute_errors,
    }

    def __init__(
        self,
        loss: ZfitLoss,
        params: dict[ZfitParameter, float],
        minimizer: ZfitMinimizer,
        valid: bool,
        edm: float,
        fminopt: float,
        criterion: ConvergenceCriterion | None,
        status: int | None = None,
        converged: bool | None = None,
        message: str | None = None,
        info: Mapping | None = None,
        approx: Mapping | Approximations | None = None,
        niter: int | None = None,
        evaluator: LossEval = None,
    ) -> None:
        """Create a ``FitResult`` from a minimization. Store parameter values, minimization infos and calculate errors.

        .. _info-fields:
        .. _field:

        Any errors calculated are saved under ``self.params`` dictionary with::

            {parameter: {error_name1: {'low': value, 'high': value or similar}}

        The ``FitResult`` can be used to temporarily update the parameter values to the values found by the minimization

        .. code-block::

            with result:
                # do something with the new values
                ...

        Args:
            loss: |@doc:result.init.loss| The loss function that was minimized.
               Usually, but not necessary, contains
               also the pdf, data and constraints. |@docend:result.init.loss|
            params: |@doc:result.init.params| Result of the fit where each
               :py:class:`~zfit.Parameter` key has the
               value from the minimum found by the minimizer. |@docend:result.init.params|
            minimizer: |@doc:result.init.minimizer| Minimizer that was used to obtain this ``FitResult`` and will be used to
                   calculate certain errors. If the minimizer
                   is state-based (like "iminuit"), then this is a copy
                   and the state of other ``FitResults`` or of the *actual*
                   minimizer that performed the minimization
                   won't be altered. |@docend:result.init.minimizer|
            valid: |@doc:result.init.valid| Indicating whether the result is valid or not. This is the strongest
                   indication and serves as
                   the global flag. The reasons why a result may be
                   invalid can be arbitrary, including but not exclusive:

                   - parameter(s) at the limit
                   - maxiter reached without proper convergence
                   - the minimizer maybe even converged but it is known
                     that this is only a local minimum

                   To indicate the reason for the invalidity, pass a message. |@docend:result.init.valid|
            edm: |@doc:result.init.edm| The estimated distance to minimum
                   which is the criterion value at the minimum. |@docend:result.init.edm|
            fminopt: |@doc:result.init.fmin| Value of the function at the minimum. |@docend:result.init.fmin|
            criterion: |@doc:result.init.criterion| Criterion that was used during the minimization.
                   This determines the estimated distance to the
                   minimum (edm) |@docend:result.init.criterion|
            status: |@doc:result.init.status| A status code (if available) that describes
                   the minimization termination. 0 means a valid
                   termination. |@docend:result.init.status|
            converged: |@doc:result.init.converged| Whether the fit has successfully converged or not.
                   The result itself can still be an invalid minimum
                   such as if the parameters are at or close
                   to the limits or in case another minimum is found. |@docend:result.init.converged|
            message: |@doc:result.init.message| Human-readable message to indicate the reason
                   if the fitresult is not valid.
                   If the fit is valid, the message (should)
                   be an empty string (or None),
                   otherwise, it should denote the reason for the invalidity. |@docend:result.init.message|
            info: |@doc:result.init.info| Additional information (if available)
                   such as *number of gradient function calls* or the
                   original minimizer return message.
                   This is a relatively free field and _no single field_
                   in it is guaranteed to be stable.
                   Some recommended fields:

                   - *original*: contains the original returned object
                     by the minimizer used internally.
                   - *optimizer*: the actual instance of the wrapped
                     optimizer (if available) |@docend:result.init.info|
            approx: |@doc:result.init.approx| Collection of approximations found during
                   the minimization process such as gradient and hessian. |@docend:result.init.approx|
            niter: |@doc:result.init.niter| Approximate number of iterations ~= number
                   of function evaluations ~= number of gradient evaluations.
                   This is an approximated value and the exact meaning
                   can differ between different minimizers. |@docend:result.init.niter|
            evaluator: |@doc:result.init.evaluator| Loss evaluator that was used during the
                   minimization and that may contain information
                   about the last evaluations of the gradient
                   etc. which can serve as approximations. |@docend:result.init.evaluator|
        """
        super().__init__()

        if status is None:
            status = 0 if valid else -999
        if converged is None and valid:
            converged = True
        if message is None:
            message = "" if valid else "Invalid, unknown reason (not specified)"

        approx = self._input_convert_approx(approx, evaluator, info, params)

        if evaluator is not None:
            niter = evaluator.niter if niter is None else niter

        info = {"n_eval": niter} if info is None else info
        param_at_limit = any(param.at_limit for param in params)
        if param_at_limit:
            valid = False
            if message:
                message += " AND "
            message += "parameter(s) at their limit."

        self._cache_minuit = None  # in case used in errors
        self._is_frozen = False  # flag to prevent double freezing

        self._evaluator = evaluator  # keep private for now
        self._niter = niter  # keep private for now
        self._approx = approx
        self._status = status
        self._message = "" if message is None else message
        self._converged = converged
        self._params = self._input_convert_params(params)
        self._values = ValuesHolder(params)
        self._params_at_limit = param_at_limit
        self._edm = edm
        self._criterion = criterion
        self._fminopt = fminopt
        self._info = info
        self._loss = loss
        self._minimizer = minimizer
        self._valid = valid
        self._covariance_dict = {}
        self._tmp_old_param_values = []
        try:
            fminfull = loss.value(full=True)
        except Exception as error:
            warnings.warn(
                f"Could not calculate fminfull due to {error}. Setting to 0. This is a new feature and is caught to not break backwards compatibility.",
                RuntimeWarning,
                stacklevel=3,
            )
            fminfull = 0
        self._fmin = float(fminfull)

    def _input_convert_approx(self, approx, evaluator, info, params):
        """Convert approx (if a Mapping) to an `Approximation` using the information provided.

        Args:
            approx: |@doc:result.init.approx| Collection of approximations found during
                   the minimization process such as gradient and hessian. |@docend:result.init.approx|
            evaluator: |@doc:result.init.evaluator| Loss evaluator that was used during the
                   minimization and that may contain information
                   about the last evaluations of the gradient
                   etc. which can serve as approximations. |@docend:result.init.evaluator|
            info: |@doc:result.init.info| Additional information (if available)
                   such as *number of gradient function calls* or the
                   original minimizer return message.
                   This is a relatively free field and _no single field_
                   in it is guaranteed to be stable.
                   Some recommended fields:

                   - *original*: contains the original returned object
                     by the minimizer used internally.
                   - *optimizer*: the actual instance of the wrapped
                     optimizer (if available) |@docend:result.init.info|
            params: |@doc:result.init.params| Result of the fit where each
               :py:class:`~zfit.Parameter` key has the
               value from the minimum found by the minimizer. |@docend:result.init.params|

        Returns:
            The created approximation.
        """
        approx = {} if approx is None else approx
        if isinstance(approx, collections.abc.Mapping):
            if "params" not in approx:
                approx["params"] = params

            if info:
                if "gradient" not in approx:
                    approx["gradient"] = info.get("grad", info.get("gradient"))
                if "hessian" not in approx:
                    approx["hessian"] = info.get("hess", info.get("hesse", info.get("hessian")))
                if "inv_hessian" not in approx:
                    approx["inv_hessian"] = info.get("inv_hess", info.get("inv_hesse", info.get("inv_hessian")))
            if evaluator is not None:
                if "gradient" not in approx:
                    approx["gradient"] = evaluator.last_gradient
                if "hessian" not in approx:
                    approx["hessian"] = evaluator.last_hessian

            approx = Approximations(**approx)
        return approx

    def _input_convert_params(self, params):
        return ParamHolder((p, {"value": v}) for p, v in params.items())

    def _check_get_uncached_params(self, params, method_name, cl, weightcorr=None):
        uncached = []
        for p in params:
            errordict = self.params[p].get(method_name)
            if errordict is not None:
                if round(errordict["cl"], 3) != round(cl, 3):
                    msg = (
                        f"Error with name {method_name} already exists in {self!r} with a different"
                        f" convidence level of {errordict['cl']} instead of the requested {cl}."
                        f" Use a different name."
                    )
                    raise NameError(
                        msg,
                    )
                if weightcorr is not None and weightcorr != errordict.get("weightcorr", WeightCorr.FALSE):
                    msg = (
                        f"Error with name {method_name} already exists in {self!r} with a different"
                        f" weight correction of {errordict['weightcorr']} instead of the requested {weightcorr}."
                        f" Use a different name."
                    )
                    raise NameError(
                        msg,
                    )

            uncached.append(p)
        return uncached


    @classmethod
    def from_ipopt(
        cls,
        loss: ZfitLoss,
        params: Iterable[ZfitParameter],
        problem: ipyopt.Problem,
        minimizer: zfit.minimize.Ipyopt,
        valid: bool,
        values: np.ndarray,
        message: str | None,
        converged: bool | None,
        edm: zfit.minimizers.termination.CriterionNotAvailable | float,
        niter: int | None,
        fminopt: float | None,
        status: int | None,
        criterion: zfit.minimizers.termination.ConvergenceCriterion,
        evaluator: zfit.minimizers.evaluation.LossEval | None,
    ) -> FitResult:
        """Create a ``FitResult`` from an ipopt minimization.

        Args:
            loss: |@doc:result.init.loss| The loss function that was minimized.
               Usually, but not necessary, contains
               also the pdf, data and constraints. |@docend:result.init.loss|
            params: |@doc:result.init.params| Result of the fit where each
               :py:class:`~zfit.Parameter` key has the
               value from the minimum found by the minimizer. |@docend:result.init.params|
            problem: The ipopt problem instance that was used for the minimization.
            minimizer: |@doc:result.init.minimizer| Minimizer that was used to obtain this ``FitResult`` and will be used to
                   calculate certain errors. If the minimizer
                   is state-based (like "iminuit"), then this is a copy
                   and the state of other ``FitResults`` or of the *actual*
                   minimizer that performed the minimization
                   won't be altered. |@docend:result.init.minimizer|
            valid: |@doc:result.init.valid| Indicating whether the result is valid or not. This is the strongest
                   indication and serves as
                   the global flag. The reasons why a result may be
                   invalid can be arbitrary, including but not exclusive:

                   - parameter(s) at the limit
                   - maxiter reached without proper convergence
                   - the minimizer maybe even converged but it is known
                     that this is only a local minimum

                   To indicate the reason for the invalidity, pass a message. |@docend:result.init.valid|
            values: |@doc:result.init.values| Values of the parameters at the
                   found minimum. |@docend:result.init.values|
            message: |@doc:result.init.message| Human-readable message to indicate the reason
                   if the fitresult is not valid.
                   If the fit is valid, the message (should)
                   be an empty string (or None),
                   otherwise, it should denote the reason for the invalidity. |@docend:result.init.message|
            converged: |@doc:result.init.converged| Whether the fit has successfully converged or not.
                   The result itself can still be an invalid minimum
                   such as if the parameters are at or close
                   to the limits or in case another minimum is found. |@docend:result.init.converged|
            edm: |@doc:result.init.edm| The estimated distance to minimum
                   which is the criterion value at the minimum. |@docend:result.init.edm|
            niter: |@doc:result.init.niter| Approximate number of iterations ~= number
                   of function evaluations ~= number of gradient evaluations.
                   This is an approximated value and the exact meaning
                   can differ between different minimizers. |@docend:result.init.niter|
            fminopt: |@doc:result.init.fmin| Value of the function at the minimum. |@docend:result.init.fmin|
            status: |@doc:result.init.status| A status code (if available) that describes
                   the minimization termination. 0 means a valid
                   termination. |@docend:result.init.status|
            criterion: |@doc:result.init.criterion| Criterion that was used during the minimization.
                   This determines the estimated distance to the
                   minimum (edm) |@docend:result.init.criterion|
            evaluator: |@doc:result.init.evaluator| Loss evaluator that was used during the
                   minimization and that may contain information
                   about the last evaluations of the gradient
                   etc. which can serve as approximations. |@docend:result.init.evaluator|

        Returns:
            ``zfit.minimize.FitResult``:
        """
        info = {"problem": problem}
        params = dict(zip(params, values, strict=True))
        valid = valid if converged is None else valid and converged
        if evaluator is not None:
            valid = valid and not evaluator.maxiter_reached
        return cls(
            params=params,
            loss=loss,
            fminopt=fminopt,
            edm=edm,
            message=message,
            criterion=criterion,
            info=info,
            valid=valid,
            converged=converged,
            niter=niter,
            status=status,
            minimizer=minimizer,
            evaluator=evaluator,
        )

    @classmethod
    def from_minuit(
        cls,
        loss: ZfitLoss,
        params: Iterable[ZfitParameter],
        minuit: iminuit.Minuit,
        minimizer: ZfitMinimizer | iminuit.Minuit,
        valid: bool | None,
        values: np.ndarray | None = None,
        message: str | None = None,
        converged: bool | None = None,
        edm: None | (zfit.minimizers.termination.CriterionNotAvailable | float) = None,
        niter: int | None = None,
        fminopt: float | None = None,
        status: int | None = None,
        criterion: zfit.minimizers.termination.ConvergenceCriterion | None = None,
        evaluator: zfit.minimizers.evaluation.LossEval | None = None,
    ) -> FitResult:
        """Create a `FitResult` from a :py:class:`~iminuit.util.MigradResult` returned by
        :py:meth:`iminuit.Minuit.migrad` and a iminuit :py:class:`~iminuit.Minuit` instance with the corresponding zfit
        objects.

        Args:
            loss: zfit Loss that was minimized.
            params: Iterable of the zfit parameters that were floating during the minimization.
            minuit: Return value of the iminuit migrad command, the instance of :class:`iminuit.Minuit`
            minimizer: Instance of the zfit Minuit minimizer that was used to minimize the loss.
            valid: |@doc:result.init.valid| Indicating whether the result is valid or not. This is the strongest
                   indication and serves as
                   the global flag. The reasons why a result may be
                   invalid can be arbitrary, including but not exclusive:

                   - parameter(s) at the limit
                   - maxiter reached without proper convergence
                   - the minimizer maybe even converged but it is known
                     that this is only a local minimum

                   To indicate the reason for the invalidity, pass a message. |@docend:result.init.valid|
            values: |@doc:result.init.values| Values of the parameters at the
                   found minimum. |@docend:result.init.values|
            message: |@doc:result.init.message| Human-readable message to indicate the reason
                   if the fitresult is not valid.
                   If the fit is valid, the message (should)
                   be an empty string (or None),
                   otherwise, it should denote the reason for the invalidity. |@docend:result.init.message|
            converged: |@doc:result.init.converged| Whether the fit has successfully converged or not.
                   The result itself can still be an invalid minimum
                   such as if the parameters are at or close
                   to the limits or in case another minimum is found. |@docend:result.init.converged|
            edm: |@doc:result.init.edm| The estimated distance to minimum
                   which is the criterion value at the minimum. |@docend:result.init.edm|
            niter: |@doc:result.init.niter| Approximate number of iterations ~= number
                   of function evaluations ~= number of gradient evaluations.
                   This is an approximated value and the exact meaning
                   can differ between different minimizers. |@docend:result.init.niter|
            fminopt: |@doc:result.init.fmin| Value of the function at the minimum. |@docend:result.init.fmin|
            status: |@doc:result.init.status| A status code (if available) that describes
                   the minimization termination. 0 means a valid
                   termination. |@docend:result.init.status|
            criterion: |@doc:result.init.criterion| Criterion that was used during the minimization.
                   This determines the estimated distance to the
                   minimum (edm) |@docend:result.init.criterion|
            evaluator: |@doc:result.init.evaluator| Loss evaluator that was used during the
                   minimization and that may contain information
                   about the last evaluations of the gradient
                   etc. which can serve as approximations. |@docend:result.init.evaluator|


            Returns:
                ``zfit.minimize.FitResult``: A `FitResult` as if zfit Minuit was used.
        """
        from .minimizer_minuit import Minuit  # noqa: PLC0415
        from .termination import EDM  # noqa: PLC0415

        if not isinstance(minimizer, Minuit):
            if isinstance(minimizer, iminuit.Minuit):
                minimizer_new = Minuit()
                minimizer_new._minuit_minimizer = minimizer
                minimizer = minimizer_new
            else:
                msg = f"Minimizer {minimizer} not supported. Use `Minuit` from zfit or from iminuit."
                raise ValueError(msg)

        params_result = list(minuit.params)

        fmin_object = minuit.fmin
        minuit_converged = not fmin_object.is_above_max_edm
        converged = minuit_converged if converged is None else (converged and minuit_converged)
        niter = fmin_object.nfcn if niter is None else niter
        info = {
            "n_eval": niter,
            "minuit": minuit,
            "original": fmin_object,
        }
        if fmin_object.has_covariance and (minuitcov := minuit.covariance) is not None:
            info["inv_hessian"] = np.array(minuitcov)

        edm = fmin_object.edm if edm is None else edm
        if criterion is None:
            criterion = EDM(tol=minimizer.tol, loss=loss, params=params)
            criterion.last_value = edm
        fminopt = fmin_object.fval if fminopt is None else fminopt
        minuit_valid = fmin_object.is_valid
        valid = minuit_valid if valid is None else minuit_valid and valid
        if evaluator is not None:
            valid = valid and not evaluator.maxiter_reached
        if values is None:
            values = tuple(res.value for res in params_result)
        params = dict(zip(params, values, strict=True))
        return cls(
            params=params,
            edm=edm,
            fminopt=fminopt,
            info=info,
            loss=loss,
            niter=niter,
            converged=converged,
            status=status,
            message=message,
            valid=valid,
            criterion=criterion,
            minimizer=minimizer,
            evaluator=evaluator,
        )

    @classmethod
    def from_scipy(
        cls,
        loss: ZfitLoss,
        params: Iterable[ZfitParameter],
        result: scipy.optimize.OptimizeResult,
        minimizer: ZfitMinimizer,
        message: str | None,
        valid: bool,
        criterion: ConvergenceCriterion,
        edm: float | None = None,
        niter: int | None = None,
        evaluator: zfit.minimize.LossEval | None = None,
    ) -> FitResult:
        """Create a ``FitResult`` from a SciPy ``~scipy.optimize.OptimizeResult``.

        Args:
            loss: |@doc:result.init.loss| The loss function that was minimized.
               Usually, but not necessary, contains
               also the pdf, data and constraints. |@docend:result.init.loss|
            params: |@doc:result.init.params| Result of the fit where each
               :py:class:`~zfit.Parameter` key has the
               value from the minimum found by the minimizer. |@docend:result.init.params|
            result: Result of the SciPy optimization.
            minimizer: |@doc:result.init.minimizer| Minimizer that was used to obtain this ``FitResult`` and will be used to
                   calculate certain errors. If the minimizer
                   is state-based (like "iminuit"), then this is a copy
                   and the state of other ``FitResults`` or of the *actual*
                   minimizer that performed the minimization
                   won't be altered. |@docend:result.init.minimizer|
            message: |@doc:result.init.message| Human-readable message to indicate the reason
                   if the fitresult is not valid.
                   If the fit is valid, the message (should)
                   be an empty string (or None),
                   otherwise, it should denote the reason for the invalidity. |@docend:result.init.message|
            edm: |@doc:result.init.edm| The estimated distance to minimum
                   which is the criterion value at the minimum. |@docend:result.init.edm|
            niter: |@doc:result.init.niter| Approximate number of iterations ~= number
                   of function evaluations ~= number of gradient evaluations.
                   This is an approximated value and the exact meaning
                   can differ between different minimizers. |@docend:result.init.niter|
            valid: |@doc:result.init.valid| Indicating whether the result is valid or not. This is the strongest
                   indication and serves as
                   the global flag. The reasons why a result may be
                   invalid can be arbitrary, including but not exclusive:

                   - parameter(s) at the limit
                   - maxiter reached without proper convergence
                   - the minimizer maybe even converged but it is known
                     that this is only a local minimum

                   To indicate the reason for the invalidity, pass a message. |@docend:result.init.valid|
            criterion: |@doc:result.init.criterion| Criterion that was used during the minimization.
                   This determines the estimated distance to the
                   minimum (edm) |@docend:result.init.criterion|
            evaluator: |@doc:result.init.evaluator| Loss evaluator that was used during the
                   minimization and that may contain information
                   about the last evaluations of the gradient
                   etc. which can serve as approximations. |@docend:result.init.evaluator|

        Returns:
            `zfit.minimize.FitResult`:
        """
        result_values = result["x"]
        if niter is None:
            niter = result.get("nit", 0)

        converged = result.get("success", valid)
        status = result["status"]

        if message is None and (not converged or not valid):
            message = result.get("message")
        grad = result.get("grad")
        info = {
            "n_eval": result["nfev"],
            "n_iter": niter,
            "niter": niter,
            "grad": result.get("jac") if grad is None else grad,
            "message": message,
            "evaluator": evaluator,
            "original": result,
        }
        approx = {
            "params": params,
            "gradient": info.get("grad"),
        }
        if info.get("niter", 0) > 25:  # unreliable if too few iterations, fails for EDM
            inv_hesse = result.get("hess_inv")
            if isinstance(inv_hesse, LbfgsInvHessProduct):
                inv_hesse = inv_hesse.todense()
            hesse = info.get("hesse")
            info["inv_hesse"] = inv_hesse
            info["hesse"] = hesse
            approx["hessian"] = hesse
            approx["inv_hessian"] = inv_hesse

        fminopt = result["fun"]
        params = dict(zip(params, result_values, strict=True))
        if evaluator is not None:
            valid = valid and not evaluator.maxiter_reached

        return cls(
            params=params,
            edm=edm,
            fminopt=fminopt,
            info=info,
            approx=approx,
            converged=converged,
            status=status,
            message=message,
            valid=valid,
            niter=niter,
            loss=loss,
            minimizer=minimizer,
            criterion=criterion,
            evaluator=evaluator,
        )

    @classmethod
    def from_nlopt(
        cls,
        loss: ZfitLoss,
        opt,
        params: Iterable[ZfitParameter],
        minimizer: ZfitMinimizer | iminuit.Minuit,
        valid: bool | None,
        values: np.ndarray | None = None,
        message: str | None = None,
        converged: bool | None = None,
        edm: None | (zfit.minimizers.termination.CriterionNotAvailable | float) = None,
        niter: int | None = None,
        fminopt: float | None = None,
        status: int | None = None,
        criterion: zfit.minimizers.termination.ConvergenceCriterion | None = None,
        evaluator: zfit.minimizers.evaluation.LossEval | None = None,
        inv_hessian: np.ndarray | None = None,
        hessian: np.ndarray | None = None,
    ) -> FitResult:
        """Create a ``FitResult`` from an NLopt optimizer.

        Args:
            loss: |@doc:result.init.loss| The loss function that was minimized.
               Usually, but not necessary, contains
               also the pdf, data and constraints. |@docend:result.init.loss|
            opt: Optimizer instance of NLopt
            params: |@doc:result.init.params| Result of the fit where each
               :py:class:`~zfit.Parameter` key has the
               value from the minimum found by the minimizer. |@docend:result.init.params|
            minimizer: |@doc:result.init.minimizer| Minimizer that was used to obtain this ``FitResult`` and will be used to
                   calculate certain errors. If the minimizer
                   is state-based (like "iminuit"), then this is a copy
                   and the state of other ``FitResults`` or of the *actual*
                   minimizer that performed the minimization
                   won't be altered. |@docend:result.init.minimizer|
            valid: |@doc:result.init.valid| Indicating whether the result is valid or not. This is the strongest
                   indication and serves as
                   the global flag. The reasons why a result may be
                   invalid can be arbitrary, including but not exclusive:

                   - parameter(s) at the limit
                   - maxiter reached without proper convergence
                   - the minimizer maybe even converged but it is known
                     that this is only a local minimum

                   To indicate the reason for the invalidity, pass a message. |@docend:result.init.valid|
            values: |@doc:result.init.values| Values of the parameters at the
                   found minimum. |@docend:result.init.values|
            message: |@doc:result.init.message| Human-readable message to indicate the reason
                   if the fitresult is not valid.
                   If the fit is valid, the message (should)
                   be an empty string (or None),
                   otherwise, it should denote the reason for the invalidity. |@docend:result.init.message|
            converged: |@doc:result.init.converged| Whether the fit has successfully converged or not.
                   The result itself can still be an invalid minimum
                   such as if the parameters are at or close
                   to the limits or in case another minimum is found. |@docend:result.init.converged|
            edm: |@doc:result.init.edm| The estimated distance to minimum
                   which is the criterion value at the minimum. |@docend:result.init.edm|
            niter: |@doc:result.init.niter| Approximate number of iterations ~= number
                   of function evaluations ~= number of gradient evaluations.
                   This is an approximated value and the exact meaning
                   can differ between different minimizers. |@docend:result.init.niter|
            fminopt: |@doc:result.init.fmin| Value of the function at the minimum. |@docend:result.init.fmin|
            status: |@doc:result.init.status| A status code (if available) that describes
                   the minimization termination. 0 means a valid
                   termination. |@docend:result.init.status|
            criterion: |@doc:result.init.criterion| Criterion that was used during the minimization.
                   This determines the estimated distance to the
                   minimum (edm) |@docend:result.init.criterion|
            evaluator: |@doc:result.init.evaluator| Loss evaluator that was used during the
                   minimization and that may contain information
                   about the last evaluations of the gradient
                   etc. which can serve as approximations. |@docend:result.init.evaluator|
            inv_hessian: The (approximated) inverse hessian matrix.
            hessian: The (approximated) hessian matrix.

        Returns:
            zfit.minimizers.fitresult.FitResult:
        """
        converged = converged if converged is None else bool(converged)
        param_dict = dict(zip(params, values, strict=True))
        if fminopt is None:
            fminopt = opt.last_optimum_value()
        status_nlopt = opt.last_optimize_result()
        if status is None:
            status = status_nlopt
        niter = opt.get_numevals() if niter is None else niter
        converged = 1 <= status_nlopt <= 4 and converged is not False

        messages = {
            1: "NLOPT_SUCCESS",
            2: "NLOPT_STOPVAL_REACHED",
            3: "NLOPT_FTOL_REACHED",
            4: "NLOPT_XTOL_REACHED",
            5: "NLOPT_MAXEVAL_REACHED",
            6: "NLOPT_MAXTIME_REACHED",
            -1: "NLOPT_FAILURE",
            -2: "NLOPT_INVALID_ARGS",
            -3: "NLOPT_OUT_OF_MEMORY",
            -4: "NLOPT_ROUNDOFF_LIMITED",
            -5: "NLOPT_FORCED_STOP",
        }
        message_nlopt = messages[status_nlopt]
        info = {
            "n_eval": niter,
            "niter": niter,
            "message": message_nlopt,
            "original": status,
            "evaluator": evaluator,
            "status": status,
        }
        if message is None:
            message = message_nlopt

        valid = valid and converged
        if evaluator is not None:
            valid = valid and not evaluator.maxiter_reached

        approx = {}
        if inv_hessian is None and hessian is None and evaluator is not None:
            hessian = evaluator.last_hessian

        if inv_hessian is not None:
            info["inv_hesse"] = inv_hessian
            approx["inv_hessian"] = inv_hessian

        return cls(
            params=param_dict,
            edm=edm,
            fminopt=fminopt,
            status=status,
            converged=converged,
            info=info,
            niter=niter,
            valid=valid,
            loss=loss,
            minimizer=minimizer,
            criterion=criterion,
            message=message,
            evaluator=evaluator,
        )



    @property
    def values(self) -> Mapping[str | ZfitParameter, float]:
        return self._values

    @property
    def criterion(self) -> ConvergenceCriterion:
        return self._criterion


    @property
    def edm(self) -> float:
        pass



    @property
    def fminopt(self) -> float:
        pass

    @property
    def fmin(self) -> float:
        pass

    @property
    @deprecated(
        None,
        "Use `fmin` instead which now returns the full minimum value. This will be removed in the future.",
    )
    def fminfull(self) -> float:
        pass


    @property
    def info(self) -> Mapping[str, object]:
        return self._info

    @property
    def converged(self) -> bool:
        return bool(self._converged)




    @contextlib.contextmanager
    def _input_check_reset_params(self, params):
        params = self._input_check_params(params=params)
        if self._is_frozen:
            yield params
            return
        old_values = np.asarray(params)
        try:
            yield params
        except Exception:
            warnings.warn(
                "Exception occurred, parameter values are not reset and in an arbitrary, last"
                " used state. If this happens during normal operation, make sure you reset the values.",
                RuntimeWarning,
                stacklevel=3,
            )
            raise
        set_values(params=params, values=old_values, allow_partial=True)  # TODO: or set?

    def _input_check_params(self, params):
        return convert_to_container(params) if params is not None else list(self.params.keys())

    @deprecated_args(None, "Use `name` instead", "error_name")
    def hesse(
        self,
        params: ParamsTypeOpt = None,
        method: str | Callable | None = None,
        *,
        cl: float | None = None,
        name: str | bool | None = None,
        weightcorr: WeightCorr | None = None,
        error_name: str | None = None,
    ) -> dict[ZfitIndependentParameter, dict]:
        r"""Calculate for `params` the symmetric error using the Hessian/covariance matrix.

        This method estimates the covariance matrix using the inverse of the Hessian matrix. The assumption is
        that the loss profile - usually a likelihood or a :math:`\chi^2` - is hyperbolic. This is usually the case for
        fits with many observations, i.e. it is exact in the asymptotic limit. If the loss profile is not hyperbolic,
        another method, "zfit_error" or "minuit_minos" should be used.

        **Weights**
        Weighted likelihoods are a special class of likelihoods as they are not an actual likelihood. The
        minimum is still valid, however the profile is not a proper likelihood. Therefore, corrections
        will be automatically applied to the Hessian uncertainty estimation in order to take the effects
        of the weights into account. Various corrections are available, some faster, others more correct.


        Args:
            params: The parameters to calculate the
                Hessian symmetric error. If None, use all parameters.
            method: the method to calculate the covariance matrix. Can be
                {'minuit_hesse', 'hesse_np', 'approx'} or a callable.
            cl: Confidence level for the error. If None, use the default value of 0.683.
            name: The name for the error in the dictionary. This will be added to
                the information collected in params under ``params[p][name]`` where
                p is a Parameter. If the name is `False`, it won't be added and only
                returned. Defaults to `'hesse'`.
            weightcorr: |@doc:result.hesse.weightcorr.method| Method to correct the estimation of the covariance matrix/hesse error
                   for a weighted likelihood. The following methods are available, for a comparison and
                   the derivation of the methods, see [langenbruch1]_:

                    - `False`: no correction, the covariance matrix is calculated as if the likelihood
                      was unweighted. This will generally underestimate the errors.
                    - `asymptotic`: the covariance matrix is corrected by the asymptotic formula
                      for the weighted likelihood. This is the default, yet computationally most
                      expensive method.
                    - `sumw2`: the covariance matrix is corrected by the effective sample size.
                      This is the fastest method but won't yield asymptotically correct results.

                   This is not (yet) guaranteed to fully work for binned fits and maybe under/over
                   represents errors.

                    .. [langenbruch1] Langenbruch, C. Parameter uncertainties in weighted unbinned maximum
                       likelihood fits
                       `Eur. Phys. J. C 82, 393 (2022). <https://doi.org/10.1140/epjc/s10052-022-10254-8>`_. |@docend:result.hesse.weightcorr.method|

        Returns:
            Result of the hessian (symmetric) error as dict with each parameter holding
                the error dict {'error': sym_error}.

                So given param_a (from zfit.Parameter(.))
                `error_a = result.hesse(params=param_a)[param_a]['error']`
                error_a is the hessian error.
        """
        cl = 0.68268949 if cl is None else cl  # scipy.stats.chi2(1).cdf(1)
        if cl >= 1:
            msg = f"cl is the confidence limit and has to be < 1, not {cl}"
            raise ValueError(msg)

        if method is None:
            method = self._default_hesse
            from zfit.minimizers.minimizer_minuit import Minuit  # noqa: PLC0415

            if isinstance(self.minimizer, Minuit):
                method = "minuit_hesse"
        if error_name is not None:
            name = error_name

        name_warning_triggered = False
        if name is None:
            if not isinstance(method, str):
                msg = "Need to specify `name` or use a string as `method`"
                raise ValueError(msg)
            name = "hesse"

        if self._is_frozen:
            if weightcorr is None:
                weightcorr = WeightCorr.FALSE  # Default for frozen results
            else:
                weightcorr = WeightCorr(weightcorr)
                if weightcorr != WeightCorr.FALSE:
                    msg = (
                        "Cannot compute new hesse/covariance with weight correction on a frozen result. "
                        "The result is frozen and no new calculations can be performed. "
                        "Compute hesse/covariance before freezing the result, or use weightcorr=False to retrieve cached values."
                    )
                    raise RuntimeError(msg)
        elif weightcorr is None:
            weightcorr = WeightCorr.ASYMPTOTIC if self.loss.is_weighted else WeightCorr.FALSE
        else:
            weightcorr = WeightCorr(weightcorr)
        if weightcorr != WeightCorr.FALSE and not self._is_frozen and not self.loss.is_weighted:
            msg = "Weight correction is only available for weighted likelihoods, weightcorr cannot be given."
            raise ValueError(msg)

        with self._input_check_reset_params(params) as checkedparams:
            uncached_params = self._check_get_uncached_params(
                params=checkedparams, method_name=name, cl=cl, weightcorr=weightcorr
            )
            if uncached_params:
                error_dict = self._hesse(params=uncached_params, method=method, cl=cl, weightcorr=weightcorr)
                if any(val["error"] is None for val in error_dict.values()):
                    return {}
                for p in error_dict:
                    error_dict[p]["cl"] = round(cl, 3)  # Round to 3 digits to avoid numerical issues
                    error_dict[p]["weightcorr"] = weightcorr
                if name:
                    self._cache_errors(name=name, errors=error_dict)
            else:
                error_dict = {}

        error_dict.update({p: self.params[p][name] for p in checkedparams if p not in uncached_params})
        if name_warning_triggered:
            error_dict.update({p: self.params[p][method] for p in checkedparams if p not in uncached_params})
        return {p: error_dict[p] for p in checkedparams}

    def _cache_errors(self, name, errors):
        for param, error in errors.items():
            self.params[param][name] = error

    def _hesse(self, params, method, cl, weightcorr):
        pseudo_sigma = scipy.stats.chi2(1).ppf(cl) ** 0.5

        covariance_dict = self.covariance(params, method, as_dict=True, weightcorr=weightcorr)
        return {
            p: {
                "error": (
                    float(covariance_dict[(p, p)]) ** 0.5 * pseudo_sigma
                    if covariance_dict[(p, p)] is not None
                    else None
                )
            }
            for p in params
        }

    def error(
        self,
        params: ParamsTypeOpt = None,  # noqa: ARG002
        method: str | Callable | None = None,  # noqa: ARG002
        error_name: str | None = None,  # noqa: ARG002
        sigma: float = 1.0,  # noqa: ARG002
    ) -> dict:
        pass

    @deprecated_args(None, "Use name instead.", "error_name")
    def errors(
        self,
        params: ParamsTypeOpt = None,
        method: str | Callable | None = None,
        name: str | None = None,
        cl: float | None = None,
        *,
        sigma=None,
        error_name: str | None = None,
    ) -> tuple[dict, None | FitResult]:
        pass


    def covariance(
        self,
        params: ParamsTypeOpt = None,
        method: str | Callable | None = None,
        as_dict: bool = False,
        *,
        weightcorr: WeightCorr = None,
    ):
        """Calculate the covariance matrix for `params`.

        Args:
            params: The parameters to calculate
                the covariance matrix. If `params` is `None`, use all *floating* parameters.
            method: The method to use to calculate the covariance matrix. Valid choices are
                {'minuit_hesse', 'hesse_np'} or a Callable.
            as_dict: Default `False`. If `True` then returns a dictionnary.
            weightcorr: |@doc:result.hesse.weightcorr.method| Method to correct the estimation of the covariance matrix/hesse error
                   for a weighted likelihood. The following methods are available, for a comparison and
                   the derivation of the methods, see [langenbruch1]_:

                    - `False`: no correction, the covariance matrix is calculated as if the likelihood
                      was unweighted. This will generally underestimate the errors.
                    - `asymptotic`: the covariance matrix is corrected by the asymptotic formula
                      for the weighted likelihood. This is the default, yet computationally most
                      expensive method.
                    - `sumw2`: the covariance matrix is corrected by the effective sample size.
                      This is the fastest method but won't yield asymptotically correct results.

                   This is not (yet) guaranteed to fully work for binned fits and maybe under/over
                   represents errors.

                    .. [langenbruch1] Langenbruch, C. Parameter uncertainties in weighted unbinned maximum
                       likelihood fits
                       `Eur. Phys. J. C 82, 393 (2022). <https://doi.org/10.1140/epjc/s10052-022-10254-8>`_. |@docend:result.hesse.weightcorr.method|

        Returns:
            2D `numpy.array` of shape (N, N);
            `dict`(param1, param2) -> covariance if `as_dict == True`.
        """
        if method is None:
            method = self._default_hesse
        if weightcorr is None:
            weightcorr = WeightCorr.ASYMPTOTIC
        weightcorr = WeightCorr(weightcorr)

        cache_key = (method, weightcorr)
        if cache_key not in self._covariance_dict:
            with self._input_check_reset_params(params) as checkedparams:
                self._covariance_dict[cache_key] = self._covariance(method=method, weightcorr=weightcorr)

        else:
            checkedparams = self._input_check_params(params)
        covariance = {
            k: self._covariance_dict[cache_key].get(k) for k in itertools.product(checkedparams, checkedparams)
        }

        if as_dict:
            return covariance
        return dict_to_matrix(checkedparams, covariance)

    def _covariance(self, method, *, weightcorr: WeightCorr = None):
        if self._is_frozen:
            msg = (
                "Cannot compute covariance on a frozen result. "
                "The covariance must be computed before freezing the result. "
                "Call `result.covariance()` before `result.freeze()`."
            )
            raise RuntimeError(msg)
        if not callable(method):
            if method not in self._hesse_methods:
                msg = f"The following method is not a valid, implemented method: {method}. Use one of {self._hesse_methods.keys()}"
                raise KeyError(msg)
            method = self._hesse_methods[method]
        params = list(self.params.keys())

        if (weightcorr != weightcorr.FALSE) and any(
            isinstance(data, ZfitUnbinnedData) and data.has_weights for data in self.loss.data
        ):
            return covariance_with_weights(hinv=method, result=self, params=params, weightcorr=weightcorr)

        return method(result=self, params=params)

    def correlation(
        self,
        params: ParamsTypeOpt = None,
        method: str | Callable | None = None,
        as_dict: bool = False,
    ):
        pass

    def freeze(self):
        pass

    def __str__(self):
        string = Style.BRIGHT + "FitResult" + Style.NORMAL + f" of\n{self.loss} \nwith\n{self.minimizer}\n\n"
        string += tabulate(
            [
                [
                    color_on_bool(self.valid),
                    color_on_bool(self.converged, on_true=False),
                    color_on_bool(self.params_at_limit, on_true=colored.bg(9), on_false=False),
                    format_value(self.edm, highprec=False),
                    f"          {self._fmin:.2f} | {format_value(self.fminopt)}",
                ]
            ],
            [
                "valid",
                "converged",
                "param at limit",
                "edm",
                "approx. fmin (full | opt.)",
            ],
            tablefmt="fancy_grid",
            disable_numparse=True,
            colalign=["center", "center", "center", "center", "right"],
        )
        string += "\n\n" + Style.BRIGHT + "Parameters\n" + Style.NORMAL
        string += str(self.params)
        return string


    def __enter__(self):
        self._tmp_old_param_values.append(znp.asarray(tuple(self.params.keys())))
        self.update_params()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        old_vals = self._tmp_old_param_values.pop()
        set_values(tuple(self.params.keys()), old_vals)

    def update_params(self) -> typing.Self:
        pass








class ListWithKeys(collections.UserList):
    __slots__ = ("_initdict",)

    def __init__(self, initdict) -> None:
        super().__init__(initlist=initdict.values())
        self._initdict = initdict

    def __getitem__(self, item):
        if isinstance(item, ZfitParameter):
            return self._initdict[item]
        return super().__getitem__(item)

    def keys(self):
        return self._initdict.keys()

    def values(self):
        return self._initdict.values()

    def items(self):
        return self._initdict.items()


class ValuesHolder(NameToParamGetitem, ListWithKeys):
    __slots__ = ()


class ParamHolder(NameToParamGetitem, collections.UserDict):
    def __str__(self) -> str:
        order_keys = ["value", "hesse"]
        keys = OrderedSet()
        for pdict in self.values():
            keys.update(OrderedSet(pdict))
        order_keys = OrderedSet([key for key in order_keys if key in keys])
        order_keys.update(keys)

        rows = []
        for param, pdict in self.items():
            name = param.name if isinstance(param, ZfitParameter) else param
            row = [name]
            row.extend(format_value(pdict.get(key, " ")) for key in order_keys)
            if isinstance(param, ZfitParameter):
                row.append(
                    color_on_bool(
                        bool(param.at_limit),
                        on_true=colored.bg("light_red"),
                        on_false=False,
                    )
                )
            rows.append(row)

        order_keys = ["name", *list(order_keys), "at limit"]
        order_keys[order_keys.index("value")] = "value  (rounded)"
        return tabulate(rows, order_keys, numalign="right", stralign="right", colalign=("left",))
