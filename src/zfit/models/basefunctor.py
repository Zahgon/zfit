
from __future__ import annotations

import typing
from collections.abc import Iterable

import pydantic.v1 as pydantic
import tensorflow as tf

from zfit._interfaces import ZfitFunctorMixin, ZfitModel, ZfitParameter, ZfitSpace

from .. import z
from ..core.coordinates import convert_to_obs_str
from ..core.dimension import get_same_obs
from ..core.parameter import convert_to_parameter
from ..core.space import Space, combine_spaces
from ..serialization import SpaceRepr
from ..serialization.pdfrepr import BasePDFRepr
from ..serialization.serializer import Serializer
from ..settings import ztypes
from ..util import ztyping
from ..util.container import convert_to_container
from ..util.deprecation import deprecated_norm_range
from ..util.exception import (
    LimitsIncompatibleError,
    ModelIncompatibleError,
    NormRangeNotSpecifiedError,
    ObsIncompatibleError,
)
from ..util.warnings import warn_advanced_feature, warn_changed_feature
from ..z import numpy as znp

if typing.TYPE_CHECKING:
    import zfit  # noqa: F401


def extract_daughter_input_obs(obs: ztyping.ObsTypeInput, spaces: Iterable[ZfitSpace]) -> ZfitSpace:
    """Extract the common space from `spaces` by combining them, test against obs.

    The `obs` are assumed to be the obs given to a functor while the `spaces` are the spaces of the daughters.
    First, the combined space from the daughters is extracted. If no `obs` are given, this is returned.
    If `obs` are given, it is checked whether they agree. If they agree, and no limit is set on `obs` (i.e. they
    are pure strings), the inferred limits are used, sorted by obs. Otherwise, obs is directly used.

    Args:
        obs:
        spaces:

    Returns:
    """
    spaces = convert_to_container(spaces)
    try:
        models_space = combine_spaces(*spaces)
    except LimitsIncompatibleError:  # then only add obs
        extracted_obs = _extract_common_obs(obs=tuple(space.obs for space in spaces))
        models_space = Space(obs=extracted_obs)

    if obs is None:
        obs = models_space
    else:
        obs = obs if isinstance(obs, Space) else Space(obs=obs)
        if obs.obs != models_space.obs and not obs.limits_are_set:
            obs = models_space.with_obs(obs.obs)

    return obs


class FunctorMixin(ZfitFunctorMixin):
    def __init__(self, models, obs, **kwargs):
        models = convert_to_container(models, container=list)
        obs = extract_daughter_input_obs(obs=obs, spaces=[model.space for model in models])

        self._model_obs = tuple(model.obs for model in models)
        self._models = models
        super().__init__(obs=obs, **kwargs)

    def _get_params(
        self,
        floating: bool | None = True,
        is_yield: bool | None = None,
        extract_independent: bool | None = True,
        *,
        autograd: bool | None = None,
    ) -> set[ZfitParameter]:
        params = super()._get_params(floating, is_yield, extract_independent, autograd=autograd)
        if is_yield is not True:
            params = params.union(
                *(
                    model.get_params(
                        floating=floating, is_yield=False, extract_independent=extract_independent, autograd=autograd
                    )
                    for model in self.models
                )
            )
        return params

    @property
    def models(self) -> list[ZfitModel]:
        pass





class FunctorPDFRepr(BasePDFRepr):
    _implementation = None
    pdfs: list[Serializer.types.PDFTypeDiscriminated]
    obs: SpaceRepr | None = None



def _extract_common_obs(obs: tuple[tuple[str, ...] | Space, ...]) -> tuple[str, ...]:
    obs_iter = [space.obs if isinstance(space, Space) else space for space in obs]
    unique_obs = []
    for currobs in obs_iter:
        for o in currobs:
            if o not in unique_obs:
                unique_obs.append(o)
    return tuple(unique_obs)


def _preprocess_init_sum(fracs, obs, pdfs):
    frac_param_created = False
    if len(pdfs) < 2:
        msg = f"Cannot build a sum of less than two pdfs {pdfs}"
        raise ValueError(msg)
    common_obs = obs if obs is not None else pdfs[0].obs
    common_obs = convert_to_obs_str(common_obs)
    if any(frozenset(pdf.obs) != frozenset(common_obs) for pdf in pdfs):
        msg = "Currently, sums are only supported in the same observables"
        raise ObsIncompatibleError(msg)
    are_extended = [pdf.is_extended for pdf in pdfs]
    all_extended = all(are_extended)
    no_extended = not any(are_extended)
    fracs = convert_to_container(fracs)
    if fracs:  # not None or empty list
        fracs = [convert_to_parameter(frac) for frac in fracs]
    elif not all_extended:
        msg = f"Not all pdf {pdfs} are extended and no fracs {fracs} are provided."
        raise ModelIncompatibleError(msg)
    if not no_extended and fracs:
        warn_advanced_feature(
            f"This SumPDF is built with fracs {fracs} and {'all' if all_extended else 'some'} "
            f"pdf are extended: {pdfs}."
            f" This will ignore the yields of the already extended pdfs and the result will"
            f" be a not extended SumPDF.",
            identifier="sum_extended_frac",
        )
    if fracs:
        if len(fracs) == len(pdfs) - 1:
            frac_param_created = True
            frac_params_tmp = {f"frac_{i}": frac for i, frac in enumerate(fracs)}


            remaining_frac = convert_to_parameter(remaining_frac_func, params=frac_params_tmp)
            z.assert_non_negative(
                remaining_frac,
                f"The remaining fraction is negative, the sum of fracs is > 0. Fracs: {fracs}",
            )  # check fractions

            fracs_cleaned = [*fracs, remaining_frac]

        elif len(fracs) == len(pdfs):
            if Serializer._existing_params is None:  # todo: make a better context for serialization/deserialization
                warn_changed_feature(
                    "A SumPDF with the number of fractions equal to the number of pdf will no longer "
                    "be extended. To make it extended, either manually use 'create_exteneded' or set "
                    "the yield. OR provide all pdfs as extended pdfs and do not provide a fracs "
                    "argument.",
                    identifier="new_sum",
                )
            fracs_cleaned = fracs

        else:
            msg = (
                f"If all PDFs are not extended {pdfs}, the fracs {fracs} have to be of"
                f" the same length as pdf or one less."
            )
            raise ModelIncompatibleError(msg)
        param_fracs = fracs_cleaned
    sum_yields = None
    if all_extended and not fracs:
        yields = [pdf.get_yield() for pdf in pdfs]


        sum_yields = convert_to_parameter(sum_yields_func, params={f"yield_{i}": y for i, y in enumerate(yields)})
        yield_fracs = [
            convert_to_parameter(
                lambda params: params["yield_"] / params["sum_yields"],
                params={"sum_yields": sum_yields, "yield_": yield_},
            )
            for yield_ in yields
        ]

        fracs_cleaned = None
        param_fracs = yield_fracs
    params = {}
    for i, frac in enumerate(param_fracs):
        params[f"frac_{i}"] = frac
    return (
        all_extended,
        fracs_cleaned,
        param_fracs,
        params,
        sum_yields,
        frac_param_created,
    )
