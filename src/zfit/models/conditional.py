
from __future__ import annotations

import sys
import typing

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from ordered_set import OrderedSet

from ..util.ztyping import ExtendedInputType, NormInputType

if typing.TYPE_CHECKING:
    pass

import functools
import typing
import warnings
from collections.abc import Mapping

import tensorflow as tf

import zfit.z.numpy as znp
from zfit._interfaces import (
    ZfitIndependentParameter,
    ZfitParameter,
    ZfitPDF,
    ZfitSpace,
)

from .. import z
from ..core.parameter import set_values
from ..core.space import combine_spaces, convert_to_space, supports
from ..util.exception import WorkInProgressError
from ..util.warnings import warn_experimental_feature
from .functor import BaseFunctor

if typing.TYPE_CHECKING:
    import zfit  # noqa: F401


class ConditionalPDFV1(BaseFunctor):
    @warn_experimental_feature
    def __init__(
        self,
        pdf: ZfitPDF,
        cond: Mapping[ZfitIndependentParameter, ZfitSpace],
        *,
        name: str = "ConditionalPDF",
        extended: ExtendedInputType = None,
        norm: NormInputType = None,
        use_vectorized_map: bool = False,
        sample_with_replacement: bool = True,
        label: str | None = None,
    ) -> None:
        """EXPERIMENTAL! Implementation of a Conditional PDF, rather slow and for research purpose.

        As an example, a Gaussian is wrapped in order to make 'sigma' conditional.

        .. jupyter-execute::



        Args:
            pdf: PDF that will be wrapped. Convert one or several parameters of *pdf* to a conditional
                parameter, meaning that the parameter *param* in the ``cond`` mapping will now be
                determined by the data in the ``Space``, the value of the ``cond``.
            cond: Mapping of parameter to input data.
            extended: |@doc:pdf.init.extended| The overall yield of the PDF.
               If this is parameter-like, it will be used as the yield,
               the expected number of events, and the PDF will be extended.
               An extended PDF has additional functionality, such as the
               ``ext_*`` methods and the ``counts`` (for binned PDFs). |@docend:pdf.init.extended|
            norm: |@doc:pdf.init.norm| Normalization of the PDF.
               By default, this is the same as the default space of the PDF. |@docend:pdf.init.norm|
            name: |@doc:pdf.init.name| Name of the PDF.
               Maybe has implications on the serialization and deserialization of the PDF.
               For a human-readable name, use the label. |@docend:pdf.init.name|
            label: |@doc:pdf.init.label| Human-readable name
               or label of
               the PDF for a better description, to be used with plots etc.
               Has no programmatical functional purpose as identification. |@docend:pdf.init.label|
            use_vectorized_map:
            sample_with_replacement:
        """
        self._sample_with_replacement = sample_with_replacement
        self._use_vectorized_map = use_vectorized_map
        self._cond, cond_obs = self._check_input_cond(cond)
        obs = pdf.space * cond_obs
        super().__init__(pdfs=pdf, obs=obs, name=name, extended=extended, norm=norm, label=label)

    @property
    def cond(self) -> dict[ZfitIndependentParameter, ZfitSpace]:
        return self._cond

    def _check_input_cond(self, cond):
        spaces = []
        for param, obs in cond.items():
            if not isinstance(param, ZfitIndependentParameter):
                msg = f"parameter {param} not a ZfitIndependentParameter"
                raise TypeError(msg)
            spaces.append(convert_to_space(obs))
        return cond, combine_spaces(*spaces)

    @supports(norm=True, multiple_limits=True)
    @z.function(wraps="conditional_pdf")
    def _pdf(self, x, norm):
        pdf = self.pdfs[0]
        param_x_indices = {p: x.obs.index(p_space.obs[0]) for p, p_space in self._cond.items()}
        x_values = x.value()

        from zfit import run  # noqa: PLC0415

        if self._use_vectorized_map and run.get_graph_mode() is not False:
            tf_map = tf.vectorized_map
        else:
            output_signature = tf.TensorSpec(shape=(1, *x_values.shape[1:-1]), dtype=self.dtype)
            tf_map = functools.partial(tf.map_fn, fn_output_signature=output_signature)



        params = tuple(param_x_indices.keys())
        with set_values(params, params):
            probs = tf_map(eval_pdf, x_values)
        return probs[:, 0]  # removing stack dimension, implicitly in map_fn

    def _get_params(
        self,
        floating: bool | None = True,
        is_yield: bool | None = None,
        extract_independent: bool | None = True,
        *,
        autograd: bool | None = None,
    ) -> set[ZfitParameter]:
        params = super()._get_params(floating, is_yield, extract_independent, autograd=autograd)
        params -= OrderedSet(self._cond)
        return params

    @z.function(wraps="conditional_pdf")
    def _single_hook_integrate(self, limits, norm, x, options):
        from zfit import run  # noqa: PLC0415

        if not run.get_graph_mode():
            warnings.warn(
                "Using the Conditional PDF in eager mode (no jit) maybe gets stuck.",
                RuntimeWarning,
                stacklevel=2,
            )

        param_x_indices = {p: x.obs.index(p_space.obs[0]) for p, p_space in self._cond.items()}
        x_values = x.value()
        pdf = self.pdfs[0]

        if self._use_vectorized_map and run.get_graph_mode() is not False:
            tf_map = tf.vectorized_map
        else:
            output_signature = tf.TensorSpec(shape=(1, *x_values.shape[1:-1]), dtype=self.dtype)
            tf_map = functools.partial(tf.map_fn, fn_output_signature=output_signature)


        integrals = tf_map(eval_int, x_values)
        return integrals[:, 0]  # removing stack dimension, implicitly in map_fn

    @z.function(wraps="conditional_pdf")
    def _single_hook_sample(self, n, limits, x):
        tf.assert_equal(
            n,
            x.num_entries,
            message="Different number of n requested than x given for conditional sampling. Needs to agree",
        )

        param_x_indices = {p: x.obs.index(p_space.obs[0]) for p, p_space in self._cond.items()}
        x_values = x.value()
        pdf = self.pdfs[0]

        from zfit import run  # todo: we could use the normal python map for eager?  # noqa: PLC0415

        if self._use_vectorized_map and run.get_graph_mode() is not False:
            tf_map = tf.vectorized_map
        else:
            output_signature = tf.TensorSpec(shape=(1, pdf.n_obs), dtype=self.dtype)
            tf_map = functools.partial(tf.map_fn, fn_output_signature=output_signature)


        sample_rnd = tf_map(eval_sample, x_values)[..., 0]
        return znp.concatenate([sample_rnd, x_values], axis=-1)

    def copy(self, **override_parameters) -> Self:  # noqa: ARG002
        msg = "Currently copying not possible. Use `set_yield` to set a yield inplace."
        raise WorkInProgressError(msg)


