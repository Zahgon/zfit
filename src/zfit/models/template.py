from __future__ import annotations

import typing

import numpy as np
import tensorflow as tf

from ..core.binnedpdf import BaseBinnedPDF
from ..core.space import supports
from ..util.exception import SpecificFunctionNotImplemented
from ..z import numpy as znp

if typing.TYPE_CHECKING:
    import zfit  # noqa: F401


class BinnedTemplatePDFV1(BaseBinnedPDF):
    def __init__(
        self, data, sysshape=None, extended=None, norm=None, name="BinnedTemplatePDF", label: str | None = None
    ):
        obs = data.space
        if extended is None:
            extended = True
        if sysshape is None:
            sysshape = {}
        if sysshape is True:
            import zfit  # noqa: PLC0415

            sysshape = {
                f"sysshape_{i}": zfit.Parameter(f"auto_sysshape_{self}_{i}", 1.0)
                for i in range(data.values().shape.num_elements())
            }
        params = {}
        params.update(sysshape)
        self._template_sysshape = sysshape
        if extended is True:
            self._automatically_extended = True
            if sysshape:
                import zfit  # noqa: PLC0415


                from zfit.core.parameter import get_auto_number  # noqa: PLC0415

                extended = zfit.ComposedParameter(f"TODO_name_selfmade_{get_auto_number()}", sumfunc, params=sysshape)

            else:
                extended = znp.sum(data.values())
        elif extended is not False:
            self._automatically_extended = False
        super().__init__(obs=obs, name=name, params=params, extended=extended, norm=norm, label=label)

        self._data = data

    def _ext_pdf(self, x: tf.Tensor, norm: tf.Tensor | None) -> tf.Tensor:
        counts = self._counts(x, norm)
        areas = np.prod(self._data.axes.widths, axis=0)
        return counts / areas

    @supports(norm=False)
    def _pdf(self, x: tf.Tensor, norm: tf.Tensor | None) -> tf.Tensor:
        counts = self._rel_counts(x, norm)
        areas = np.prod(self._data.axes.widths, axis=0)
        return counts / areas

    @supports(norm="norm")
    def _counts(self, x: tf.Tensor, norm: tf.Tensor | None = None) -> tf.Tensor:  # noqa: ARG002
        if not self._automatically_extended:
            raise SpecificFunctionNotImplemented
        values = self._data.values()
        if sysshape := list(self._template_sysshape.values()):
            sysshape_flat = tf.stack(sysshape)
            sysshape = znp.reshape(sysshape_flat, values.shape)
            values = values * sysshape
        return values

    @supports(norm="norm")
    def _rel_counts(self, x: tf.Tensor, norm: tf.Tensor | None = None) -> tf.Tensor:  # noqa: ARG002
        values = self._data.values()
        if sysshape := list(self._template_sysshape.values()):
            sysshape_flat = tf.stack(sysshape)
            sysshape = znp.reshape(sysshape_flat, values.shape)
            values = values * sysshape
        return values / znp.sum(values)



