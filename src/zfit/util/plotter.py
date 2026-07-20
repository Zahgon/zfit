from __future__ import annotations

import typing
from collections.abc import Callable, Mapping

import numpy as np

from zfit._interfaces import ZfitBinnedData, ZfitData, ZfitPDF, ZfitUnbinnedData

from ..core.space import convert_to_space
from . import ztyping
from .checks import RuntimeDependency
from .warnings import warn_experimental_feature

if typing.TYPE_CHECKING:
    import zfit  # noqa: F401

try:
    import matplotlib.pyplot as plt
except ImportError as error:
    plt = RuntimeDependency("plt", error_msg=str(error))

try:
    import mplhep
except ImportError as error:
    mplhep = RuntimeDependency("mplhep", error_msg=str(error))


def plot_sumpdf_components_pdfV1(
    model,
    *,
    plotfunc: Callable | None = None,
    scale=1,
    ax=None,
    linestyle=None,
    plotkwargs: Mapping[str, object] | None = None,
    extended: bool | None = None,
):
    pass


def plot_model_pdf(
    model: ZfitPDF,
    *,
    plotfunc: Callable | None = None,
    extended: bool | None = None,
    obs: ztyping.ObsTypeInput = None,
    scale: float | int | None = None,
    ax: plt.Axes | None = None,
    num: int | None = None,
    full: bool | None = None,
    linestyle=None,
    plotkwargs=None,
):
    pass




class ZfitPDFPlotter:
    @warn_experimental_feature
    @assert_initialized
    def plotpdf(
        self,
        data: ZfitData | None = None,
        *,
        depth: int | None = None,
        density: bool | None = None,
        plotfunc: Callable | None = None,
        extended: bool | None = None,
        obs: ztyping.ObsTypeInput = None,
        scale: float | int | None = None,
        ax: plt.Axes | None = None,
        num: int | None = None,
        full: bool | None = None,
        linestyle=None,
        plotkwargs: Mapping[str, object] | None = None,
        histplotkwargs: Mapping[str, object] | None = None,
    ):
        pass

    def _plotpdf(self, **kwargs):
        raise NotImplementedError


    def __call__(self, data=None, **kwargs):
        return self.plotpdf(data=data, **kwargs)

    def _plot_scale_data(
        self, data: ZfitData, density=None, normalize=None, ax=None, histplotkwargs=None
    ) -> (plt.Axes, float):
        pass



class PDFPlotter(ZfitPDFPlotter):
    def __init__(
        self,
        pdf: ZfitPDF | None,
        pdfplotter: Callable | None = None,
        componentplotter: ZfitPDFPlotter = None,
        defaults: Mapping[str, object] | None = None,
    ):
        self.defaults = {} if defaults is None else defaults
        self.pdf = pdf
        if pdfplotter is not None and not callable(pdfplotter):
            msg = f"pdfplotter must be a callable, is {type(pdfplotter)}."
            raise TypeError(msg)
        self._pdfplotter = plot_model_pdf if pdfplotter is None else pdfplotter
        if componentplotter is not None and not isinstance(componentplotter, ZfitPDFPlotter):
            msg = f"componentplotter must be a ZfitPDFPlotter, is {type(componentplotter)}."
            raise TypeError(msg)
        self._componentplotter = componentplotter




class SumCompPlotter(ZfitPDFPlotter):
    def __init__(
        self,
        pdf: ZfitPDF | None,
        *args,
        **kwargs,
    ):
        if pdf is not None and not isinstance(pdf, ZfitPDF):
            msg = f"pdf must be a ZfitPDF, is {type(pdf)}."
            raise TypeError(msg)
        self.pdf = pdf
        super().__init__(*args, **kwargs)

