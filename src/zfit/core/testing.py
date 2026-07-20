

from __future__ import annotations

import typing
from collections.abc import Callable, Iterable

import scipy.stats

from ..util.container import convert_to_container
from zfit._interfaces import ZfitPDF

if typing.TYPE_CHECKING:
    import zfit  # noqa: F401

__all__ = ["tester"]

import scipy.integrate




class AutoTester:
    def __init__(self):
        self.pdfs = []

    def register_pdf(
            self,
            pdf_class: ZfitPDF,
            params_factories: Callable | Iterable[Callable],
            scipy_dist: scipy.stats.rv_continuous = None,
            analytic_int_axes: None | int | list[tuple[int, ...]] = None,
    ):
        params_factories = convert_to_container(params_factories)

        if isinstance(analytic_int_axes, tuple):
            raise TypeError(
                "`analytic_int_axes` is either a number or a list of tuples."
            )
        analytic_int_axes = convert_to_container(
            analytic_int_axes, non_containers=[tuple]
        )
        if analytic_int_axes is not None:
            isinstance(analytic_int_axes)
        registration = {
            "pdf_class": pdf_class,
            "params": params_factories,
            "scipy_dist": scipy_dist,

        }
        self.pdfs.append(registration)



tester = AutoTester()
