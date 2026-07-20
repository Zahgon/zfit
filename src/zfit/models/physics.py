
from __future__ import annotations

import typing
from typing import Literal

import numpy as np
import pydantic.v1 as pydantic
import tensorflow as tf

import zfit.z.numpy as znp
from zfit import z

from ..core.basepdf import BasePDF
from ..core.serialmixin import SerializableMixin
from ..core.space import ANY_LOWER, ANY_UPPER, Space, supports
from ..serialization import Serializer, SpaceRepr
from ..serialization.pdfrepr import BasePDFRepr
from ..util import ztyping
from ..util.ztyping import ExtendedInputType, NormInputType

if typing.TYPE_CHECKING:
    import zfit  # noqa: F401


def _powerlaw(x, a, k):
    return a * znp.power(x, k)


@z.function(wraps="tensor", keepalive=True)
def crystalball_func(x, mu, sigma, alpha, n):
    t = (x - mu) / sigma * tf.sign(alpha)
    abs_alpha = znp.abs(alpha)
    a = znp.power((n / abs_alpha), n) * znp.exp(-0.5 * znp.square(alpha))
    b = (n / abs_alpha) - abs_alpha
    cond = tf.less(t, -abs_alpha)
    func = z.safe_where(
        cond,
        lambda t: _powerlaw(b - t, a, -n),
        lambda t: znp.exp(-0.5 * znp.square(t)),
        values=t,
        value_safer=lambda t: znp.ones_like(t) * (b - 2),
    )
    return znp.maximum(func, znp.zeros_like(func))


@z.function(wraps="tensor", keepalive=True, stateless_args=False)
def double_crystalball_func(x, mu, sigma, alphal, nl, alphar, nr):
    cond = tf.less(x, mu)

    return tf.where(
        cond,
        crystalball_func(x, mu, sigma, alphal, nl),
        crystalball_func(x, mu, sigma, -alphar, nr),
    )


@z.function(wraps="tensor", keepalive=True, stateless_args=False)
def generalized_crystalball_func(x, mu, sigmal, alphal, nl, sigmar, alphar, nr):
    cond = tf.less(x, mu)

    return tf.where(
        cond,
        crystalball_func(x, mu, sigmal, alphal, nl),
        crystalball_func(x, mu, sigmar, -alphar, nr),
    )














class CrystalBall(BasePDF, SerializableMixin):
    _N_OBS = 1

    def __init__(
        self,
        mu: ztyping.ParamTypeInput,
        sigma: ztyping.ParamTypeInput,
        alpha: ztyping.ParamTypeInput,
        n: ztyping.ParamTypeInput,
        obs: ztyping.ObsTypeInput,
        *,
        extended: ExtendedInputType = None,
        norm: NormInputType = None,
        name: str | None = None,
        label: str | None = None,
    ):
        r"""Crystal Ball shaped PDF. A combination of a Gaussian with a powerlaw tail.

        The function is defined as follows:

        .. math::
            f(x;\mu, \sigma, \alpha, n) =  \begin{cases} \exp(- \frac{(x - \mu)^2}{2 \sigma^2}),
            & \mbox{for }\frac{x - \mu}{\sigma} \geqslant -\alpha \newline
            A \cdot (B - \frac{x - \mu}{\sigma})^{-n}, & \mbox{for }\frac{x - \mu}{\sigma}
             < -\alpha \end{cases}

        with

        .. math::
            A = \left(\frac{n}{\left| \alpha \right|}\right)^n \cdot
            \exp\left(- \frac {\left|\alpha \right|^2}{2}\right)

            B = \frac{n}{\left| \alpha \right|}  - \left| \alpha \right|

        Args:
            mu: The mean of the gaussian
            sigma: Standard deviation of the gaussian
            alpha: parameter where to switch from a gaussian to the powertail
            n: Exponent of the powertail
            obs: |@doc:pdf.init.obs| Observables of the
               model. This will be used as the default space of the PDF and,
               if not given explicitly, as the normalization range.

               The default space is used for example in the sample method: if no
               sampling limits are given, the default space is used.

               If the observables are binned and the model is unbinned, the
               model will be a binned model, by wrapping the model in a
               :py:class:`~zfit.pdf.BinnedFromUnbinnedPDF`, equivalent to
               calling :py:meth:`~zfit.pdf.BasePDF.to_binned`.

               If the observables are binned and the model is unbinned, the
               model will be a binned model, by wrapping the model in a
               :py:class:`~zfit.pdf.BinnedFromUnbinnedPDF`, equivalent to
               calling :py:meth:`~zfit.pdf.BasePDF.to_binned`.

               The observables are not equal to the domain as it does not restrict or
               truncate the model outside this range. |@docend:pdf.init.obs|
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

        .. _CBShape: https://en.wikipedia.org/wiki/Crystal_Ball_function
        """
        if name is None:
            name = "CrystalBall"
        params = {"mu": mu, "sigma": sigma, "alpha": alpha, "n": n}
        super().__init__(obs=obs, name=name, params=params, extended=extended, norm=norm, label=label)

    @supports(norm=False)
    def _pdf(self, x, norm, params):
        del norm
        mu = params["mu"]
        sigma = params["sigma"]
        alpha = params["alpha"]
        n = params["n"]
        x = z.unstack_x(x)
        return crystalball_func(x=x, mu=mu, sigma=sigma, alpha=alpha, n=n)


class CrystalBallPDFRepr(BasePDFRepr):
    _implementation = CrystalBall
    hs3_type: Literal["CrystalBall"] = pydantic.Field("CrystalBall", alias="type")
    x: SpaceRepr
    mu: Serializer.types.ParamTypeDiscriminated
    sigma: Serializer.types.ParamTypeDiscriminated
    alpha: Serializer.types.ParamTypeDiscriminated
    n: Serializer.types.ParamTypeDiscriminated


crystalball_integral_limits = Space(axes=(0,), lower=ANY_LOWER, upper=ANY_UPPER)

CrystalBall.register_analytic_integral(func=crystalball_integral, limits=crystalball_integral_limits)


class DoubleCB(BasePDF, SerializableMixin):
    _N_OBS = 1

    def __init__(
        self,
        mu: ztyping.ParamTypeInput,
        sigma: ztyping.ParamTypeInput,
        alphal: ztyping.ParamTypeInput,
        nl: ztyping.ParamTypeInput,
        alphar: ztyping.ParamTypeInput,
        nr: ztyping.ParamTypeInput,
        obs: ztyping.ObsTypeInput,
        *,
        extended: ExtendedInputType = None,
        norm: NormInputType = None,
        name: str | None = None,
        label: str | None = None,
    ):
        r"""Double-sided Crystal Ball shaped PDF. A combination of two CB using the **mu** (not a frac) on each side.

        The function is defined as follows:

        .. math::
            f(x;\mu, \sigma, \alpha_{L}, n_{L}, \alpha_{R}, n_{R}) =  \begin{cases}
            A_{L} \cdot (B_{L} - \frac{x - \mu}{\sigma})^{-n_{L}},
             & \mbox{for }\frac{x - \mu}{\sigma} < -\alpha_{L} \newline
            \exp(- \frac{(x - \mu)^2}{2 \sigma^2}),
            & \mbox{for }-\alpha_{L} \leqslant \frac{x - \mu}{\sigma} \leqslant \alpha_{R} \newline
            A_{R} \cdot (B_{R} + \frac{x - \mu}{\sigma})^{-n_{R}},
             & \mbox{for }\frac{x - \mu}{\sigma} > \alpha_{R}
            \end{cases}

        with

        .. math::
            A_{L/R} = \left(\frac{n_{L/R}}{\left| \alpha_{L/R} \right|}\right)^n_{L/R} \cdot
            \exp\left(- \frac {\left|\alpha_{L/R} \right|^2}{2}\right)

            B_{L/R} = \frac{n_{L/R}}{\left| \alpha_{L/R} \right|}  - \left| \alpha_{L/R} \right|

        Args:
            mu: The mean of the gaussian
            sigma: Standard deviation of the gaussian
            alphal: parameter where to switch from a gaussian to the powertail on the left
                side
            nl: Exponent of the powertail on the left side
            alphar: parameter where to switch from a gaussian to the powertail on the right
                side
            nr: Exponent of the powertail on the right side
            obs: |@doc:pdf.init.obs| Observables of the
               model. This will be used as the default space of the PDF and,
               if not given explicitly, as the normalization range.

               The default space is used for example in the sample method: if no
               sampling limits are given, the default space is used.

               If the observables are binned and the model is unbinned, the
               model will be a binned model, by wrapping the model in a
               :py:class:`~zfit.pdf.BinnedFromUnbinnedPDF`, equivalent to
               calling :py:meth:`~zfit.pdf.BasePDF.to_binned`.

               If the observables are binned and the model is unbinned, the
               model will be a binned model, by wrapping the model in a
               :py:class:`~zfit.pdf.BinnedFromUnbinnedPDF`, equivalent to
               calling :py:meth:`~zfit.pdf.BasePDF.to_binned`.

               The observables are not equal to the domain as it does not restrict or
               truncate the model outside this range. |@docend:pdf.init.obs|
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
        """
        if name is None:
            name = "DoubleCB"
        params = {
            "mu": mu,
            "sigma": sigma,
            "alphal": alphal,
            "nl": nl,
            "alphar": alphar,
            "nr": nr,
        }
        super().__init__(obs=obs, name=name, params=params, extended=extended, norm=norm, label=label)

    @supports(norm=False)
    def _pdf(self, x, norm, params):
        assert norm is False, "Norm cannot be a space"
        mu = params["mu"]
        sigma = params["sigma"]
        alphal = params["alphal"]
        nl = params["nl"]
        alphar = params["alphar"]
        nr = params["nr"]
        x = x[0]
        return double_crystalball_func(
            x=x,
            mu=mu,
            sigma=sigma,
            alphal=alphal,
            nl=nl,
            alphar=alphar,
            nr=nr,
        )


class DoubleCBPDFRepr(BasePDFRepr):
    _implementation = DoubleCB
    hs3_type: Literal["DoubleCB"] = pydantic.Field("DoubleCB", alias="type")
    x: SpaceRepr
    mu: Serializer.types.ParamTypeDiscriminated
    sigma: Serializer.types.ParamTypeDiscriminated
    alphal: Serializer.types.ParamTypeDiscriminated
    nl: Serializer.types.ParamTypeDiscriminated
    alphar: Serializer.types.ParamTypeDiscriminated
    nr: Serializer.types.ParamTypeDiscriminated


DoubleCB.register_analytic_integral(func=double_crystalball_mu_integral, limits=crystalball_integral_limits)


class GeneralizedCB(BasePDF, SerializableMixin):
    _N_OBS = 1

    def __init__(
        self,
        mu: ztyping.ParamTypeInput,
        sigmal: ztyping.ParamTypeInput,
        alphal: ztyping.ParamTypeInput,
        nl: ztyping.ParamTypeInput,
        sigmar: ztyping.ParamTypeInput,
        alphar: ztyping.ParamTypeInput,
        nr: ztyping.ParamTypeInput,
        obs: ztyping.ObsTypeInput,
        *,
        extended: ExtendedInputType = None,
        norm: NormInputType = None,
        name: str | None = None,
        label: str | None = None,
    ):
        r"""Generalized asymmetric double-sided Crystal Ball shaped PDF. A combination of two CB using the **mu** (not a
        frac) and a different **sigma** on each side.

        The function is defined as follows:

        .. math::
            f(x;\mu, \sigma_{L}, \alpha_{L}, n_{L}, \sigma_{R}, \alpha_{R}, n_{R}) =  \begin{cases}
            A_{L} \cdot (B_{L} - \frac{x - \mu}{\sigma_{L}})^{-n_{L}},
             & \mbox{for }\frac{x - \mu}{\sigma_{L}} < -\alpha_{L} \newline
            \exp(- \frac{(x - \mu)^2}{2 \sigma_{L}^2}),
            & \mbox{for }-\alpha_{L} \leqslant \frac{x - \mu}{\sigma_{L}} \leqslant 0 \newline
            \exp(- \frac{(x - \mu)^2}{2 \sigma_{R}^2}),
            & \mbox{for }0 \leqslant \frac{x - \mu}{\sigma_{R}} \leqslant \alpha_{R} \newline
            A_{R} \cdot (B_{R} + \frac{x - \mu}{\sigma_{R}})^{-n_{R}},
             & \mbox{for }\frac{x - \mu}{\sigma_{R}} > \alpha_{R}
            \end{cases}

        with

        .. math::
            A_{L/R} = \left(\frac{n_{L/R}}{\left| \alpha_{L/R} \right|}\right)^n_{L/R} \cdot
            \exp\left(- \frac {\left|\alpha_{L/R} \right|^2}{2}\right)

            B_{L/R} = \frac{n_{L/R}}{\left| \alpha_{L/R} \right|}  - \left| \alpha_{L/R} \right|

        Args:
            mu: The mean of the gaussian
            sigmal: Standard deviation of the gaussian on the left side
            alphal: parameter where to switch from a gaussian to the powertail on the left
                side
            nl: Exponent of the powertail on the left side
            sigmar: Standard deviation of the gaussian on the right side
            alphar: parameter where to switch from a gaussian to the powertail on the right
                side
            nr: Exponent of the powertail on the right side
            obs: |@doc:pdf.init.obs| Observables of the
               model. This will be used as the default space of the PDF and,
               if not given explicitly, as the normalization range.

               The default space is used for example in the sample method: if no
               sampling limits are given, the default space is used.

               If the observables are binned and the model is unbinned, the
               model will be a binned model, by wrapping the model in a
               :py:class:`~zfit.pdf.BinnedFromUnbinnedPDF`, equivalent to
               calling :py:meth:`~zfit.pdf.BasePDF.to_binned`.

               If the observables are binned and the model is unbinned, the
               model will be a binned model, by wrapping the model in a
               :py:class:`~zfit.pdf.BinnedFromUnbinnedPDF`, equivalent to
               calling :py:meth:`~zfit.pdf.BasePDF.to_binned`.

               The observables are not equal to the domain as it does not restrict or
               truncate the model outside this range. |@docend:pdf.init.obs|
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
        """
        if name is None:
            name = "GeneralizedCB"
        params = {
            "mu": mu,
            "sigmal": sigmal,
            "alphal": alphal,
            "nl": nl,
            "sigmar": sigmar,
            "alphar": alphar,
            "nr": nr,
        }
        super().__init__(obs=obs, name=name, params=params, extended=extended, norm=norm, label=label)

    @supports(norm=False)
    def _pdf(self, x, norm, params):
        assert norm is False, "Norm has to be False"
        mu = params["mu"]
        sigmal = params["sigmal"]
        alphal = params["alphal"]
        sigmar = params["sigmar"]
        nl = params["nl"]
        alphar = params["alphar"]
        nr = params["nr"]
        x = x[0]
        return generalized_crystalball_func(
            x=x,
            mu=mu,
            sigmal=sigmal,
            alphal=alphal,
            sigmar=sigmar,
            nl=nl,
            alphar=alphar,
            nr=nr,
        )


class GeneralizedCBPDFRepr(BasePDFRepr):
    _implementation = GeneralizedCB
    hs3_type: Literal["GeneralizedCB"] = pydantic.Field("GeneralizedCB", alias="type")
    x: SpaceRepr
    mu: Serializer.types.ParamTypeDiscriminated
    sigmal: Serializer.types.ParamTypeDiscriminated
    alphal: Serializer.types.ParamTypeDiscriminated
    sigmar: Serializer.types.ParamTypeDiscriminated
    nl: Serializer.types.ParamTypeDiscriminated
    alphar: Serializer.types.ParamTypeDiscriminated
    nr: Serializer.types.ParamTypeDiscriminated


GeneralizedCB.register_analytic_integral(func=generalized_crystalball_mu_integral, limits=crystalball_integral_limits)


@z.function(wraps="tensor", keepalive=True)
def gaussexptail_func(x, mu, sigma, alpha):
    t = (x - mu) / sigma * tf.sign(alpha)
    abs_alpha = znp.abs(alpha)
    cond = tf.less(t, -abs_alpha)
    func = z.safe_where(
        cond,
        lambda t: 0.5 * znp.square(abs_alpha) + abs_alpha * t,
        lambda t: -0.5 * znp.square(t),
        values=t,
        value_safer=lambda t: znp.ones_like(t) * abs_alpha,
    )
    return znp.exp(func)


@z.function(wraps="tensor", keepalive=True, stateless_args=False)
def generalized_gaussexptail_func(x, mu, sigmal, alphal, sigmar, alphar):
    cond = tf.less(x, mu)

    return tf.where(
        cond,
        gaussexptail_func(x, mu, sigmal, alphal),
        gaussexptail_func(x, mu, sigmar, -alphar),
    )










class GaussExpTail(BasePDF, SerializableMixin):
    _N_OBS = 1

    def __init__(
        self,
        mu: ztyping.ParamTypeInput,
        sigma: ztyping.ParamTypeInput,
        alpha: ztyping.ParamTypeInput,
        obs: ztyping.ObsTypeInput,
        *,
        extended: ExtendedInputType = None,
        norm: NormInputType = None,
        name: str | None = None,
        label: str | None = None,
    ):
        r"""GaussExpTail shaped PDF. A combination of a Gaussian with an exponential tail on one side.

        The function is defined as follows:

        .. math::
            f(x;\mu, \sigma, \alpha) =  \begin{cases} \exp(- \frac{(x - \mu)^2}{2 \sigma^2}),
            & \mbox{for }\frac{x - \mu}{\sigma} \geqslant -\alpha \newline
            \exp{\left(\frac{|\alpha|^2}{2} + |\alpha|  \left(\frac{x - \mu}{\sigma}\right)\right)},
            & \mbox{for }\frac{x - \mu}{\sigma} < -\alpha \end{cases}

        Args:
            mu: The mean of the gaussian
            sigma: Standard deviation of the gaussian
            alpha: parameter where to switch from a gaussian to the expontial tail
            obs: |@doc:pdf.init.obs| Observables of the
               model. This will be used as the default space of the PDF and,
               if not given explicitly, as the normalization range.

               The default space is used for example in the sample method: if no
               sampling limits are given, the default space is used.

               If the observables are binned and the model is unbinned, the
               model will be a binned model, by wrapping the model in a
               :py:class:`~zfit.pdf.BinnedFromUnbinnedPDF`, equivalent to
               calling :py:meth:`~zfit.pdf.BasePDF.to_binned`.

               If the observables are binned and the model is unbinned, the
               model will be a binned model, by wrapping the model in a
               :py:class:`~zfit.pdf.BinnedFromUnbinnedPDF`, equivalent to
               calling :py:meth:`~zfit.pdf.BasePDF.to_binned`.

               The observables are not equal to the domain as it does not restrict or
               truncate the model outside this range. |@docend:pdf.init.obs|
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
        """
        if name is None:
            name = "GaussExpTail"
        params = {"mu": mu, "sigma": sigma, "alpha": alpha}
        super().__init__(obs=obs, name=name, params=params, extended=extended, norm=norm, label=label)

    @supports(norm=False)
    def _pdf(self, x, norm, params):
        assert norm is False, "Norm has to be False"
        mu = params["mu"]
        sigma = params["sigma"]
        alpha = params["alpha"]
        x = x[0]
        return gaussexptail_func(x=x, mu=mu, sigma=sigma, alpha=alpha)


class GaussExpTailPDFRepr(BasePDFRepr):
    _implementation = GaussExpTail
    hs3_type: Literal["GaussExpTail"] = pydantic.Field("GaussExpTail", alias="type")
    x: SpaceRepr
    mu: Serializer.types.ParamTypeDiscriminated
    sigma: Serializer.types.ParamTypeDiscriminated
    alpha: Serializer.types.ParamTypeDiscriminated


gaussexptail_integral_limits = Space(axes=0, limits=(ANY_LOWER, ANY_UPPER))

GaussExpTail.register_analytic_integral(func=gaussexptail_integral, limits=gaussexptail_integral_limits)


class GeneralizedGaussExpTail(BasePDF, SerializableMixin):
    _N_OBS = 1

    def __init__(
        self,
        mu: ztyping.ParamTypeInput,
        sigmal: ztyping.ParamTypeInput,
        alphal: ztyping.ParamTypeInput,
        sigmar: ztyping.ParamTypeInput,
        alphar: ztyping.ParamTypeInput,
        obs: ztyping.ObsTypeInput,
        *,
        extended: ExtendedInputType = None,
        norm: NormInputType = None,
        name: str | None = None,
        label: str | None = None,
    ):
        r"""GeneralizedGaussedExpTail shaped PDF which is Generalized assymetric double-sided GaussExpTail shaped PDF. A
        combination of two GaussExpTail using the **mu** (not a frac) and a different **sigma** on each side.

        The function is defined as follows:

        .. math::
            f(x;\mu, \sigma_{L}, \alpha_{L}, \sigma_{R}, \alpha_{R}) =  \begin{cases}
            \exp{\left(\frac{|\alpha_{L}|^2}{2} + |\alpha_{L}|  \left(\frac{x - \mu}{\sigma_{L}}\right)\right)},
             & \mbox{for }\frac{x - \mu}{\sigma_{L}} < -\alpha_{L} \newline
            \exp(- \frac{(x - \mu)^2}{2 \sigma_{L}^2}),
            & \mbox{for }-\alpha_{L} \leqslant \frac{x - \mu}{\sigma_{L}} \leqslant 0 \newline
            \exp(- \frac{(x - \mu)^2}{2 \sigma_{R}^2}),
            & \mbox{for }0 \leqslant \frac{x - \mu}{\sigma_{R}} \leqslant \alpha_{R} \newline
            \exp{\left(\frac{|\alpha_{R}|^2}{2} - |\alpha_{R}|  \left(\frac{x - \mu}{\sigma_{R}}\right)\right)},
             & \mbox{for }\frac{x - \mu}{\sigma_{R}} > \alpha_{R}
            \end{cases}

        Args:
            mu: The mean of the gaussian
            sigmal: Standard deviation of the gaussian on the left side
            alphal: parameter where to switch from a gaussian to the expontial tail on the left side
            sigmar: Standard deviation of the gaussian on the right side
            alphar: parameter where to switch from a gaussian to the expontial tail on the right side
            obs: |@doc:pdf.init.obs| Observables of the
               model. This will be used as the default space of the PDF and,
               if not given explicitly, as the normalization range.

               The default space is used for example in the sample method: if no
               sampling limits are given, the default space is used.

               If the observables are binned and the model is unbinned, the
               model will be a binned model, by wrapping the model in a
               :py:class:`~zfit.pdf.BinnedFromUnbinnedPDF`, equivalent to
               calling :py:meth:`~zfit.pdf.BasePDF.to_binned`.

               If the observables are binned and the model is unbinned, the
               model will be a binned model, by wrapping the model in a
               :py:class:`~zfit.pdf.BinnedFromUnbinnedPDF`, equivalent to
               calling :py:meth:`~zfit.pdf.BasePDF.to_binned`.

               The observables are not equal to the domain as it does not restrict or
               truncate the model outside this range. |@docend:pdf.init.obs|
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
        """
        if name is None:
            name = "GeneralizedGaussExpTail"
        params = {
            "mu": mu,
            "sigmal": sigmal,
            "alphal": alphal,
            "sigmar": sigmar,
            "alphar": alphar,
        }
        super().__init__(obs=obs, name=name, params=params, extended=extended, norm=norm, label=label)

    @supports(norm=False)
    def _pdf(self, x, norm, params):
        assert norm is False, "Norm has to be False"
        mu = params["mu"]
        sigmal = params["sigmal"]
        alphal = params["alphal"]
        sigmar = params["sigmar"]
        alphar = params["alphar"]
        x = x[0]
        return generalized_gaussexptail_func(
            x=x,
            mu=mu,
            sigmal=sigmal,
            alphal=alphal,
            sigmar=sigmar,
            alphar=alphar,
        )


class GeneralizedGaussExpTailPDFRepr(BasePDFRepr):
    _implementation = GeneralizedGaussExpTail
    hs3_type: Literal["GeneralizedGaussExpTail"] = pydantic.Field("GeneralizedGaussExpTail", alias="type")
    x: SpaceRepr
    mu: Serializer.types.ParamTypeDiscriminated
    sigmal: Serializer.types.ParamTypeDiscriminated
    alphal: Serializer.types.ParamTypeDiscriminated
    sigmar: Serializer.types.ParamTypeDiscriminated
    alphar: Serializer.types.ParamTypeDiscriminated


GeneralizedGaussExpTail.register_analytic_integral(
    func=generalized_gaussexptail_integral, limits=gaussexptail_integral_limits
)
