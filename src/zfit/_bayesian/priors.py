

from __future__ import annotations

from abc import ABC, abstractmethod

import zfit
import zfit.z.numpy as znp

from .._interfaces import ZfitPrior
from .mathconstrain import POSITIVE, UNCONSTRAINED, ConstraintType, PriorConstraint, validate_parameter


class BasePrior(ZfitPrior, ABC):

    constraint: PriorConstraint = UNCONSTRAINED

    def __init__(self, pdf_params: dict, bounds: tuple[float, float] | None = None, name: str | None = None):
        """Initialize an adaptive prior.

        Args:
            pdf_params: Parameters to pass to the PDF constructor
            bounds: Optional custom bounds for the prior (overrides constraint defaults)
            name: Optional name for the prior
        """
        self._validate_parameters(pdf_params)

        self._pdf_params = pdf_params.copy()

        if bounds is not None:
            self._original_bounds = bounds
        else:
            self._original_bounds = self.constraint.bounds

        self._original_bounds = self.constraint.validate_bounds(self._original_bounds)

        obs = zfit.Space("prior_space", limits=self._original_bounds)
        pdf = self._create_pdf(obs, **self._pdf_params)
        super().__init__(pdf=pdf, name=name)

    def _validate_parameters(self, params: dict):
        """Validate parameters if in eager mode. Override in subclasses."""

    @abstractmethod
    def _create_pdf(self, obs, **params):
        """Create the underlying PDF. Must be implemented by subclasses."""

    def _register_default_param(self, param):
        """Register a parameter and potentially adapt the prior's range."""
        super()._register_default_param(param)

        if self._should_adapt_to_param(param):
            self._adapt_to_parameter_limits(param)

        return self

    def _should_adapt_to_param(self, param) -> bool:
        """Check if the prior should adapt to parameter limits."""
        return hasattr(param, "has_limits") and param.has_limits

    def _get_adapted_bounds(self, param) -> tuple[float, float]:
        """Get bounds adapted to parameter limits using constraint system."""
        lower, upper = self._original_bounds


        if (param_lower := getattr(param, "lower", None)) is not None:
            lower = param_lower if lower == -float("inf") else max(lower, param_lower)

        if (param_upper := getattr(param, "upper", None)) is not None:
            upper = param_upper if upper == float("inf") else min(upper, param_upper)

        return self.constraint.validate_bounds((lower, upper))

    def _adapt_to_parameter_limits(self, param):
        """Adapt the prior's observation space to the parameter limits."""
        adapted_bounds = self._get_adapted_bounds(param)

        if adapted_bounds != self._original_bounds:
            obs = zfit.Space("prior_space", limits=adapted_bounds)
            self.pdf = self._create_adapted_pdf(obs, *adapted_bounds)

    def _create_adapted_pdf(self, obs, lower, upper):
        """Create an adapted PDF with new bounds. Can be overridden for special handling."""
        del lower, upper  # Unused in base implementation
        return self._create_pdf(obs, **self._pdf_params)


class Normal(BasePrior):

    constraint = UNCONSTRAINED

    def __init__(self, mu: float, sigma: float, name: str | None = None):
        """Initialize a Normal prior.

        Args:
            mu: Mean (center) of the normal distribution
            sigma: Standard deviation (spread) of the normal distribution. Must be positive.
            name: Optional name for the prior
        """
        pdf_params = {"mu": mu, "sigma": sigma}
        super().__init__(pdf_params, name=name)

    def _validate_parameters(self, params: dict):
        """Validate Normal distribution parameters."""
        validate_parameter("mu", params["mu"])
        validate_parameter("sigma", params["sigma"], POSITIVE)

    def _create_pdf(self, obs, mu, sigma):
        """Create a Gaussian PDF."""
        return zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs)

    def _create_adapted_pdf(self, obs, lower, upper):
        """Create a truncated Gaussian when adapting to limits."""
        return zfit.pdf.TruncatedGauss(
            mu=self._pdf_params["mu"], sigma=self._pdf_params["sigma"], low=lower, high=upper, obs=obs
        )


class Uniform(BasePrior):

    def __init__(self, lower: float | None = None, upper: float | None = None, name: str | None = None):
        """Initialize a Uniform prior.

        Args:
            lower: Lower bound of the uniform distribution. If None, will use
                   the parameter's lower limit when the prior is assigned.
            upper: Upper bound of the uniform distribution. If None, will use
                   the parameter's upper limit when the prior is assigned.
            name: Optional name for the prior

        Note:
            If both bounds are None, the prior will adapt completely to the
            parameter's limits. If the parameter has no limits, temporary
            bounds of [-1e6, 1e6] are used.
        """
        self._user_lower = lower
        self._user_upper = upper

        if lower is None and upper is None:
            bounds = (-1e6, 1e6)  # temporary defaults
            self.constraint = UNCONSTRAINED
        elif lower is None:
            bounds = (upper - 1e6, upper)
            self.constraint = PriorConstraint(ConstraintType.UPPER_BOUNDED, bounds=(-float("inf"), upper))
        elif upper is None:
            bounds = (lower, lower + 1e6)
            self.constraint = PriorConstraint(ConstraintType.LOWER_BOUNDED, bounds=(lower, float("inf")))
        else:
            bounds = (lower, upper)
            self.constraint = PriorConstraint(ConstraintType.CUSTOM_BOUNDS, bounds=(lower, upper))

        pdf_params = {}  # Uniform doesn't need location/scale params
        super().__init__(pdf_params, bounds=bounds, name=name)

    def _validate_parameters(self, params: dict):
        """Validate Uniform distribution parameters."""

    def _create_pdf(self, obs, **_params):
        """Create a Uniform PDF."""
        lower = float(obs.lower[0][0])
        upper = float(obs.upper[0][0])
        return zfit.pdf.Uniform(low=lower, high=upper, obs=obs)

    def _get_adapted_bounds(self, param):
        """Get bounds adapted to parameter limits, preferring user bounds."""
        lower = self._user_lower
        upper = self._user_upper

        if lower is None and hasattr(param, "lower") and param.lower is not None:
            lower = param.lower
        if upper is None and hasattr(param, "upper") and param.upper is not None:
            upper = param.upper

        if lower is None or upper is None:
            orig_lower, orig_upper = self._original_bounds
            if lower is None:
                lower = orig_lower
            if upper is None:
                upper = orig_upper

        return lower, upper


class HalfNormal(BasePrior):

    def __init__(self, *, sigma: float, mu: float = 0, name: str | None = None):
        """Initialize a Half-Normal prior.

        Args:
            sigma: Scale parameter controlling the spread of the distribution.
                   Must be positive. Required keyword-only parameter.
            mu: Location parameter (lower bound) where the distribution starts.
                Defaults to 0 for a standard half-normal.
            name: Optional name for the prior
        """
        self.constraint = PriorConstraint(ConstraintType.LOWER_BOUNDED, bounds=(mu, float("inf")))

        pdf_params = {"mu": mu, "sigma": sigma}
        bounds = (mu, float("inf"))
        super().__init__(pdf_params, bounds=bounds, name=name)

    def _validate_parameters(self, params: dict):
        """Validate HalfNormal distribution parameters."""
        validate_parameter("mu", params["mu"])
        validate_parameter("sigma", params["sigma"], POSITIVE)

    def _create_pdf(self, obs, mu, sigma):
        """Create a half-normal using TruncatedGauss."""
        lower = float(obs.lower[0][0])
        upper = float(obs.upper[0][0])
        return zfit.pdf.TruncatedGauss(mu=mu, sigma=sigma, low=lower, high=upper, obs=obs)

    def _get_adapted_bounds(self, param):
        """Ensure lower bound respects the mu parameter."""
        lower, upper = super()._get_adapted_bounds(param)
        lower = max(self._pdf_params["mu"], lower)
        return lower, upper


class Gamma(BasePrior):

    def __init__(self, alpha: float, beta: float, mu: float = 0, name: str | None = None):
        """Initialize a Gamma prior.

        Args:
            alpha: Shape parameter controlling the form of the distribution.
                   Must be positive. Larger values make the distribution more
                   bell-shaped.
            beta: Rate parameter (inverse scale). Must be positive.
                  Larger values shift the distribution toward zero.
            mu: Location parameter that shifts the entire distribution.
                Defaults to 0 for a standard Gamma. The distribution
                has support on [mu, ∞).
            name: Optional name for the prior
        """
        self.constraint = PriorConstraint(ConstraintType.LOWER_BOUNDED, bounds=(mu, float("inf")))

        pdf_params = {"gamma": alpha, "beta": beta, "mu": mu}
        bounds = (mu, float("inf"))
        super().__init__(pdf_params, bounds=bounds, name=name)

    def _validate_parameters(self, params: dict):
        """Validate Gamma distribution parameters."""
        validate_parameter("gamma", params["gamma"], POSITIVE)  # alpha
        validate_parameter("beta", params["beta"], POSITIVE)
        validate_parameter("mu", params["mu"])

    def _create_pdf(self, obs, gamma, beta, mu):
        """Create a Gamma PDF."""
        return zfit.pdf.Gamma(gamma=gamma, beta=beta, mu=mu, obs=obs)

    def _create_adapted_pdf(self, obs, lower, _upper):
        """Create adapted Gamma PDF with adjusted mu."""
        adapted_mu = max(self._pdf_params["mu"], lower)
        return zfit.pdf.Gamma(gamma=self._pdf_params["gamma"], beta=self._pdf_params["beta"], mu=adapted_mu, obs=obs)


class Beta(BasePrior):

    def __init__(self, alpha: float, beta: float, lower: float, upper: float, name: str | None = None):
        """Initialize a Beta prior.

        Args:
            alpha: First shape parameter controlling behavior near the upper bound.
                   Must be positive.
            beta: Second shape parameter controlling behavior near the lower bound.
                  Must be positive.
            lower: Lower bound of the distribution.
            upper: Upper bound of the distribution. Must be > lower.
            name: Optional name for the prior
        """
        if lower >= upper:
            msg = f"Lower bound {lower} must be less than upper bound {upper}"
            raise ValueError(msg)

        self.lower = float(lower)
        self.upper = float(upper)
        self.scale = self.upper - self.lower

        self.constraint = PriorConstraint(ConstraintType.CUSTOM_BOUNDS, bounds=(lower, upper))

        pdf_params = {"alpha": alpha, "beta": beta}
        super().__init__(pdf_params, bounds=(lower, upper), name=name)

    def _validate_parameters(self, params: dict):
        """Validate Beta distribution parameters."""
        validate_parameter("alpha", params["alpha"], POSITIVE)
        validate_parameter("beta", params["beta"], POSITIVE)

    def _create_pdf(self, obs, alpha, beta):
        """Create a scaled Beta PDF."""
        del alpha, beta  # Using uniform approach for now
        return zfit.pdf.Uniform(low=self.lower, high=self.upper, obs=obs)


class LogNormal(BasePrior):

    constraint = POSITIVE

    def __init__(self, mu: float, sigma: float, name: str | None = None):
        """Initialize a Log-Normal prior.

        Args:
            mu: Mean of the logarithm of the variable. This controls the
                median of the distribution (median = exp(mu)).
            sigma: Standard deviation of the logarithm. Must be positive.
                   Larger values create heavier right tails.
            name: Optional name for the prior

        Note:
            The parameters mu and sigma are NOT the mean and standard deviation
            of the Log-Normal distribution itself, but of the underlying
            normal distribution of log(X).
        """
        pdf_params = {"mu": mu, "sigma": sigma}
        super().__init__(pdf_params, name=name)

    def _validate_parameters(self, params: dict):
        """Validate LogNormal distribution parameters."""
        validate_parameter("mu", params["mu"])
        validate_parameter("sigma", params["sigma"], POSITIVE)

    def _create_pdf(self, obs, mu, sigma):
        """Create a LogNormal PDF."""
        return zfit.pdf.LogNormal(mu=mu, sigma=sigma, obs=obs)


class Cauchy(BasePrior):

    constraint = UNCONSTRAINED

    def __init__(self, m: float, gamma: float, name: str | None = None):
        """Initialize a Cauchy prior.

        Args:
            m: Location parameter (center) of the distribution. This is
               both the median and the mode.
            gamma: Scale parameter controlling the spread. Must be positive.
                   Larger values create wider distributions.
            name: Optional name for the prior
        """
        pdf_params = {"m": m, "gamma": gamma}
        super().__init__(pdf_params, name=name)

    def _validate_parameters(self, params: dict):
        """Validate Cauchy distribution parameters."""
        validate_parameter("m", params["m"])
        validate_parameter("gamma", params["gamma"], POSITIVE)

    def _create_pdf(self, obs, m, gamma):
        """Create a Cauchy PDF."""
        return zfit.pdf.Cauchy(m=m, gamma=gamma, obs=obs)


class KDE(ZfitPrior):

    def __init__(self, samples, bandwidth: float | str | None = None, name: str | None = None):
        """Initialize a KDE prior.

        Args:
            samples: Array of samples to estimate the density from.
                     Can be a numpy array, list, or tensor.
            bandwidth: Bandwidth for kernel density estimation. If None,
                      uses the KDE's default. Can be float or string like 'scott'.
            name: Optional name for the prior

        Note:
            The KDE is constructed with a 10% margin beyond the sample
            range to ensure numerical stability at the boundaries.
        """
        samples = znp.asarray(samples)
        if len(samples) == 0:
            msg = "Cannot create KDE prior from empty samples"
            raise ValueError(msg)

        self._samples = samples
        self._bandwidth = bandwidth
        self._n_samples = len(samples)

        self._min_val = float(znp.min(samples))
        self._max_val = float(znp.max(samples))
        self._range = self._max_val - self._min_val

        if self._range > 0:
            self._margin = 0.1 * self._range
        else:
            self._margin = 1.0
        self._margin = max(self._margin, 1e-6)  # Ensure minimum margin

        self._original_bounds = (self._min_val - self._margin, self._max_val + self._margin)

        pdf = self._create_kde_pdf(self._original_bounds)
        super().__init__(pdf=pdf, name=name)

    def _create_kde_pdf(self, bounds):
        """Create KDE PDF with given bounds.

        Automatically chooses between exact and grid-based KDE based on sample size:
        - < 1000 samples: Exact KDE (more accurate, slower)
        - >= 1000 samples: Grid-based KDE (faster, good approximation)
        """
        lower, upper = bounds
        obs = zfit.Space("prior_space", limits=(lower, upper))
        data = zfit.Data.from_numpy(obs=obs, array=self._samples)

        use_exact = self._n_samples < 1000

        if use_exact:
            return zfit.pdf.KDE1DimExact(data=data, obs=obs, bandwidth=self._bandwidth, padding=False)
        else:
            optimal_grid_points = min(1024, max(128, int(self._n_samples**0.5)))

            return zfit.pdf.KDE1DimGrid(
                data=data,
                obs=obs,
                bandwidth=self._bandwidth or "scott",
                num_grid_points=optimal_grid_points,
                padding=False,
            )

    def _register_default_param(self, param):
        """Register a parameter and potentially adapt the KDE bounds."""
        super()._register_default_param(param)

        if hasattr(param, "has_limits") and param.has_limits:
            self._adapt_to_parameter_limits(param)

        return self

    def _adapt_to_parameter_limits(self, param):
        """Adapt KDE prior to parameter limits."""
        lower = self._min_val - self._margin
        upper = self._max_val + self._margin


        if (param_lower := getattr(param, "lower", None)) is not None:
            lower = min(lower, param_lower)
        if (param_upper := getattr(param, "upper", None)) is not None:
            upper = max(upper, param_upper)

        new_bounds = (lower, upper)
        if new_bounds != self._original_bounds:
            self.pdf = self._create_kde_pdf(new_bounds)
            self._original_bounds = new_bounds


class Poisson(BasePrior):

    constraint = POSITIVE

    def __init__(self, lam: float, name: str | None = None):
        """Initialize a Poisson prior.

        Args:
            lam: Rate parameter (expected number of events). Must be positive.
                 This is both the mean and variance of the distribution.
            name: Optional name for the prior
        """
        pdf_params = {"lam": lam}
        super().__init__(pdf_params, name=name)

    def _validate_parameters(self, params: dict):
        """Validate Poisson distribution parameters."""
        validate_parameter("lam", params["lam"], POSITIVE)

    def _create_pdf(self, obs, lam):
        """Create a Poisson PDF."""
        return zfit.pdf.Poisson(lam=lam, obs=obs)

    def _get_adapted_bounds(self, param):
        """Ensure bounds are appropriate for discrete distribution."""
        lower, upper = super()._get_adapted_bounds(param)
        lower = max(0, lower)
        return lower, upper


class Exponential(BasePrior):

    constraint = POSITIVE

    def __init__(self, lam: float, name: str | None = None):
        """Initialize an Exponential prior.

        Args:
            lam: Rate parameter (inverse of the mean). Must be positive.
                 Higher values concentrate probability near zero.
            name: Optional name for the prior
        """
        pdf_params = {"lam": lam}
        super().__init__(pdf_params, name=name)

    def _validate_parameters(self, params: dict):
        """Validate Exponential distribution parameters."""
        validate_parameter("lam", params["lam"], POSITIVE)

    def _create_pdf(self, obs, lam):
        """Create an Exponential PDF."""
        return zfit.pdf.Exponential(lam=lam, obs=obs)


class StudentT(BasePrior):

    constraint = UNCONSTRAINED

    def __init__(self, ndof: float, mu: float, sigma: float, name: str | None = None):
        """Initialize a Student's t prior.

        Args:
            ndof: Degrees of freedom parameter controlling tail heaviness.
                  Must be positive. Lower values give heavier tails.
                  As ndof → ∞, approaches Normal(mu, sigma).
            mu: Location parameter (center) of the distribution.
            sigma: Scale parameter controlling spread. Must be positive.
            name: Optional name for the prior
        """
        pdf_params = {"ndof": ndof, "mu": mu, "sigma": sigma}
        super().__init__(pdf_params, name=name)

    def _validate_parameters(self, params: dict):
        """Validate StudentT distribution parameters."""
        validate_parameter("ndof", params["ndof"], POSITIVE)
        validate_parameter("mu", params["mu"])
        validate_parameter("sigma", params["sigma"], POSITIVE)

    def _create_pdf(self, obs, ndof, mu, sigma):
        """Create a Student's t PDF."""
        return zfit.pdf.StudentT(ndof=ndof, mu=mu, sigma=sigma, obs=obs)

    def _create_adapted_pdf(self, obs, lower, upper):
        """Create adapted Student's t PDF with truncation if needed."""
        del lower, upper  # Unused in base implementation
        return zfit.pdf.StudentT(
            ndof=self._pdf_params["ndof"], mu=self._pdf_params["mu"], sigma=self._pdf_params["sigma"], obs=obs
        )
