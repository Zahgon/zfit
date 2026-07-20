

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import zfit
from zfit.util.container import convert_to_container, is_container

if TYPE_CHECKING:
    from collections.abc import Iterable

    import arviz as az
    import numpy.typing as npt
    import pandas as pd

    from zfit.mcmc import MCMCSampler

    from .._interfaces import ZfitLoss, ZfitParameter
    from .priors import KDE


class PosteriorSamples:
    def __init__(
        self,
        samples: npt.NDArray[np.float64],
        params: Iterable[ZfitParameter],
        loss: ZfitLoss,
        sampler: MCMCSampler,
        n_warmup: int,
        n_samples: int,
        raw_result: object | None = None,
        info: dict | None = None,
    ):
        """Posterior samples from MCMC bayesian inference.

        Args:
            samples: Array of shape (n_samples, n_params) containing MCMC samples.
            params: List of ZfitParameter objects.
            loss: The ZfitLoss that was sampled.
            sampler: The ZfitSampler that generated the samples.
            n_warmup: Number of warmup/burn-in steps.
            n_samples: Number of posterior samples per walker.
            raw_result: Raw result from the sampler.
            info: Additional information dictionary.
        """

        if not isinstance(n_warmup, (int, np.integer)) or n_warmup < 0:
            msg = f"n_warmup must be a non-negative integer, got {n_warmup}"
            raise ValueError(msg)

        if not isinstance(n_samples, (int, np.integer)) or n_samples <= 0:
            msg = f"n_samples must be a positive integer, got {n_samples}"
            raise ValueError(msg)

        if info is None:
            info = {}

        self.samples = np.asarray(samples)

        if self.samples.ndim != 2:
            msg = f"samples must be a 2D array, got shape {self.samples.shape}"
            raise ValueError(msg)

        if len(self.samples) == 0:
            msg = "samples cannot be empty"
            raise ValueError(msg)

        self._params = convert_to_container(params)

        if not self._params:
            msg = "params cannot be empty"
            raise ValueError(msg)

        if self.samples.shape[1] != len(self._params):
            msg = f"Number of parameters in samples ({self.samples.shape[1]}) does not match number of parameters ({len(self._params)})"
            raise ValueError(msg)

        self._loss = loss
        self._sampler = sampler
        self.n_warmup = n_warmup
        self.n_samples = n_samples
        self.raw_result = raw_result
        self.info = info

        self._param_by_name = {param.name: param for param in self._params}
        self._name_by_param = {param: param.name for param in self._params}

        self._position_by_name = {param.name: i for i, param in enumerate(self._params)}

        self._compute_convergence_diagnostics()

    def mean(
        self, params: str | ZfitParameter | Iterable[str | ZfitParameter] | None = None
    ) -> float | npt.NDArray[np.float64]:
        """Posterior mean(s).

        Args:
            params: Parameter name, object, index, or list thereof. If None, return all means.

        Returns:
            Mean value(s).
            - Single parameter: returns float.
            - Collection of parameters: returns array.
        """
        if len(self.samples) == 0:
            msg = "Cannot compute mean of empty samples"
            raise ValueError(msg)

        if params is None:
            params = [param.name for param in self._params]
            was_container = True
        else:
            was_container = is_container(params)

        indices = self._get_param_positions(params)

        if not indices:
            msg = "No valid parameters specified for mean calculation"
            raise ValueError(msg)

        samples_np = np.asarray(self.samples)
        selected_samples = samples_np[:, indices]
        means = np.mean(selected_samples, axis=0)

        if not was_container:
            return float(means[0])
        return means

    def symerr(
        self,
        params: str | ZfitParameter | Iterable[str | ZfitParameter] | None = None,
        *,
        sigma: float | None = None,
    ) -> float | npt.NDArray[np.float64]:
        pass

    def std(
        self,
        params: str | ZfitParameter | Iterable[str | ZfitParameter] | None = None,
    ) -> float | npt.NDArray[np.float64]:
        """Standard deviation of posterior samples

        Args:
            params: Parameter name, object, index, or list thereof. If None, return all stds.

        Returns:
            Standard deviation(s).
            - Single parameter: returns float.
            - Collection of parameters: returns array.

        Examples:
            >>> result.std()  # All parameters
            array([0.102, 0.234])

            >>> result.std(['mu', 'sigma'])  # Multiple parameters
            array([0.102, 0.234])

            >>> result.std('mu')  # Single parameter
            0.102

            >>> result.std(['mu'])  # Single parameter in list
            array([0.102])
        """
        if len(self.samples) == 0:
            msg = "Cannot compute standard deviation of empty samples"
            raise ValueError(msg)

        if params is None:
            params = [param.name for param in self._params]
            was_container = True
        else:
            was_container = is_container(params)

        indices = self._get_param_positions(params)

        if not indices:
            msg = "No valid parameters specified for std calculation"
            raise ValueError(msg)

        samples_np = np.asarray(self.samples)
        selected_samples = samples_np[:, indices]
        stds = np.std(selected_samples, axis=0)

        if not was_container:
            return float(stds[0])
        return stds

    def credible_interval(
        self,
        params: str | ZfitParameter | Iterable[str | ZfitParameter] | None = None,
        *,
        alpha: float | None = None,
        sigma: float | None = None,
    ) -> tuple[float | npt.NDArray[np.float64], float | npt.NDArray[np.float64]]:
        pass

    def get_samples(
        self,
        params: str | ZfitParameter | Iterable[str | ZfitParameter] | None = None,
    ) -> npt.NDArray[np.float64]:
        pass

    def as_prior(self, param: str | ZfitParameter | int) -> KDE:
        pass

    def to_arviz(self) -> az.InferenceData:
        """Convert to ArviZ InferenceData format.

        Returns:
            ArviZ InferenceData object for advanced analysis.
        """
        try:
            import arviz as az  # noqa: PLC0415
        except ImportError as error:
            msg = "ArviZ is required for to_arviz(). Install with 'pip install arviz'."
            raise ImportError(msg) from error

        samples_np = self.samples

        total_samples = len(samples_np)

        if hasattr(self._sampler, "nwalkers") and self._sampler.nwalkers is not None:
            nwalkers = self._sampler.nwalkers
            ndraws = total_samples // nwalkers
        else:
            nwalkers = 1
            ndraws = total_samples

        if nwalkers > 1:
            samples_reshaped = np.reshape(samples_np, (nwalkers, ndraws, -1))
        else:
            samples_reshaped = samples_np[np.newaxis, :, :]  # Add chain dimension

        return az.from_dict(
            {param.name: samples_reshaped[:, :, i] for i, param in enumerate(self._params)},
            coords={"chain": range(nwalkers), "draw": range(ndraws)},
        )

    def update_params(
        self,
        params: str | ZfitParameter | Iterable[str | ZfitParameter] | None = None,
        *,
        what: str | None = None,
    ) -> PosteriorSamples:
        pass

    def _set_params_to_mean(
        self,
        params: str | ZfitParameter | Iterable[str | ZfitParameter] | None = None,
    ) -> PosteriorSamples:
        pass

    def __enter__(self):
        """Context manager: set parameters to posterior means."""
        self._old_values = [param.value() for param in self._params]
        self.update_params()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager: restore original parameter values."""
        from zfit.core.parameter import set_values  # noqa: PLC0415

        set_values(self._params, self._old_values)

    @property
    def params(self) -> list[ZfitParameter]:
        pass

    @property
    def param_names(self) -> list[str]:
        pass

    @property
    def sampler(self) -> MCMCSampler:
        pass

    @property
    def loss(self) -> ZfitLoss:
        pass

    @property
    def valid(self) -> bool:
        pass

    @property
    def converged(self) -> bool:
        """Whether the MCMC chains have converged based on diagnostics.

        Convergence is determined by:
        - R-hat < 1.1 for all parameters (Gelman-Rubin statistic)
        - Effective sample size > 100 for all parameters
        - No NaN or infinite values
        """
        return self._converged

    def covariance(
        self,
        params: str | ZfitParameter | Iterable[str | ZfitParameter] | None = None,
    ) -> npt.NDArray[np.float64]:
        """Covariance matrix from posterior samples.

        Args:
            params: Parameters to include. If None, use all parameters.

        Returns:
            Covariance matrix as numpy array.
            - Single parameter: returns scalar (variance).
            - Collection of parameters: returns matrix.
        """
        if len(self.samples) == 0:
            msg = "Cannot compute covariance of empty samples"
            raise ValueError(msg)

        if len(self.samples) < 2:
            msg = f"Need at least 2 samples to compute covariance, got {len(self.samples)}"
            raise ValueError(msg)

        if params is None:
            params = [param.name for param in self._params]
            was_container = True
        else:
            was_container = is_container(params)

        indices = self._get_param_positions(params)

        if not indices:
            msg = "No valid parameters specified for covariance calculation"
            raise ValueError(msg)

        samples_np = np.asarray(self.samples)
        selected_samples = samples_np[:, indices]

        if len(indices) == 1 and not was_container:
            variance = np.var(selected_samples, axis=0, ddof=1)
            return float(variance[0]) if variance.ndim > 0 else float(variance)

        cov_matrix = np.cov(selected_samples, rowvar=False)
        return np.atleast_2d(cov_matrix)

    def summary(self, round_to: int | None = None) -> pd.DataFrame:
        pass

    def _get_param_positions(
        self,
        params: str | ZfitParameter | Iterable[str | ZfitParameter],
    ) -> list[int]:
        """Get parameter positions in the samples array from names or objects.

        Args:
            params: Single parameter or collection of parameters.
                   Can be parameter name(s) or object(s).

        Returns:
            List of parameter positions in the samples array.
        """
        if isinstance(params, dict):
            msg = f"Invalid parameter type: {type(params)}"
            raise TypeError(msg)

        params = convert_to_container(params)

        positions = []
        for param in params:
            if isinstance(param, str):
                if param not in self._param_by_name:
                    msg = f"Parameter '{param}' not found"
                    raise ValueError(msg)
                positions.append(self._position_by_name[param])
            elif isinstance(param, zfit.Parameter):
                if param not in self._name_by_param:
                    msg = f"Parameter {param} not found in posterior samples"
                    raise ValueError(msg)
                positions.append(self._position_by_name[param.name])
            else:
                msg = (
                    f"Invalid parameter type: {type(param)}. Expected string (parameter name) or ZfitParameter object."
                )
                raise TypeError(msg)

        return positions

    def __repr__(self) -> str:
        return f"PosteriorSamples(n_samples={len(self.samples)}, params={[param.name for param in self._params]})"

    def __str__(self) -> str:
        """Nice string representation of posterior results."""
        import colored  # noqa: PLC0415
        from colorama import Style  # noqa: PLC0415
        from tabulate import tabulate  # noqa: PLC0415

        string = Style.BRIGHT + "PosteriorSamples" + Style.NORMAL + f" from\n{self.loss} \nwith\n{self.sampler}\n\n"

        def color_on_bool(value, on_true=None, on_false=None):
            pass

        rhat_str = "N/A (single chain)"
        if self._rhat is not None:
            max_rhat = np.max(self._rhat)
            rhat_str = f"{max_rhat:.4f}"
            if max_rhat > 1.1:
                rhat_str = colored.fg("red") + rhat_str + Style.RESET_ALL

        ess_str = "N/A"
        if self._ess is not None:
            min_ess = np.min(self._ess)
            ess_str = f"{min_ess:.0f}"
            if min_ess < 100:
                ess_str = colored.fg("red") + ess_str + Style.RESET_ALL

        string += tabulate(
            [
                [
                    color_on_bool(self.valid),
                    color_on_bool(self.converged, on_true=colored.bg("green"), on_false=colored.bg("yellow")),
                    rhat_str,
                    ess_str,
                    f"{len(self.samples):>13} | {self.n_warmup:>6} | {self.n_samples:>10}",
                ]
            ],
            [
                "valid",
                "converged",
                "max R̂",
                "min ESS",
                "total samples | warmup | per walker",
            ],
            tablefmt="fancy_grid",
            disable_numparse=True,
            colalign=["center", "center", "center", "center", "right"],
        )

        string += "\n\n" + Style.BRIGHT + "Parameters\n" + Style.NORMAL

        param_data = []

        means = self.mean()
        stds = self.std()
        lower, upper = self.credible_interval(alpha=0.05)

        all_ci_values = []
        for i in range(len(self._params)):
            ci_lower = lower[i] if hasattr(lower, "__len__") else lower
            ci_upper = upper[i] if hasattr(upper, "__len__") else upper
            all_ci_values.extend([ci_lower, ci_upper])

        max_abs_ci = max(abs(val) for val in all_ci_values)
        if max_abs_ci >= 1000:
            ci_width = 10  # For large numbers like n_sig, n_bkg
        elif max_abs_ci >= 10:
            ci_width = 8  # For moderate numbers
        else:
            ci_width = 7  # For small numbers like mu, sigma

        for i, param in enumerate(self._params):
            param_name = param.name
            mean_val = means[i]
            std_val = stds[i]
            ci_lower = lower[i] if hasattr(lower, "__len__") else lower
            ci_upper = upper[i] if hasattr(upper, "__len__") else upper

            rhat_param = f"{self._rhat[i]:.3f}" if self._rhat is not None else "N/A"
            ess_param = f"{self._ess[i]:.0f}" if self._ess is not None else "N/A"

            ci_formatted = f"[{ci_lower:>{ci_width}.4f}, {ci_upper:>{ci_width}.4f}]"

            param_data.append(
                [
                    param_name,
                    f"{mean_val:>8.4f}",
                    f"± {std_val:.4f}",
                    ci_formatted,
                    rhat_param,
                    ess_param,
                ]
            )

        string += tabulate(
            param_data,
            headers=["parameter", "mean", "std", "95% CI", "R̂", "ESS"],
            tablefmt="simple",
            floatfmt=".4f",
            colalign=["left", "right", "right", "right", "right", "right"],
        )

        if hasattr(self._sampler, "nwalkers"):
            string += f"\n\nSampler: {self._sampler.__class__.__name__} with {self._sampler.nwalkers} walkers"

        return string

    def _repr_pretty_(self, p, cycle):
        pass

    def _compute_convergence_diagnostics(self):
        """Compute MCMC convergence diagnostics."""
        self._valid = not (np.any(np.isnan(self.samples)) or np.any(np.isinf(self.samples)))

        if not self._valid:
            self._converged = False
            self._rhat = None
            self._ess = None
            return

        import arviz as az  # noqa: PLC0415

        idata = self.to_arviz()

        rhat_data = az.rhat(idata)
        ess_data = az.ess(idata)

        self._rhat = np.array([rhat_data[param.name].values for param in self._params])
        self._ess = np.array([ess_data[param.name].values for param in self._params])

        rhat_converged = bool(np.all(self._rhat < 1.1))
        ess_converged = bool(np.all(self._ess > 100))
        self._converged = rhat_converged and ess_converged

    @property
    def rhat(self) -> npt.NDArray[np.float64] | None:
        """Gelman-Rubin R-hat convergence diagnostic.

        Values < 1.1 indicate good convergence.
        Only available when multiple chains are used.
        """
        return self._rhat

    @property
    def ess(self) -> npt.NDArray[np.float64] | None:
        """Effective sample size for each parameter.

        Accounts for autocorrelation in MCMC chains.
        Higher values indicate more independent samples.
        """
        return self._ess

    def convergence_summary(self) -> dict:
        pass

    def diagnostics(self) -> dict:
        pass
