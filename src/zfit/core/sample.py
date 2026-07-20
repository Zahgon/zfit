
from __future__ import annotations

import typing
from collections.abc import Callable, Iterable
from contextlib import suppress

import tensorflow as tf
from tensorflow_probability import distributions as tfd

import zfit.z.numpy as znp
from zfit._interfaces import ZfitPDF, ZfitSpace

from .. import settings, z
from ..settings import run, ztypes
from ..util import ztyping
from ..util.container import convert_to_container
from ..util.exception import WorkInProgressError
from .data import Data
from .space import Space

if typing.TYPE_CHECKING:
    import zfit  # noqa: F401


class UniformSampleAndWeights:
    def __call__(self, n_to_produce: int | tf.Tensor, limits: Space, dtype, prng=None):
        if dtype is not None and dtype != ztypes.float:
            msg = "Only float is supported for now."
            raise ValueError(msg)

        if prng is None:
            prng = z.random.get_prng()
        rnd_samples = []
        thresholds_unscaled_list = []
        weights = tf.broadcast_to(z.constant(1.0, shape=(1,)), shape=(n_to_produce,))
        n_produced = tf.constant(0, tf.int64)
        space: Space
        for i, space in enumerate(limits):
            lower, upper = space.v0.limits  # TODO: remove new space
            if i == len(limits) - 1:
                n_partial_to_produce = n_to_produce - n_produced  # to prevent roundoff errors, shortcut for 1 space
            else:
                if isinstance(space, EventSpace):
                    frac = 1.0  # TODO(Mayou36): remove hack for Eventspace
                else:
                    tot_area = limits.volume
                    frac = (space.volume / tot_area)[0]
                n_partial_to_produce = znp.asarray(
                    z.to_real(n_to_produce) * z.to_real(frac), dtype=tf.int64
                )  # TODO(Mayou36): split right!

            sample_drawn = prng.uniform(
                shape=(n_partial_to_produce, limits.n_obs + 1),
                dtype=ztypes.float,
            )

            rnd_sample = sample_drawn[:, :-1] * (upper - lower) + lower  # -1: all except func value
            thresholds_unscaled = sample_drawn[:, -1]

            rnd_samples.append(rnd_sample)
            thresholds_unscaled_list.append(thresholds_unscaled)
            n_produced += n_partial_to_produce

        rnd_sample = znp.concatenate(rnd_samples, axis=0)
        thresholds_unscaled = znp.concatenate(thresholds_unscaled_list, axis=0)

        n_drawn = n_to_produce
        return rnd_sample, thresholds_unscaled, weights, weights, n_drawn


class EventSpace(Space):

    def __init__(
        self,
        obs: ztyping.ObsTypeInput,
        limits: ztyping.LimitsTypeInput,
        factory=None,
        dtype=ztypes.float,
        name: str | None = "Space",
    ):
        if limits is None:
            msg = "Limits cannot be None for EventSpaces (currently)"
            raise ValueError(msg)
        self._limits_tensor = None
        self.dtype = dtype
        self._factory = factory
        super().__init__(obs, limits, name)




    def create_limits(self, n):
        if self._factory is not None:
            self._limits_tensor = self._factory(n)


    def add(self, other: ztyping.SpaceOrSpacesTypeInput):  # noqa: ARG002
        msg = "Cannot be called with an event space."
        raise RuntimeError(msg)

    def combine(self, other: ztyping.SpaceOrSpacesTypeInput):  # type: ignore[override]  # noqa: ARG002
        msg = "Cannot be called with an event space."
        raise RuntimeError(msg)


    def __hash__(self):
        return id(self)


@z.function(wraps="sample")
def accept_reject_sample(
    prob: Callable,
    n: int,
    limits: ZfitSpace,
    sample_and_weights_factory: Callable = UniformSampleAndWeights,
    dtype=ztypes.float,
    prob_max: None | int = None,
    efficiency_estimation: float = 0.5,
) -> tf.Tensor:
    """Accept reject sample from a probability distribution.

    Args:
        prob: A function taking x a Tensor as an argument and returning the probability
            (or anything that is proportional to the probability).
        n: Number of samples to produce
        limits: The limits to sample from
        sample_and_weights_factory: An (immutable!) factory function that returns the following function:
            A function that returns the sample to insert into ``prob`` and the weights
            (probability density) of each sample together with the random thresholds. The API looks as follows:

            - Parameters:

                - n_to_produce (int, tf.Tensor): The number of events to produce (not exactly).
                - limits (Space): the limits in which the samples will be.
                - dtype (dtype): DType of the output.

            - Return:
                A tuple of length 5:
                - proposed sample (tf.Tensor with shape=(n_to_produce, n_obs)): The new (proposed) sample
                    whose values are inside ``limits``.
                - thresholds_unscaled (tf.Tensor with shape=(n_to_produce,): Uniformly distributed
                    random values **between 0 and 1**.
                - weights (tf.Tensor with shape=(n_to_produce)): (Proportional to the) probability
                    for each sample of the distribution it was drawn from.
                - weights_max (int, tf.Tensor, None): The maximum of the weights (if known). This is
                    what the probability maximum will be scaled with, so it should be rather lower than the maximum
                    if the peaks do not exactly coincide. Otherwise return None (which will **assume**
                    that the peaks coincide).
                - n_produced: the number of events produced. Can deviate from the requested number.

        dtype:
        prob_max: The maximum of the model function for the given limits. If None
            is given, it will be automatically, safely estimated (by a 10% increase in computation time
            (constant weak scaling)).
        efficiency_estimation: estimation of the initial sampling efficiency.

    Returns:
    """
    prob_max_init = prob_max
    multiple_limits = len(limits) > 1
    if prob_max is None:
        n_min_to_produce = 3000
        overestimate_factor_scaling = 1.25
    else:  # if an exact estimation is given
        n_min_to_produce = 0
        overestimate_factor_scaling = 1.001

    sample_and_weights = sample_and_weights_factory()
    n = znp.asarray(n, dtype=tf.int64)
    z.assert_non_negative(n)

    dynamic_array_shape = True

    initial_is_sampled = tf.constant("EMPTY")
    if (isinstance(limits, EventSpace) and not limits.is_generator) or limits.n_events > 1:
        dynamic_array_shape = False
        z.assert_equal(znp.asarray(limits.n_events, dtype=tf.int64), n)

        initial_is_sampled = tf.fill(value=False, dims=(n,))
        efficiency_estimation = 1.0  # generate exactly n

    inital_n_produced = tf.constant(0, dtype=tf.int64)
    initial_n_drawn = tf.constant(0, dtype=tf.int64)

    sample = tf.zeros(shape=(n, limits.n_obs), dtype=dtype)



    efficiency_estimation = z.to_real(efficiency_estimation)
    weights_scaling = z.constant(0.0)
    weights_maximum = z.constant(0.0)
    prob_maximum = z.constant(0.0)
    n_min_to_produce = znp.asarray(n_min_to_produce, dtype=tf.int64)
    inital_n_produced = znp.asarray(inital_n_produced, dtype=tf.int64)
    initial_n_drawn = znp.asarray(initial_n_drawn, dtype=tf.int64)

    loop_vars = (
        n,
        sample,
        inital_n_produced,
        initial_n_drawn,
        efficiency_estimation,
        initial_is_sampled,
        weights_scaling,
        weights_maximum,
        prob_maximum,
        n_min_to_produce,
    )

    is_sampled_shape = tf.TensorShape([None])

    shape_invariants = (
        n.get_shape(),  # n - fixed
        sample.get_shape(),  # sample - fixed
        inital_n_produced.get_shape(),  # n_produced - fixed
        initial_n_drawn.get_shape(),  # n_total_drawn - fixed
        efficiency_estimation.get_shape(),  # eff - fixed
        is_sampled_shape,  # is_sampled
        weights_scaling.get_shape(),  # weights_scaling - fixed
        weights_maximum.get_shape(),  # weights_maximum - fixed
        prob_maximum.get_shape(),  # prob_maximum - fixed
        n_min_to_produce.get_shape(),  # n_min_to_produce - fixed
    )

    loop_result = tf.while_loop(
        cond=not_enough_produced,
        body=sample_body,  # paraopt
        loop_vars=loop_vars,
        shape_invariants=shape_invariants,
        swap_memory=True,
        parallel_iterations=1,
    )

    new_sample = loop_result[1]  # the sample buffer

    if multiple_limits:
        new_sample = z.random.shuffle(new_sample)  # to make sure, randomly remove and not biased.
    if dynamic_array_shape:  # if not dynamic we produced exact n -> no need to cut
        new_sample = new_sample[:n, :]  # cutting away to many produced
    new_sample = tf.stop_gradient(new_sample)  # stopping backprop

    if run.executing_eagerly():
        n_dims = limits.n_obs
        with suppress(AttributeError):  # if n_samples_int is not a numpy object
            new_sample.set_shape((int(n), n_dims))
    return new_sample


def extract_extended_pdfs(pdfs: Iterable[ZfitPDF] | ZfitPDF) -> list[ZfitPDF]:
    """Return all extended pdfs that are daughters.

    Args:
        pdfs:

    Returns:
        List[pdfs]:
    """
    from ..models.functor import BaseFunctor  # noqa: PLC0415

    pdfs = convert_to_container(pdfs)
    indep_pdfs = []

    for pdf in pdfs:
        if not pdf.is_extended:
            continue
        if isinstance(pdf, BaseFunctor):
            if all(pdf.pdfs_extended):
                indep_pdfs.extend(extract_extended_pdfs(pdfs=pdf.pdfs))
            elif not any(pdf.pdfs_extended):
                indep_pdfs.append(pdf)
            else:
                msg = "Should not reach this point, wrong assumptions. Please report bug."
                raise AssertionError(msg)
        else:  # extended, but not a functor
            indep_pdfs.append(pdf)

    return indep_pdfs


def extended_sampling(pdfs: Iterable[ZfitPDF] | ZfitPDF, limits: Space) -> tf.Tensor:
    """Create a sample from extended pdfs by sampling from a Poisson using the yield.

    Args:
        pdfs:
        limits:

    Returns:
        Union[tf.Tensor]:
    """
    samples = []
    pdfs = convert_to_container(pdfs)
    pdfs = extract_extended_pdfs(pdfs)

    for pdf in pdfs:
        n = z.random.poisson(lam=pdf.get_yield(), shape=(), dtype=ztypes.float)
        n = znp.asarray(n, dtype=tf.int64)
        sample = pdf.sample(limits=limits, n=n)
        samples.append(sample.value())

    return znp.concatenate(samples, axis=0)
