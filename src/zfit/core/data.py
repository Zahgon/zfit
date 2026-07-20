
from __future__ import annotations

import typing
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from typing import Literal

import numpy as np
import pandas as pd
import pydantic.v1 as pydantic
import tensorflow as tf
import uproot
import xxhash
from pydantic.v1 import Field
from tensorflow.python.types.core import TensorLike
from tensorflow.python.util.deprecation import deprecated, deprecated_args

import zfit.z.numpy as znp
from zfit._interfaces import ZfitBinnedData, ZfitSpace, ZfitUnbinnedData

from .. import z
from ..exception import OutsideLimitsError
from ..serialization import SpaceRepr
from ..serialization.serializer import BaseRepr, to_orm_init
from ..settings import run, ztypes
from ..util import ztyping
from ..util.cache import GraphCachable, invalidate_graph
from ..util.container import convert_to_container
from ..util.exception import (
    BreakingAPIChangeError,
    ObsIncompatibleError,
    ShapeIncompatibleError,
    WorkInProgressError,
)
from ..util.temporary import TemporarilySet
from .baseobject import BaseObject, convert_param_values
from .coordinates import convert_to_obs_str
from .dimension import BaseDimensional
from .serialmixin import SerializableMixin, ZfitSerializable
from .space import Space, convert_to_space

if typing.TYPE_CHECKING:
    import zfit


def convert_to_data(data, obs=None, *, check_limits: bool = False) -> Data:
    if isinstance(data, ZfitUnbinnedData):
        return data
    elif isinstance(data, LightDataset):
        return Data(data=data, obs=obs)

    if check_limits:
        if not isinstance(obs, ZfitSpace):
            msg = "If check_limits is True, obs has to be a ZfitSpace."
            raise ValueError(msg)
        data_nocut = convert_to_data(data, obs=obs.obs, check_limits=False)
        not_inside = ~obs.inside(data_nocut.value())
        if np.any(not_inside):
            msg = f"Data {data} is not inside the limits {obs}."
            raise OutsideLimitsError(msg)
    if isinstance(data, pd.DataFrame):
        return Data.from_pandas(df=data, obs=obs)
    elif isinstance(data, Mapping):
        return Data.from_mapping(mapping=data, obs=obs)

    if obs is None:
        msg = f"If data is not a Data-like object, obs has to be specified. Data is {data} and obs is {obs}."
        raise ValueError(msg)
    if isinstance(data, int | float):
        data = znp.array([data])
    if isinstance(data, Iterable):
        data = znp.array(data)
    if isinstance(data, np.ndarray):
        return Data.from_numpy(obs=obs, array=data)
    if isinstance(data, tf.Tensor | znp.ndarray | tf.Variable):
        return Data.from_tensor(obs=obs, tensor=data)

    msg = f"Cannot convert {data} to a Data object."
    raise TypeError(msg)


class DataMeta(type):
    def __call__(cls, data, obs=None, *args, **kwargs):
        """Construct an instance of a class whose metaclass is Meta."""
        assert isinstance(cls, DataMeta)
        if binned := (obs is not None and isinstance(obs, ZfitSpace) and obs.is_binned):
            binned_obs = obs
            obs = obs.with_binning(False)

        if isinstance(data, LightDataset):
            obj = cls.__new__(cls, *args, **kwargs)
            obj.__init__(data, obs=obs, **kwargs)
        elif isinstance(data, pd.DataFrame):
            obj = cls.from_pandas(data, obs=obs, **kwargs)
        elif isinstance(data, Mapping):
            obj = cls.from_mapping(data, obs=obs, **kwargs)
        elif tf.is_tensor(data):
            obj = cls.from_tensor(tensor=data, obs=obs, **kwargs)
        elif isinstance(data, np.ndarray):
            obj = cls.from_numpy(array=data, obs=obs, **kwargs)
        else:
            try:
                obj = cls.from_numpy(array=data, obs=obs, **kwargs)
            except Exception as error:
                msg = f"Cannot convert {data} to a Data object. Use an explicit constructor (`from_pandas`, `from_mapping`, `from_tensor`, `from_numpy` etc)."
                raise TypeError(msg) from error
        if binned:
            obj = obj.to_binned(binned_obs)

        return obj


class Data(
    ZfitUnbinnedData,
    BaseDimensional,
    BaseObject,
    GraphCachable,
    SerializableMixin,
    ZfitSerializable,
    metaclass=DataMeta,
):
    USE_HASH = False
    BATCH_SIZE = 1_000_000

    def __init__(
        self,
        data: LightDataset | pd.DataFrame | Mapping[str, np.ndarray] | tf.Tensor | np.ndarray | zfit.Data,
        *,
        obs: ztyping.ObsTypeInput = None,
        weights: TensorLike = None,
        name: str | None = None,
        label: str | None = None,
        dtype: tf.DType = None,
        use_hash: bool | None = None,
        guarantee_limits: bool = False,
    ):
        """Create data, a thin wrapper around an array-like structure that supports weights and limits.

        Instead of creating a `Data` object directly, the `from_*` constructors, such as `from_pandas`, `from_mapping`,
        `from_tensor`, and `from_numpy`, can be used for a more fine-grained control of some arguments and
        for more extensive documentation on the allowed arguments.

        The data is unbinned, i.e. it is a collection of events. The data can be weighted and is defined in a
        space, which is a set of observables, whose limits are enforced.

        Args:
            data: A dataset storing the actual values. A variety of data-types are possible, as long as they are
               array-like.
            obs: |@doc:data.init.obs| Space of the data.
               The space is used to define the observables and the limits of the data.
               If the :py:class:`~zfit.Space` has limits, these will be used to cut the
               data. If the data is already cut, use ``guarantee_limits`` for a possible
               performance improvement. |@docend:data.init.obs|

                Some data-types, such as `pd.DataFrame`, already have
                observables defined implicitly. If `obs` is `None`, the observables are inferred from the data.
                If the ``obs`` is binned, the unbinned data will be binned according to the binning of the ``obs``
                and a :py:class:`~zfit.data.BinnedData` will be returned.

            weights: |@doc:data.init.weights| Weights of the data.
               Has to be 1-D and match the shape of the data (nevents).
               Note that a weighted dataset may not be supported by all methods
               or need additional approximations to correct for the weights, taking
               more time. |@docend:data.init.weights|
            name: |@doc:data.init.name| Name of the data.
               This can possibly be used for future identification, with possible
               implications on the serialization and deserialization of the data.
               The name should therefore be "machine-readable" and not contain
               special characters.
               (currently not used for a special purpose)
               For a human-readable name or description, use the label. |@docend:data.init.name|
            label: |@doc:data.init.label| Human-readable name
               or label of the data for a better description, to be used with plots etc.
               Can contain arbitrary characters.
               Has no programmatical functional purpose as identification. |@docend:data.init.label|
            guarantee_limits: |@doc:data.init.guarantee_limits| Guarantee that the data is within the limits.
               If ``True``, the data will not be checked and _is assumed_ to be within the limits,
               possibly because it was already cut before. This can lead to a performance
               improvement as the data does not have to be checked. |@docend:data.init.guarantee_limits| For example, if the data is a `pd.DataFrame` and the limits
                of ``obs`` have already been enforced through a ``query`` on the DataFrame, the limits are guaranteed
                to be correct and the data will not be checked again.
                Possible speedup, should not have any effect on the result.
            dtype: DType of the data. Default is `np.float64`.
            use_hash: |@doc:data.init.use_hash| If true, store a hash for caching.
               If a PDF can cache values, this option needs to be enabled for the PDF
               to be able to cache values. |@docend:data.init.use_hash|

        Returns:
            |@doc:data.init.returns| ``zfit.Data`` or ``zfit.BinnedData``:
               A ``Data`` object containing the unbinned data
               or a ``BinnedData`` if the obs is binned. |@docend:data.init.returns|

        Raises:
            ShapeIncompatibleError: If the shape of the data is incompatible with the observables.
            ValueError: If the data is not a recognized type.
        """
        if use_hash is None:
            use_hash = self.USE_HASH
        self._use_hash = use_hash

        if dtype is None:
            dtype = ztypes.float

        super().__init__(name=name)

        self._permutation_indices_data = None
        self._next_batch = None
        self._dtype = dtype
        self._nentries = None
        self._weights = None
        self._label = label if label is not None else (name if name is not None else "Data")

        self._data_range = None
        self._set_space(obs)
        self._original_space = self.space
        self._data_range = self.space

        if not guarantee_limits:  # TODO: find correct observables
            tensormap = data._tensormap if (ismap := data._tensor is None) else data.value()
            value, weights = check_cut_data_weights(limits=self.space, data=tensormap, weights=weights)
            if ismap:
                data = LightDataset(tensormap=value, ndims=self.space.n_obs)
            else:
                data = LightDataset.from_tensor(value, ndims=self.space.n_obs)

        self._name = name
        self._hashint = None

        self.dataset = data
        self._set_weights(weights=weights)

        self._update_hash()




    @property
    def nentries(self) -> int:
        pass


    @property
    def samplesize(self) -> tf.Tensor:
        pass

    def enable_hashing(self) -> None:
        pass







    def _set_space(self, obs: Space, autofill: bool = True) -> None:
        obs = convert_to_space(obs)
        self._check_n_obs(space=obs)
        if autofill:
            obs = obs.with_autofill_axes(overwrite=True)
        self._space = obs



    def _copy(self, deep, name, overwrite_params):
        """Copy the object, overwrite params with overwrite_params."""
        del deep  # no meaning...
        newpar = {
            "obs": self.space,
            "weights": self.weights,
            "name": name,
            "data": self.dataset,
            "label": self.label,
            "dtype": self.dtype,
            "use_hash": self._use_hash,
            **overwrite_params,
        }
        newpar["guarantee_limits"] = (
            "obs" not in overwrite_params and "data" not in overwrite_params
        ) or overwrite_params.get("guarantee_limits")
        if "tensor" in overwrite_params:
            msg = "do not give tensor in copy, instead give a LightDataset."
            raise BreakingAPIChangeError(msg)

        return Data(**newpar)

    @property
    def weights(self) -> tf.Tensor | None:
        pass

    def with_weights(self, weights: ztyping.WeightsInputType) -> Data:
        pass

    @deprecated(None, "Use `with_weights` instead.")
    @invalidate_graph
    def set_weights(self, weights: ztyping.WeightsInputType):
        pass

    def _set_weights(self, weights):
        if weights is not None and not isinstance(
            weights, tf.Variable
        ):  # tf.Variable means it's changeable and we trust it
            weights = znp.asarray(weights, dtype=self.dtype)
            weights = znp.atleast_1d(weights)
            if weights.shape.ndims > 1:
                msg = f"Weights have to be 1-Dim objects, is currently {weights} with shape {weights.shape}."
                raise ShapeIncompatibleError(msg)
        self._weights = weights
        self._update_hash()
        return weights


    @classmethod
    def from_pandas(
        cls,
        df: pd.DataFrame,
        obs: ztyping.ObsTypeInput = None,
        *,
        weights: ztyping.WeightsInputType | str = None,
        name: str | None = None,
        label: str | None = None,
        dtype: tf.DType = None,
        use_hash: bool | None = None,
        guarantee_limits: bool = False,
    ) -> Data | ZfitBinnedData:
        """Create a ``Data`` from a pandas DataFrame. If ``obs`` is ``None``, columns are used as obs.

        Args:
            df: pandas DataFrame that contains the data. If ``obs`` is ``None``, columns are used as obs. Can be
                a superset of obs.
            obs: |@doc:data.init.obs| Space of the data.
               The space is used to define the observables and the limits of the data.
               If the :py:class:`~zfit.Space` has limits, these will be used to cut the
               data. If the data is already cut, use ``guarantee_limits`` for a possible
               performance improvement. |@docend:data.init.obs|
                If ``None``, columns are used as obs.

            weights: |@doc:data.init.weights| Weights of the data.
               Has to be 1-D and match the shape of the data (nevents).
               Note that a weighted dataset may not be supported by all methods
               or need additional approximations to correct for the weights, taking
               more time. |@docend:data.init.weights|
            name: |@doc:data.init.name| Name of the data.
               This can possibly be used for future identification, with possible
               implications on the serialization and deserialization of the data.
               The name should therefore be "machine-readable" and not contain
               special characters.
               (currently not used for a special purpose)
               For a human-readable name or description, use the label. |@docend:data.init.name|
            label: |@doc:data.init.label| Human-readable name
               or label of the data for a better description, to be used with plots etc.
               Can contain arbitrary characters.
               Has no programmatical functional purpose as identification. |@docend:data.init.label|
            guarantee_limits: |@doc:data.init.guarantee_limits| Guarantee that the data is within the limits.
               If ``True``, the data will not be checked and _is assumed_ to be within the limits,
               possibly because it was already cut before. This can lead to a performance
               improvement as the data does not have to be checked. |@docend:data.init.guarantee_limits| For example, if the data is a `pd.DataFrame` and the limits
                of ``obs`` have already been enforced through a ``query`` on the DataFrame, the limits are guaranteed
                to be correct and the data will not be checked again.
                Possible speedup, should not have any effect on the result.
            dtype: DType of the data. Default is `np.float64`.
            use_hash: |@doc:data.init.use_hash| If true, store a hash for caching.
               If a PDF can cache values, this option needs to be enabled for the PDF
               to be able to cache values. |@docend:data.init.use_hash|

        Returns:
            |@doc:data.init.returns| ``zfit.Data`` or ``zfit.BinnedData``:
               A ``Data`` object containing the unbinned data
               or a ``BinnedData`` if the obs is binned. |@docend:data.init.returns|

        Raises:
            ValueError: If the observables are not in the dataframe.
        """
        weights_requested = weights is not None
        if dtype is None:
            dtype = ztypes.float
        if weights is None:
            weights = ""
        if obs is None:
            obs = list(df.columns)
        if isinstance(df, pd.Series):
            df = df.to_frame()
        obs = convert_to_space(obs)
        if not_in_df := set(obs.obs) - set(df.columns):
            msg = f"Observables {not_in_df} not in dataframe with columns {df.columns}"
            raise ValueError(msg)
        space = obs
        if isinstance(weights, str):  # it's in the df
            if weights not in df.columns:
                if weights_requested:
                    msg = f"Weights {weights} is a string and not in dataframe with columns {df.columns}"
                    raise ValueError(msg)
                weights = None
            else:
                obs = [o for o in space.obs if o != weights]
                weights = df[weights].to_numpy(dtype=np.float64)
                space = space.with_obs(obs=obs)

        not_in_df = set(space.obs) - set(df.columns)
        if not_in_df:
            msg = f"Observables {not_in_df} not in dataframe with columns {df.columns}"
            raise ValueError(msg)

        mapping = df[list(space.obs)].to_dict(orient="series")  # pandas indexes with lists, not tuples
        return Data.from_mapping(
            mapping=mapping,
            obs=space,
            weights=weights,
            name=name,
            label=label,
            dtype=dtype,
            use_hash=use_hash,
            guarantee_limits=guarantee_limits,
        )

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, ztyping.ArrayLike],
        obs: ztyping.ObsTypeInput = None,
        *,
        weights: TensorLike | None = None,
        label: str | None = None,
        name: str | None = None,
        dtype: tf.DType = None,
        use_hash: bool | None = None,
        guarantee_limits: bool | None = False,
    ) -> Data | ZfitBinnedData:
        """Create a ``Data`` from a mapping of observables to arrays.

        Args:
            mapping: A mapping from the observables to the data, with the observables as keys and the data as values.
            obs: |@doc:data.init.obs| Space of the data.
               The space is used to define the observables and the limits of the data.
               If the :py:class:`~zfit.Space` has limits, these will be used to cut the
               data. If the data is already cut, use ``guarantee_limits`` for a possible
               performance improvement. |@docend:data.init.obs|
                They will be matched to the data in the same order. Can be omitted, in which case the keys of the
                mapping are used as observables.
            weights: |@doc:data.init.weights| Weights of the data.
               Has to be 1-D and match the shape of the data (nevents).
               Note that a weighted dataset may not be supported by all methods
               or need additional approximations to correct for the weights, taking
               more time. |@docend:data.init.weights|
                Can also be a string that is a column in the dataframe. By default, look for a column ``""``, i.e.,
                an empty string.
            name: |@doc:data.init.name| Name of the data.
               This can possibly be used for future identification, with possible
               implications on the serialization and deserialization of the data.
               The name should therefore be "machine-readable" and not contain
               special characters.
               (currently not used for a special purpose)
               For a human-readable name or description, use the label. |@docend:data.init.name|
            label: |@doc:data.init.label| Human-readable name
               or label of the data for a better description, to be used with plots etc.
               Can contain arbitrary characters.
               Has no programmatical functional purpose as identification. |@docend:data.init.label|
            dtype: dtype of the data
            use_hash: |@doc:data.init.use_hash| If true, store a hash for caching.
               If a PDF can cache values, this option needs to be enabled for the PDF
               to be able to cache values. |@docend:data.init.use_hash|
            guarantee_limits: |@doc:data.init.guarantee_limits| Guarantee that the data is within the limits.
               If ``True``, the data will not be checked and _is assumed_ to be within the limits,
               possibly because it was already cut before. This can lead to a performance
               improvement as the data does not have to be checked. |@docend:data.init.guarantee_limits|

        Returns:
            |@doc:data.init.returns| ``zfit.Data`` or ``zfit.BinnedData``:
               A ``Data`` object containing the unbinned data
               or a ``BinnedData`` if the obs is binned. |@docend:data.init.returns|

        Raises:
            ValueError: If the observables are not in the mapping.
        """
        if obs is None:
            obs = tuple(mapping.keys())
        obs = convert_to_space(obs)
        if missing_obs := set(obs.obs) - set(mapping.keys()):
            msg = f"Not all observables (missing: {missing_obs}) requested ({obs}) are in the mapping: {mapping}."
            raise ValueError(msg)
        tensormap = {i: znp.asarray(mapping[obs], dtype=dtype) for i, obs in enumerate(obs.obs)}
        weights = znp.asarray(weights, dtype=dtype) if weights is not None else None
        dataset = LightDataset(tensormap=tensormap, ndims=obs.n_obs)
        return Data(  # *not* class, if subclass, keep constructor
            data=dataset,
            obs=obs,
            weights=weights,
            name=name,
            label=label,
            dtype=dtype,
            use_hash=use_hash,
            guarantee_limits=guarantee_limits,
        )

    @classmethod
    def from_root(
        cls,
        path: str,
        treepath: str,
        obs: ZfitSpace = None,
        *,
        weights: ztyping.WeightsStrInputType = None,
        obs_alias: Mapping[str, str] | None = None,
        name: str | None = None,
        label: str | None = None,
        dtype: tf.DType = None,
        root_dir_options=None,
        use_hash: bool | None = None,
        branches: list[str] | None = None,
        branches_alias: dict | None = None,
    ) -> Data:
        pass

    @classmethod
    def from_numpy(
        cls,
        obs: ztyping.ObsTypeInput,
        array: np.ndarray,
        *,
        weights: ztyping.WeightsInputType = None,
        name: str | None = None,
        label: str | None = None,
        dtype: tf.DType = None,
        use_hash=None,
        guarantee_limits: bool = False,
    ) -> Data | ZfitBinnedData:
        """Create ``Data`` from a ``np.array``.

        Args:
            obs: |@doc:data.init.obs| Space of the data.
               The space is used to define the observables and the limits of the data.
               If the :py:class:`~zfit.Space` has limits, these will be used to cut the
               data. If the data is already cut, use ``guarantee_limits`` for a possible
               performance improvement. |@docend:data.init.obs|
            array: Numpy array containing the data. Has to be of shape (nevents, nobs) or,
                if only one observable, (nevents) is also possible.
            weights: |@doc:data.init.weights| Weights of the data.
               Has to be 1-D and match the shape of the data (nevents).
               Note that a weighted dataset may not be supported by all methods
               or need additional approximations to correct for the weights, taking
               more time. |@docend:data.init.weights|
            name: |@doc:data.init.name| Name of the data.
               This can possibly be used for future identification, with possible
               implications on the serialization and deserialization of the data.
               The name should therefore be "machine-readable" and not contain
               special characters.
               (currently not used for a special purpose)
               For a human-readable name or description, use the label. |@docend:data.init.name|
            label: |@doc:data.init.label| Human-readable name
               or label of the data for a better description, to be used with plots etc.
               Can contain arbitrary characters.
               Has no programmatical functional purpose as identification. |@docend:data.init.label|
            dtype: dtype of the data.
            use_hash: |@doc:data.init.use_hash| If true, store a hash for caching.
               If a PDF can cache values, this option needs to be enabled for the PDF
               to be able to cache values. |@docend:data.init.use_hash|
            guarantee_limits: |@doc:data.init.guarantee_limits| Guarantee that the data is within the limits.
               If ``True``, the data will not be checked and _is assumed_ to be within the limits,
               possibly because it was already cut before. This can lead to a performance
               improvement as the data does not have to be checked. |@docend:data.init.guarantee_limits|

        Returns:
            |@doc:data.init.returns| ``zfit.Data`` or ``zfit.BinnedData``:
               A ``Data`` object containing the unbinned data
               or a ``BinnedData`` if the obs is binned. |@docend:data.init.returns|

        Raises:
            TypeError: If the array is not a numpy array.
        """
        if isinstance(array, float | int):
            array = np.array([array])
        if not isinstance(array, (np.ndarray)) and not (tf.is_tensor(array) and hasattr(array, "numpy")):
            msg = f"`array` has to be a `np.ndarray`. Is currently {type(array)}"
            raise TypeError(msg)
        if dtype is None:
            dtype = ztypes.float
        tensor = znp.asarray(array, dtype=dtype)
        return Data.from_tensor(  # *not* class, if subclass, keep constructor
            obs=obs,
            tensor=tensor,
            weights=weights,
            name=name,
            label=label,
            dtype=dtype,
            use_hash=use_hash,
            guarantee_limits=guarantee_limits,
        )

    @classmethod
    def from_tensor(
        cls,
        obs: ztyping.ObsTypeInput,
        tensor: tf.Tensor,
        *,
        weights: ztyping.WeightsInputType = None,
        name: str | None = None,
        label: str | None = None,
        dtype: tf.DType = None,
        use_hash=None,
        guarantee_limits: bool = False,
    ) -> Data | ZfitBinnedData:
        """Create a ``Data`` from a ``tf.Tensor``

        Args:
            obs: |@doc:data.init.obs| Space of the data.
               The space is used to define the observables and the limits of the data.
               If the :py:class:`~zfit.Space` has limits, these will be used to cut the
               data. If the data is already cut, use ``guarantee_limits`` for a possible
               performance improvement. |@docend:data.init.obs|
            tensor: Tensor containing the data. Has to be of shape (nevents, nobs) or, if only one observable,
                (nevents) is also possible.
            weights: |@doc:data.init.weights| Weights of the data.
               Has to be 1-D and match the shape of the data (nevents).
               Note that a weighted dataset may not be supported by all methods
               or need additional approximations to correct for the weights, taking
               more time. |@docend:data.init.weights|
            name: |@doc:data.init.name| Name of the data.
               This can possibly be used for future identification, with possible
               implications on the serialization and deserialization of the data.
               The name should therefore be "machine-readable" and not contain
               special characters.
               (currently not used for a special purpose)
               For a human-readable name or description, use the label. |@docend:data.init.name|
            label: |@doc:data.init.label| Human-readable name
               or label of the data for a better description, to be used with plots etc.
               Can contain arbitrary characters.
               Has no programmatical functional purpose as identification. |@docend:data.init.label|
            dtype: dtype of the data.
            use_hash: |@doc:data.init.use_hash| If true, store a hash for caching.
               If a PDF can cache values, this option needs to be enabled for the PDF
               to be able to cache values. |@docend:data.init.use_hash|
            guarantee_limits: |@doc:data.init.guarantee_limits| Guarantee that the data is within the limits.
               If ``True``, the data will not be checked and _is assumed_ to be within the limits,
               possibly because it was already cut before. This can lead to a performance
               improvement as the data does not have to be checked. |@docend:data.init.guarantee_limits|

        Returns:
            |@doc:data.init.returns| ``zfit.Data`` or ``zfit.BinnedData``:
               A ``Data`` object containing the unbinned data
               or a ``BinnedData`` if the obs is binned. |@docend:data.init.returns|

        Raises:
            TypeError: If the tensor is not a tensorflow tensor.
        """
        if dtype is None:
            dtype = ztypes.float
        tensor = znp.asarray(tensor, dtype=dtype)
        tensor = znp.atleast_1d(tensor)
        if len(tensor.shape) == 1:
            tensor = znp.expand_dims(tensor, -1)
        space = convert_to_space(obs)
        dataset = LightDataset.from_tensor(tensor, ndims=space.n_obs)

        return Data(
            data=dataset,
            obs=obs,
            name=name,
            label=label,
            weights=weights,
            dtype=dtype,
            use_hash=use_hash,
            guarantee_limits=guarantee_limits,
        )

    def _update_hash(self):
        if not run.executing_eagerly() or not self._use_hash:
            self._hashint = None
        else:
            hashval = self.dataset.calc_hash()
            if self.has_weights:
                hashval.update(np.asarray(self.weights))
            if hasattr(self, "_hashint"):
                self._hashint = hashval.intdigest() % (64**2)

            else:  # if the dataset is not yet initialized; this is allowed
                self._hashint = None

    def with_obs(self, obs: ztyping.ObsTypeInput, *, guarantee_limits: bool = False) -> Data:
        """Create a new ``Data`` with a subset of the data using the *obs*.

        Args:
            obs: Observables to return. Has to be a subset of the original observables.
            guarantee_limits: |@doc:data.init.guarantee_limits| Guarantee that the data is within the limits.
               If ``True``, the data will not be checked and _is assumed_ to be within the limits,
               possibly because it was already cut before. This can lead to a performance
               improvement as the data does not have to be checked. |@docend:data.init.guarantee_limits|
        Returns:
            ``zfit.Data``: A new ``Data`` object containing the subset of the data.
        """
        if not isinstance(obs, ZfitSpace):
            if not isinstance(obs, list | tuple):
                obs = [obs]
            if isinstance(obs[0], str):
                obs = self.space.with_obs(obs)
            elif isinstance(obs[0], int):
                obs = self.space.with_axes(obs)
            guarantee_limits = True
        elif obs == self.space.with_obs(obs):
            guarantee_limits = True
        if obs.is_binned:
            msg = "obs is binned, no implicit conversion to binned data allowed. Use `to_binned` instead."
            raise ValueError(msg)
        indices = self._get_permutation_indices(obs=obs)
        dataset = self.dataset.with_indices(indices)
        weights = self.weights
        return self.copy(obs=obs, data=dataset, weights=weights, guarantee_limits=guarantee_limits)

    def to_pandas(self, obs: ztyping.ObsTypeInput = None, weightsname: str | None = None) -> pd.DataFrame:
        pass

    def unstack_x(self, obs: ztyping.ObsTypeInput = None, always_list=None):
        """Return the unstacked data: a list of tensors or a single Tensor.

        Args:
            obs: Observables to return. If ``None``, all observables are returned. Can be a subset of the original
            always_list: If ``True``, always return a list, even if only one observable is requested.

        Returns:
            List(tf.Tensor)
        """
        if always_list is None:
            always_list = False
        nolist = (not always_list) and isinstance(obs, str)
        if obs is None:
            obs_str = self.obs
            if len(obs_str) == 1:
                nolist = True  # legacy behavior
        else:
            obs_str = convert_to_obs_str(obs)

            if missingobs := set(obs_str) - set(self.obs):
                msg = f"Observables {missingobs} not in data. Available observables: {self.obs}"
                raise ObsIncompatibleError(msg)
        values = [self.value(obs=ob) for ob in obs_str]
        if nolist:
            return values[0]
        return values

    def value(self, obs: ztyping.ObsTypeInput = None, axis: int | None = None) -> tf.Tensor:
        """Return the data as a numpy-like object in ``obs`` order.

        Args:
            obs: Observables to return. If ``None``, all observables are returned. Can be a subset of the original
                observables. If a string is given, a 1-D array is returned with shape (nevents,). If a list of strings
                or a ``zfit.Space`` is given, a 2-D array is returned with shape (nevents, nobs).
            axis: If given, the axis to return instead of the full data. If ``obs`` is a string, this has to be ``None``.

        Returns:
        """
        if axis is not None:
            if obs is not None:
                msg = "Cannot specify both `obs` and `axis`."
                raise ValueError(msg)
            indices = convert_to_container(axis, container=tuple)
            if not all(isinstance(ax, int) for ax in indices):
                msg = "All axes have to be integers."
                raise ValueError(msg)
            if not set(indices).issubset(set(self.space.axes)):
                msg = "All axes have to be in the space."
                raise ValueError(msg)
        else:
            indices = self.space.with_obs(obs=obs).axes
        out = self.dataset.value(indices)
        if isinstance(obs, str) or (axis is not None and isinstance(axis, int)):
            out = znp.squeeze(out, axis=-1)
        return out

    def numpy(self) -> np.ndarray:
        return self.to_numpy()

    @property
    def shape(self) -> tuple:
        return self.num_entries, self.n_obs

    def to_numpy(self) -> np.ndarray:
        """Return the data as a numpy array.

        Pandas DataFrame equivalent method
        Returns:
            np.ndarray: The data as a numpy array.
        """
        return self.value().numpy()


    def _get_permutation_indices(self, obs):
        obs = convert_to_obs_str(obs)
        perm_indices = self.space.axes  # if self.space.axes != no_change_indices else False
        if obs:
            if not frozenset(obs) <= frozenset(self.obs):
                msg = (
                    f"The observable(s) {frozenset(obs) - frozenset(self.obs)} are not contained in the dataset. "
                    f"Only the following are: {self.obs}"
                )
                raise ValueError(msg)
            perm_indices = self.space.get_reorder_indices(obs=obs)

        return perm_indices




    def _convert_sort_space(
        self,
        obs: ztyping.ObsTypeInput = None,
        axes: ztyping.AxesTypeInput = None,
        limits: ztyping.LimitsTypeInput = None,
    ) -> Space | None:
        """Convert the inputs (using eventually ``obs``, ``axes``) to :py:class:`~zfit.Space` and sort them according to
        own `obs`.

        Args:
            obs:
            axes:
            limits:

        Returns:
        """
        if obs is None:  # for simple limits to convert them
            obs = self.obs
        space = convert_to_space(obs=obs, axes=axes, limits=limits)

        if self.space is not None:
            space = space.with_coords(self.space, allow_subset=True)
        return space


    def to_binned(
        self,
        space: ztyping.SpaceType,
        *,
        name: str | None = None,
        label: str | None = None,
        use_hash: bool | None = None,
    ) -> ZfitBinnedData:
        """Bins the data using ``space`` and returns a ``BinnedData`` object.

        Args:
            space: The space to bin the data in.
            name: |@doc:data.init.name| Name of the data.
               This can possibly be used for future identification, with possible
               implications on the serialization and deserialization of the data.
               The name should therefore be "machine-readable" and not contain
               special characters.
               (currently not used for a special purpose)
               For a human-readable name or description, use the label. |@docend:data.init.name|
            label: |@doc:data.init.label| Human-readable name
               or label of the data for a better description, to be used with plots etc.
               Can contain arbitrary characters.
               Has no programmatical functional purpose as identification. |@docend:data.init.label|
            use_hash: |@doc:data.init.use_hash| If true, store a hash for caching.
               If a PDF can cache values, this option needs to be enabled for the PDF
               to be able to cache values. |@docend:data.init.use_hash|

        Returns:
            ``zfit.BinnedData``: A new ``BinnedData`` object containing the binned data.
        """
        from zfit._data.binneddatav1 import BinnedData  # noqa: PLC0415

        return BinnedData.from_unbinned(
            space=space,
            data=self,
            name=name or self.name,
            label=label or self.label,
            use_hash=use_hash or self._use_hash,
        )

    def __len__(self) -> int:
        return self.num_entries

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.value(axis=item)
        try:
            value = getitem_obs(self, item)
        except Exception as errorobs:
            msg = (
                f"Failed to retrieve {item} from data {self}. This can be changed behavior (since zfit 0.11): data can"
                f" no longer be accessed numpy-like but instead the 'obs' can be used, i.e. strings or spaces. This"
                f" resembles more closely the behavior of a pandas DataFrame."
            )
            raise RuntimeError(msg) from errorobs
        return value

    def __str__(self) -> str:
        return f"zfit.Data: {self.label} obs={self.obs} array={self.value()}"

    def __repr__(self) -> str:
        nevents = self.num_entries
        try:
            nevents = int(round(float(nevents), ndigits=2))
        except Exception:
            nevents = None
        return f"<zfit.Data: {self.label} obs={self.obs} shape={(nevents, self.n_obs)}>"


class DataRepr(BaseRepr):
    _implementation = Data
    _owndict = pydantic.PrivateAttr(default_factory=dict)
    hs3_type: Literal["Data"] = Field("Data", alias="type")

    data: np.ndarray
    space: SpaceRepr | list[SpaceRepr]
    name: str | None = None
    weights: np.ndarray | None = None





    @to_orm_init
    def _to_orm(self, init):
        dataset = LightDataset(znp.asarray(init.pop("data")))
        init["data"] = dataset
        init["obs"] = init.pop("space")

        spaces = init["obs"]
        space = spaces[0]
        for sp in spaces[1:]:
            space *= sp
        init["obs"] = space
        return super()._to_orm(init)




def check_cut_datamap_weights(limits, data, weights, guarantee_limits):
    inside = None
    datanew = {}
    for ax in limits.axes:
        limit = limits.with_axes(ax)
        arr = data[ax]
        arr = znp.atleast_1d(arr)
        if not guarantee_limits and limit.has_limits:
            inside = limit.inside(arr) if inside is None else inside & limit.inside(arr)
        datanew[ax] = arr

    if inside is not None and not (run.executing_eagerly() and np.all(inside)):
        for ax, arr in datanew.items():
            datanew[ax] = arr[inside]
        if weights is not None:
            weights = weights[inside]
    return datanew, weights


def check_cut_data_weights(
    limits: ZfitSpace,
    data: TensorLike | Mapping[str, TensorLike],
    weights: TensorLike | None = None,
    guarantee_limits: bool = False,
):
    """Check and cut the data and weights according to the limits.

    Args:
        limits: Limits to cut the data to.
        data: Data to cut.
        weights: Weights to cut.
        guarantee_limits: If True, the limits are guaranteed to be correct and the data is not checked.

    Returns:
    """
    if weights is not None:
        weights = znp.atleast_1d(weights)
        if weights.shape.ndims != 1:
            msg = f"Weights have to be 1-D, not {weights.shape}."
            raise ValueError(msg)
        datashape = next(iter(data.values())).shape[0] if isinstance(data, Mapping) else data.shape[0]
        if run.executing_eagerly() and weights.shape[0] != datashape:
            msg = f"Weights have to have the same length as the data, not {weights.shape[0]} != {datashape}."
            raise ValueError(msg)

    if isinstance(data, Mapping):
        return check_cut_datamap_weights(limits=limits, data=data, weights=weights, guarantee_limits=guarantee_limits)
    else:
        data = znp.atleast_1d(data)
        if len(data.shape) == 1 and limits.n_obs == 1:
            data = data[:, None]
        if data.shape.ndims != 2:
            msg = f"Data has to be 2-D, i.e. (nevents, nobs)., not {data.shape}, with data={data}."
            raise ValueError(msg)

    if limits.has_limits and not guarantee_limits:
        inside = limits.inside(data)
        data = data[inside]
        if weights is not None:
            weights = weights[inside]
    return data, weights


class SamplerData(Data):
    _cache_counting = 0

    def __init__(
        self,
        data: LightDataset,
        *,
        sample_and_weights_func: Callable,
        sample_holder: tf.Variable,
        n: ztyping.NumericalScalarType | Callable,
        weights=None,
        weights_holder: tf.Variable | None = None,
        params: dict[zfit.Parameter, ztyping.NumericalScalarType] | None = None,
        obs: ztyping.ObsTypeInput = None,
        name: str | None = None,
        label: str | None = None,
        dtype: tf.DType = ztypes.float,
        use_hash: bool | None = None,
        guarantee_limits: bool = False,
    ):
        """Create a `SamplerData` object.

        Use constructor `from_sampler` instead.
        """
        if use_hash is not None and not use_hash:
            msg = "use_hash is required for SamplerData."
            raise ValueError(msg)
        use_hash = True
        super().__init__(
            data=data,
            obs=obs,
            name=name,
            label=label,
            weights=weights,
            dtype=dtype,
            use_hash=use_hash,
            guarantee_limits=guarantee_limits,
        )
        params = convert_param_values(params)

        self._initial_resampled = False

        self.params = params
        self._sample_holder = sample_holder
        self._weights_holder = weights_holder
        self._weights = self._weights_holder
        self._sample_and_weights_func = sample_and_weights_func
        if isinstance(n, tf.Variable):
            msg = "Using a tf.Variable as `n` is not supported anymore. Use a numerical value or a callable instead."
            raise BreakingAPIChangeError(msg)
        self.n = n
        self._n_holder = n
        self._hashint_holder = tf.Variable(0, dtype=tf.int64, trainable=False)
        self.update_data(data.value(), weights=weights)  # to be used for precompilations etc
        self._sampler_guarantee_limits = guarantee_limits




    def _update_hash(self):
        if not run.executing_eagerly() or not self._using_hash:
            self._hashint = None
            return
        super()._update_hash()
        if hasattr(self, "_hashint_holder"):
            self._hashint_holder.assign(self._hashint % (64**2))



    @classmethod
    def get_cache_counting(cls):
        counting = cls._cache_counting
        cls._cache_counting += 1
        return counting


    @classmethod
    @deprecated_args(None, "Use `params` instead.", "fixed_params")
    def from_sampler(
        cls,
        *,
        sample_func: Callable | None = None,
        sample_and_weights_func: Callable | None = None,
        n: ztyping.NumericalScalarType,
        obs: ztyping.ObsTypeInput,
        params: ztyping.ParamValuesMap = None,
        fixed_params=None,
        name: str | None = None,
        label: str | None = None,
        dtype=None,
        use_hash: bool | None = None,
        guarantee_limits: bool = False,
    ):
        pass

    def update_data(
        self, sample: TensorLike | ZfitUnbinnedData, weights: TensorLike | None = None, guarantee_limits: bool = False
    ):
        """Load a new sample into the dataset, presumably similar to the previous one.

        Args:
            sample: The new sample to load. Has to have the same number of observables as the `obs` of the `SamplerData` but
                can have a different number of events. When a `ZfitUnbinnedData` is given, the weights from it are used.
            weights: The weights of the new sample. If `None`, the weights are not changed. If the `SamplerData` was
                initialized with weights, this has to be given. If the `SamplerData` was initialized without weights,
                this cannot be given.
            guarantee_limits: |@doc:data.init.guarantee_limits| Guarantee that the data is within the limits.
               If ``True``, the data will not be checked and _is assumed_ to be within the limits,
               possibly because it was already cut before. This can lead to a performance
               improvement as the data does not have to be checked. |@docend:data.init.guarantee_limits|
        """
        if isinstance(sample, ZfitUnbinnedData):
            if weights is not None:
                msg = "Cannot set weights if `sample` is an `UnbinnedData` object. Create a weighted dataset (i.e. use `with_weights`)."
                raise ValueError(msg)
            sample = sample.with_obs(self.space)
            weights = sample.weights
            sample = sample.value()

        sample = znp.asarray(sample, dtype=self.dtype)

        if sample.shape.rank == 1:
            sample = sample[:, None]
        elif sample.shape.rank != 2:
            msg = f"Sample has to have 1 or 2 dimensions, got {sample.shape.rank}."
            raise ValueError(msg)
        if sample.shape[-1] != self.space.n_obs:
            msg = (
                f"Sample has to have the same number of observables as the `obs` of the `SamplerData`. "
                f"Got {sample.shape[-1]} observables, expected {self.space.n_obs}."
            )
            raise ValueError(msg)
        if not guarantee_limits:
            sample, weights = check_cut_data_weights(limits=self.space, data=sample, weights=weights)
        self._sample_holder.assign(sample, read_value=False)
        if weights is not None:
            if self._weights_holder is None:
                msg = "Cannot set weights if no weights were given at initialization."
                raise ValueError(msg)
            weights = znp.asarray(weights, dtype=ztypes.float)
            self._weights_holder.assign(weights, read_value=False)
        elif self._weights_holder is not None:
            msg = "No weights given but weights_holder was initialized."
            raise ValueError(msg)

        self._n_holder = tf.shape(sample)[0]
        self._initial_resampled = True
        self._update_hash()

    @deprecated_args(None, "Use `params` instead.", "param_values")
    def resample(
        self,
        params: ztyping.ParamValuesMap = None,
        *,
        n: TensorLike = None,
        param_values: ztyping.ParamValuesMap = None,
    ):
        pass

    def __str__(self) -> str:
        return f"<SamplerData: {self.label} obs={self.obs} size={int(self.num_entries)} weighted={self.has_weights} array={self.value()}>"

    @classmethod
    def get_repr(cls):  # acts as data object once serialized
        return DataRepr


def concat(
    datasets: Iterable[Data],
    *,
    obs: ztyping.ObsTypeInput = None,
    axis: int | str | None = None,
    name: str | None = None,
    label: str | None = None,
    use_hash: bool | None = None,
) -> Data:
    """Concatenate multiple `Data` objects into a single one.

    Args:
        datasets: The `Data` objects to concatenate.
        obs: The observables to use. If ``None``, the observables of the first ``Data`` object are used. They have the same
            function as on a single ``Data`` object.
        axis: The axis along which to concatenate the data. If `None`, the data is concatenated along the first axis.
            Possible options are `0/index` or `1/obs`. If `obs`, the data is concatenated along the observable axis.
        name: The name of the new `Data` object. |@doc:data.init.name| Name of the data.
               This can possibly be used for future identification, with possible
               implications on the serialization and deserialization of the data.
               The name should therefore be "machine-readable" and not contain
               special characters.
               (currently not used for a special purpose)
               For a human-readable name or description, use the label. |@docend:data.init.name|
        label: The label of the new `Data` object. |@doc:data.init.label| Human-readable name
               or label of the data for a better description, to be used with plots etc.
               Can contain arbitrary characters.
               Has no programmatical functional purpose as identification. |@docend:data.init.label|
        use_hash: |@doc:data.init.use_hash| If true, store a hash for caching.
               If a PDF can cache values, this option needs to be enabled for the PDF
               to be able to cache values. |@docend:data.init.use_hash|


    Returns:
        A new `Data` object containing the concatenated data.

    Raises:
        tf.errors.InvalidArgumentError: If the number of events in the datasets is not equal.
        ObsIncompatibleError: If the observables are not unique or not the same in all datasets for merging along the observable axis.
    """
    if axis is None or axis in (0, "index"):
        axis = 0
    elif axis in (1, "obs", "columns"):
        axis = 1
    else:
        msg = f"Invalid axis {axis}. Valid options are 0/index or 1/obs."
        raise ValueError(msg)

    datasets = convert_to_container(datasets, container=tuple)
    if len(datasets) == 0:
        msg = "No `Data` objects given to concatenate."
        raise ValueError(msg)

    if axis == 0:
        return concat_data_index(datasets=datasets, obs=obs, name=name, label=label, use_hash=use_hash)
    else:
        return concat_data_obs(datasets=datasets, obs=obs, name=name, label=label, use_hash=use_hash)


def concat_data_obs(datasets, obs, name, label, use_hash):
    all_obs = [ob for data in datasets for ob in data.obs]
    obscounter = Counter(all_obs)
    if any(count > 1 for count in obscounter.values()):
        msg = "Observables have to be unique in the concatenated data."
        raise ObsIncompatibleError(msg)
    space = None
    if obs is not None:
        space = convert_to_space(obs)
        if set(space.obs) != (set_allobs := set(all_obs)):
            msg = f"The given observables ({space.obs}) have to be the same as the observables in the data ({set_allobs})."
            raise ObsIncompatibleError(msg)

    weights_new = []
    new_spaces = None
    nevents = []
    data_new = {} if (use_tensormap := all(data.dataset._tensor is None for data in datasets)) else []

    for data in datasets:
        if use_tensormap:
            value = {ob: data.value(ob) for ob in data.obs}
            data_new.update(value)
            nevents.extend([tf.shape(val) for val in value.values()])
        else:
            value = data.value()
            nevents.append(tf.shape(value)[0])
        data_new.append(value)
        if new_spaces is None:
            new_spaces = data.space
        else:
            new_spaces *= data.space

        if data.has_weights:
            weights_new.append(data.weights)

    z.assert_equal(
        tf.reduce_all(tf.equal(nevents, nevents[0])),
        True,
        message=f"Number of events in the datasets {datasets} have to be equal.",
    )
    newweights = znp.prod(weights_new, axis=0) if weights_new else None
    if use_tensormap:
        Data.from_mapping(data_new, obs=space, weights=newweights, name=name, label=label, use_hash=use_hash)
    else:
        newval = znp.concatenate(data_new, axis=-1)
        data = Data.from_tensor(
            tensor=newval,
            obs=new_spaces,
            weights=newweights,
            name=name,
            label=label,
            use_hash=use_hash,
            guarantee_limits=True,
        )
        if space is not None:
            data = data.with_obs(space)
    return data


def concat_data_index(datasets, obs, name, label, use_hash):
    if obs is None:
        space = datasets[0].space
        obs = space.obs
    else:
        if not isinstance((space := obs), ZfitSpace):
            space = datasets[0].space.with_obs(obs)
        obs = space.obs

    if no_obs := [data for data in datasets if data.space.obs is None]:
        msg = f"Data objects {no_obs} have no observables."
        raise ValueError(msg)
    if not all(set(obs) == set(data.obs) for data in datasets):
        msg = "All `Data` objects have to have the same observables."
        raise ValueError(msg)
    weighted = any(data.has_weights for data in datasets)

    if obs is None:
        all_space_equal = all(data.space.with_obs(obs) == space for data in datasets)
        if not all_space_equal:
            msg = "All `Data` objects have to have the same space, i.e. the same limits."
            raise ValueError(msg)

    newval = []
    if weighted:
        newweights = []
    for data in datasets:
        values = data.value(obs=space.obs)
        newval.append(values)
        if weighted:
            weights = tf.ones(tf.shape(values)[0:1]) if not data.has_weights else data.weights
            newweights.append(weights)
    newval = znp.concatenate(newval, axis=0)
    newweights = znp.concatenate(newweights, axis=0) if weighted else None

    return Data.from_tensor(
        tensor=newval, obs=space, weights=newweights, name=name, label=label, use_hash=use_hash, guarantee_limits=True
    )




class LightDataset:
    def __init__(self, tensor=None, tensormap=None, ndims=None):
        """A light-weight dataset that can be used for sampling and is aware of the mapping of the tensor with axes.

        Args:
            tensor: The tensor that contains the data. Has to be 2-D.
            tensormap: A mapping from the axes of the tensor to the actual axes in the data. If `None`, the tensor is
                assumed to be the data.
            ndims: The number of dimensions of the data. If `None`, it is inferred from the tensor or the tensormap.
        """
        if isinstance(tensormap, Mapping):
            if tensor is None:
                tensormapnew = tensormap.copy()
                for key, value in tensormap.items():
                    if value.dtype not in (ztypes.float, znp.float32, znp.float64, znp.int32, znp.int64):
                        msg = f"Value of tensormap has to be a float, not {value.dtype}."
                        raise TypeError(msg)
                    if value.dtype != ztypes.float:
                        tensormapnew[key] = znp.asarray(value, dtype=ztypes.float)
                tensormap = tensormapnew
            elif (tensorshape := tensor.shape[-1]) != len(tensormap):  # we need to reduce the dimensions
                if tensorshape < len(tensormap):
                    msg = "More dimensions requested than available in data"
                    raise ValueError(msg)
                tensormap = {k: znp.asarray(tensor[..., v], dtype=ztypes.float) for k, v in tensormap.items()}
                tensor = None

        elif tensormap is None:  # the actual preprocessing, otherwise we pass it through
            if not isinstance(tensor, tf.Variable):
                tensor = znp.asarray(tensor)

            if tensor.shape.rank != 2:
                msg = "Only 2D tensors are allowed."
                raise ValueError(msg)
            if ndims is not None and tensor.shape[1] != ndims:
                msg = f"Second dimension has to be {ndims} but is {tensor.shape[1]}"
                raise ValueError(msg)
            ndims = tensor.shape[1]
            tensormap = {i: i for i in range(ndims)}

        if ndims is None:
            ndims = len(tensormap)
        self._tensor = tensor
        self._tensormap = tensormap
        self._ndims = ndims
        self._nevents = None




    def __iter__(self):
        yield self.value()

    @classmethod
    def from_tensor(cls, tensor, ndims):
        if run.executing_eagerly() and tensor.shape[1] != ndims:
            msg = f"Second dimension of {tensor} has to be {ndims} but is {tensor.shape[1]}"
            raise ShapeIncompatibleError(msg)
        z.assert_equal(tf.shape(tensor)[1], ndims)
        return cls(tensor=tensor, ndims=None)

    def with_indices(self, indices: int | tuple[int] | list[int]):
        """Return a new `LightDataset` with the indices reshuffled.

        Args:
            indices: The indices to reshuffle the data. Can be a single index, a list or a tuple of indices.
        """
        if isinstance(indices, int):
            indices = (indices,)
        if not isinstance(indices, list | tuple):
            msg = f"Indices have to be an int, list or tuple, not {indices}"
            raise TypeError(msg)

        tensor, tensormap = self._get_tensor_and_tensormap()

        newmap = {}
        for i, idx in enumerate(indices):  # these are either indices that we reshuffle or a mapping to the new array
            newmap[i] = tensormap[idx]
        return LightDataset(tensor=tensor, tensormap=newmap)

    def _get_tensor_and_tensormap(self, forcemap=False):
        """Get the tensor and the tensor map, if needed, convert the tensor to the tensormap.

        Args:
            forcemap: Force the conversion of the tensor to the tensormap.

        Returns:
        """
        tensormap = self._tensormap
        if (tensor := self._tensor) is not None:
            if isvar := isinstance(tensor, tf.Variable):
                tensor = znp.asarray(tensor.value())  # to make sure the variable changes won't be reflected
            if forcemap:
                tensormap = {i: tensor[:, tensormap[i]] for i in range(self.ndims)}
                tensor = None
                if not isvar:  # we don't want to destroy the variable
                    self._tensormap = tensormap
                    self._tensor = None
        return tensor, tensormap

    def value(self, index: int | tuple[int] | list[int] | None = None):
        """Return the data as a tensor or a subset of the data as a tensor.

        Args:
            index: The axes to return. If `None`, the full tensor is returned. If an integer, a single axis is returned.

        Returns:
        """
        forcemap = False
        trivial_index = tuple(range(self.ndims))
        if index is None:
            index = trivial_index
        else:  # convert tensor to tensormap, if needed
            if not isinstance(index, int | tuple | list):
                msg = f"Index has to be an integer or a tuple/list of integers, not {index}"
                raise TypeError(msg)
            forcemap = len(set(index)) < self.ndims  # we will need a subset

        tensor, tensormap = self._get_tensor_and_tensormap(forcemap=forcemap)
        if tensor is None:
            if isinstance(index, int):
                return tensormap[index]  # todo: add case for single index in tuple?
            return znp.stack([tensormap[i] for i in index], axis=-1)
        else:
            if isint := isinstance(index, int):
                index = (index,)
            newindex = tuple([tensormap[i] for i in index])
            if newindex != trivial_index:
                tensor = znp.take(tensor, newindex, axis=-1)
            if isint:
                tensor = znp.squeeze(tensor, axis=-1)
            return tensor

    def calc_hash(self):
        """Calculate a hash of the data."""
        tensor, tensormap = self._get_tensor_and_tensormap(forcemap=False)
        hashval = xxhash.xxh128()
        for dim in range(self.ndims):
            index_or_array = tensormap[dim]
            if tensor is not None:
                index_or_array = tensor[:, index_or_array]
            hashval.update(index_or_array)

        return hashval

    def __hash__(self):
        return self.calc_hash().intdigest()

    def __eq__(self, other):
        if not isinstance(other, LightDataset):
            return False
        return self.calc_hash() == other.calc_hash()


def sum_samples(
    sample1: ZfitUnbinnedData,
    sample2: ZfitUnbinnedData,
    obs: ztyping.ObsTypeInput = None,
    weights: ztyping.WeightsInputType = None,
    shuffle: bool = False,
):
    pass


class Sampler(SamplerData):
    def __init__(self, *args, **kwargs):  # noqa: ARG002
        msg = "The class `Sampler` has been renamed to `SamplerData`."
        raise BreakingAPIChangeError(msg)
