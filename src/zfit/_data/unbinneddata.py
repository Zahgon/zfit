from __future__ import annotations

import typing

import tensorflow_probability as tfp
from zfit_interface.data import ZfitData

if typing.TYPE_CHECKING:
    import zfit  # noqa: F401


@tfp.experimental.auto_composite_tensor()
class UnbinnedData(tfp.experimental.AutoCompositeTensor, ZfitData):
    def __init__(self, data, space=None, weights=None):
        self._data = data
        self._space = space
        self._weights = weights






    def values(self):
        return self.data

    def __getitem__(self, item):
        if not isinstance(item, str):
            return super().__getitem__(item)
        for index, axis in enumerate(self.axes):  # noqa: B007
            if axis.name == item:
                break
        else:
            msg = f"{item} not in {self.axes}"
            raise KeyError(msg)
        return self.data[..., index]



