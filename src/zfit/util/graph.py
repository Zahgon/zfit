
from __future__ import annotations

import typing

from .temporary import TemporarilySet

if typing.TYPE_CHECKING:
    import zfit  # noqa: F401


class JIT:


    def _update_allowed(self, update_jit):
        from zfit import z  # noqa: PLC0415

        z.zextension.FunctionWrapperRegistry.do_jit_types.update(update_jit)

    def _get_allowed(self):
        from zfit import z  # noqa: PLC0415

        return z.zextension.FunctionWrapperRegistry.do_jit_types



jit = JIT()  # singleton
