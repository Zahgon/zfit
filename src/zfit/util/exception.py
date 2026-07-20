
from __future__ import annotations

import typing
import warnings

if typing.TYPE_CHECKING:
    import zfit  # noqa: F401


class PDFCompatibilityError(Exception):
    pass


class LogicalUndefinedOperationError(Exception):
    pass


class OperationNotAllowedError(Exception):
    pass


class ExtendedPDFError(Exception):
    pass


class AlreadyExtendedPDFError(ExtendedPDFError):
    pass


class NotExtendedPDFError(ExtendedPDFError):
    pass


class ConversionError(Exception):
    pass


class SubclassingError(Exception):
    pass


class BasePDFSubclassingError(SubclassingError):
    pass


class MinimizerSubclassingError(SubclassingError):
    pass


class IntentionAmbiguousError(Exception):
    pass


class UnderdefinedError(IntentionAmbiguousError):
    pass


class LimitsUnderdefinedError(UnderdefinedError):
    pass


class NormRangeUnderdefinedError(UnderdefinedError):
    pass


class OverdefinedError(IntentionAmbiguousError):
    pass


class LimitsOverdefinedError(OverdefinedError):
    pass


class CoordinatesUnderdefinedError(UnderdefinedError):
    pass


class AxesAmbiguousError(IntentionAmbiguousError):
    pass


class NotSpecifiedError(Exception):
    pass


class LimitsNotSpecifiedError(NotSpecifiedError):
    pass


class NormRangeNotSpecifiedError(NotSpecifiedError):
    pass


class AxesNotSpecifiedError(NotSpecifiedError):
    pass


class ObsNotSpecifiedError(NotSpecifiedError):
    pass


class ParamNameNotUniqueError(Exception):
    pass


class IncompatibleError(Exception):
    pass


class ShapeIncompatibleError(IncompatibleError):
    pass


class CoordinatesIncompatibleError(IncompatibleError):
    pass


class ObsIncompatibleError(CoordinatesIncompatibleError):
    pass


class AxesIncompatibleError(CoordinatesIncompatibleError):
    pass


class SpaceIncompatibleError(IncompatibleError):
    pass


class LimitsIncompatibleError(IncompatibleError):
    pass


class NumberOfEventsIncompatibleError(ShapeIncompatibleError):
    pass


class InvalidLimitSubspaceError(Exception):
    pass


class ModelIncompatibleError(IncompatibleError):
    pass


class WeightsNotImplementedError(Exception):
    pass


class DataIsBatchedError(Exception):
    pass


class ParameterNotIndependentError(Exception):
    pass




class NotMinimizedError(Exception):
    pass




class IllegalInGraphModeError(Exception):
    pass


class CannotConvertToNumpyError(Exception):
    pass


class ZfitNotImplementedError(NotImplementedError):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)
        if type(self) is ZfitNotImplementedError:
            warnings.warn(
                "Prefer to use a more specific subclass. See in `zfit.exceptions`", DeprecationWarning, stacklevel=2
            )


class FunctionNotImplemented(ZfitNotImplementedError):
    pass


class StandardControlFlow(Exception):
    pass


class SpecificFunctionNotImplemented(FunctionNotImplemented):
    pass


class MinimizeNotImplemented(FunctionNotImplemented):
    pass


class MinimizeStepNotImplemented(FunctionNotImplemented):
    pass


class AnalyticNotImplemented(ZfitNotImplementedError):
    pass


class AnalyticIntegralNotImplemented(AnalyticNotImplemented):
    pass


class AnalyticSamplingNotImplemented(AnalyticNotImplemented):
    pass


class NormNotImplemented(StandardControlFlow):
    pass


NormRangeNotImplemented = NormNotImplemented  # legacy


class MultipleLimitsNotImplemented(StandardControlFlow):
    pass


class InitNotImplemented(StandardControlFlow):
    pass


class VectorizedLimitsNotImplemented(StandardControlFlow):
    pass


class DerivativeCalculationError(ValueError):
    pass




class WorkInProgressError(Exception):
    pass


class BreakingAPIChangeError(Exception):
    def __init__(self, msg, *args: object) -> None:
        default_msg = "This item has been removed due to an API change. Instruction to update:\n"
        msg = default_msg + str(msg)
        super().__init__(msg, *args)


class BehaviorUnderDiscussion(Exception):
    def __init__(self, msg, *args: object) -> None:
        default_msg = (
            "The behavior of the following is currently under discussion and ideas are well needed. "
            "Please open an issue at https://github.com/zfit/zfit/issues with your opinion about this.\n"
            ""
        )
        msg = default_msg + str(msg)
        super().__init__(msg, *args)


class MaximumIterationReached(StandardControlFlow):
    pass


class AnalyticGradientNotAvailable(Exception):
    pass
