from crete import register_wrapper

from .fm_flatten_relative import FactoryMachinesFlattenRelativeWrapper
from .fm_relative import FactoryMachinesRelativeWrapper

register_wrapper(
    "FlattenRelative",
    wrapper_factory=lambda outer: FactoryMachinesFlattenRelativeWrapper(outer)
)

register_wrapper(
    "Flatten",
    wrapper_factory=lambda outer: FactoryMachinesRelativeWrapper(outer)
)
