from crete import register_env

from .factory_machines_env import FactoryMachinesEnv

register_env(
    "DiscreteStochasticMDP-v0",
    "envs:DiscreteStochasticMDP"
)

register_env(
    "FactoryMachinesEnv-v0",
    "envs:FactoryMachinesEnv"
)
