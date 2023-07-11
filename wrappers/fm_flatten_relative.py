import gymnasium as gym
from .util import make_relative


class FactoryMachinesFlattenRelativeWrapper(gym.ObservationWrapper):
    """
    Turns the dict observation from the FactoryMachinesEnv into a flat 1D list.
    Makes all locations relative.
    """
    def observation(self, obs):
        return [
            *obs["agent_loc"],
            *obs["agent_obs"],
            *obs["agent_inv"],
            *make_relative(obs["agent_loc"], obs["depot_locs"]),
            *obs["depot_queues"],
            *make_relative(obs["agent_loc"], obs["output_loc"]),
            *obs["depot_ages"]
        ]
