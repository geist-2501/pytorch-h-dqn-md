import gymnasium as gym

from .util import make_relative


class FactoryMachinesRelativeWrapper(gym.ObservationWrapper):
    """
    Makes the dict observation from a FactoryMachines env have relative locations.
    """
    def observation(self, obs):
        return {
            **obs,
            "depot_locs": make_relative(obs["agent_loc"], obs["depot_locs"]),
            "output_loc": make_relative(obs["agent_loc"], obs["output_loc"])
        }
