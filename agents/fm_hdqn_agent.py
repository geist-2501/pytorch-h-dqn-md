import numpy as np

from agents.h_dqn_agent import HDQNAgent, DictObsType, FlatObsType, ActType
from agents.utils import flatten


class FactoryMachinesHDQNAgent(HDQNAgent):
    """
    HDQN agent for the FactoryMachinesMultiEnv.
    """

    up, left, down, right, grab = range(5)

    action_names = ["up", "left", "down", "right", "grab"]

    def __init__(self, obs: DictObsType, n_actions: int, device: str = 'cpu') -> None:
        # Depot locations is a flattened x,y list, so half it's size for the total number of locations,
        # and add one for the output location.
        n_goals = (len(obs["depot_locs"]) // 2) + 1
        super().__init__(obs, n_goals, n_actions, device=device)

    def get_intrinsic_reward(self, obs: DictObsType, action: ActType, next_obs: DictObsType, goal: ActType) -> float:
        reward = 0
        if self.goal_satisfied(obs, action, next_obs, goal):
            reward += 5
        elif action == self.grab:
            reward += -1

        if self._did_collide(obs["agent_obs"].reshape((3, 3)), action):
            reward += -1

        return reward - 0.01

    def goal_satisfied(self, obs: DictObsType, action: ActType, next_obs, goal: ActType) -> bool:
        if goal == self.n_goals - 1:
            # Last goal is the output depot.
            return np.array_equal(obs["output_loc"], [0, 0]) \
                or np.array_equal(next_obs["output_loc"], [0, 0])
        else:
            offset = goal * 2
            goal_depot_loc = obs["depot_locs"][offset:offset + 2]
            return np.array_equal(goal_depot_loc, [0, 0]) and action == self.grab

    def to_q1(self, obs: DictObsType, goal: ActType) -> FlatObsType:
        return super().to_q1(flatten(obs), goal)

    def to_q2(self, obs: DictObsType) -> FlatObsType:
        return flatten(obs)

    def goal_to_str(self, goal: ActType) -> str:
        if goal == self.n_goals - 1:
            return "OUT"
        else:
            return f"D{goal}"

    def action_to_str(self, action: ActType) -> str:
        return self.action_names[action]

    def _did_collide(self, local_obs, action) -> bool:
        return (action == self.up and local_obs[0, 1] == 1) \
            or (action == self.down and local_obs[2, 1] == 1) \
            or (action == self.right and local_obs[1, 2] == 1) \
            or (action == self.left and local_obs[1, 0] == 1)
