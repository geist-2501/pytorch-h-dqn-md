from agents import FactoryMachinesHDQNAgent
from agents.h_dqn_agent import DictObsType, FlatObsType, ActType


class FactoryMachinesMaskedHDQNAgent(FactoryMachinesHDQNAgent):
    """
    Extension of the HDQN agent for FM-Multi. Uses observation masking to improve scalability.
    """
    def to_q1(self, obs: DictObsType, goal: ActType) -> FlatObsType:
        offset = goal * 2
        goal_depot_loc = [*obs["depot_locs"], *obs["output_loc"]][offset:offset + 2]
        return [
            *obs["agent_loc"],
            *obs["agent_obs"],
            *goal_depot_loc,
            *self.onehot(goal, self.n_goals)
        ]

    def to_q2(self, obs: DictObsType) -> FlatObsType:
        return [
            *obs["agent_loc"],
            *obs["agent_inv"],
            *obs["depot_queues"],
            *obs["depot_ages"],
        ]