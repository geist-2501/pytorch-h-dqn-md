from crete import register_agent

from agents.dqn_agent import DQNAgent, dqn_training_wrapper, dqn_graphing_wrapper
from agents.fm_hdqn_agent import FactoryMachinesHDQNAgent
from agents.fm_masked_hdqn_agent import FactoryMachinesMaskedHDQNAgent
from agents.h_dqn_agent import hdqn_training_wrapper, hdqn_graphing_wrapper

register_agent(
    agent_id="DQN",
    agent_factory=lambda obs, n_actions, device: DQNAgent(obs, n_actions, device=device),
    graphing_wrapper=dqn_graphing_wrapper,
    training_wrapper=dqn_training_wrapper
)

register_agent(
    agent_id="FM-HDQN",
    agent_factory=lambda obs, n_actions, device: FactoryMachinesHDQNAgent(obs, n_actions, device),
    training_wrapper=hdqn_training_wrapper,
    graphing_wrapper=hdqn_graphing_wrapper
)

register_agent(
    agent_id="FM-HDQN-masked",
    agent_factory=lambda obs, n_actions, device: FactoryMachinesMaskedHDQNAgent(obs, n_actions, device),
    training_wrapper=hdqn_training_wrapper,
    graphing_wrapper=hdqn_graphing_wrapper
)