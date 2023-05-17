from operator import itemgetter
from typing import Callable, Dict, Tuple, Any, List, Optional

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import trange, tqdm
from crete import Agent, ProfileConfig, get_cli_state

from agents.dqn import DQN, Loss, loss_factory
from agents.replay_buffer import ReplayBuffer
from agents.utils import StaticLinearDecay, smoothen, can_graph, evaluate, label_values


class DQNAgent(Agent):
    """
    Deep Q-Network agent.
    Uses the technique described in Minh et al. in 'Playing Atari with Deep Reinforcement Learning'.
    """

    action_labels = ["up", "left", "down", "right", "grab"]  # This is a nasty hack.

    def __init__(self, obs, n_actions, device='cpu'):
        super().__init__("DQN")

        self.debug_mode = get_cli_state().debug_mode

        self.epsilon = 1
        self.gamma = 0.99
        self.n_actions = n_actions
        self.device = device

        # Init both networks.
        obs_size = len(obs)
        self.net = DQN(obs_size, n_actions, device=device)
        self.target_net = DQN(obs_size, n_actions, device=device)
        self.target_net.load_state_dict(self.net.state_dict())

    def forward(self, state):
        return self.net(state)

    def update_target_net(self):
        self.target_net.load_state_dict(self.net.state_dict())

    def compute_loss(
            self,
            states: np.ndarray,
            actions: np.ndarray,
            rewards: np.ndarray,
            next_states: np.ndarray,
            is_done: np.ndarray,
            loss_func: Loss
    ) -> torch.Tensor:
        return loss_func.compute(
            states,
            actions,
            rewards,
            next_states,
            is_done,
            self.gamma,
            self.net,
            self.target_net
        )

    def get_action(self, obs, extra_state=None) -> Tuple[int, Any]:
        if self.debug_mode:
            action_values = self.net.get_all_action_values(obs)
            print(f"Current action-values: {label_values(action_values, name_list=self.action_labels)}")
        return self.get_optimal_actions(obs), extra_state

    def get_epsilon_actions(self, states: np.ndarray):
        """Pick actions according to an epsilon greedy strategy."""
        return self._get_actions(states, self.epsilon)

    def get_optimal_actions(self, states: np.ndarray):
        """Pick actions according to a greedy strategy."""
        return self._get_actions(states, 0)

    def _get_actions(self, states: np.ndarray, epsilon: float):
        """Pick actions according to an epsilon greedy strategy."""
        states = torch.tensor(states, device=self.device, dtype=torch.float32)
        return self.net.get_epsilon(states, epsilon)

    def parameters(self):
        return self.net.parameters()

    def save(self) -> Dict:
        return {
            "data": self.net.state_dict(),
            "layers": self.net.hidden_layers
        }

    def load(self, agent_data: Dict):
        data, layers = itemgetter("data", "layers")(agent_data)
        self.net.set_hidden_layers(layers)
        self.target_net.set_hidden_layers(layers)
        self.net.load_state_dict(data)
        self.target_net.load_state_dict(data)

    def get_q_values(self, obs) -> Optional[np.ndarray]:
        return self.net.get_all_action_values(obs)

    def set_hidden_layers(self, hidden_layers: List[int]):
        self.net.set_hidden_layers(hidden_layers)
        self.target_net.set_hidden_layers(hidden_layers)


def _play_into_buffer(
        env: gym.Env,
        agent: DQNAgent,
        buffer: ReplayBuffer,
        state,
        n_steps=1,
):
    s = state

    for _ in range(n_steps):
        action = agent.get_epsilon_actions(np.array([s]))[0]
        sp1, r, done, _, _ = env.step(action)
        buffer.add(s, action, r, sp1, done)
        s = sp1
        if done:
            s, _ = env.reset()

    return s


def train_dqn_agent(
        env_factory: Callable[[int], gym.Env],
        agent: DQNAgent,
        hidden_layers: List[int],
        artifacts: Dict,
        learning_rate: float = 1e-4,
        epsilon_decay: StaticLinearDecay = StaticLinearDecay(1, 0.1, 1 * 10 ** 4),
        max_steps: int = 4 * 10 ** 4,
        timesteps_per_epoch=1,
        batch_size=30,
        update_target_net_freq=100,
        evaluation_freq=1000,
        gather_freq=20,
        replay_buffer_size=10**4,
        grad_clip=5000,
        loss_name: str = "td"
):
    env = env_factory(0)
    state, _ = env.reset()

    loss_func = loss_factory(loss_name)

    agent.set_hidden_layers(hidden_layers)

    opt = torch.optim.NAdam(agent.parameters(), lr=learning_rate)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max_steps)

    # Create buffer and fill it with experiences.
    buffer = ReplayBuffer(replay_buffer_size)
    agent.epsilon = 1
    for _ in range(100):
        state = _play_into_buffer(env, agent, buffer, state, n_steps=10**2)
        if len(buffer) >= replay_buffer_size:
            break

    state, _ = env.reset()

    axs = _init_graphing()

    mean_reward_history = []
    loss_history = []
    grad_norm_history = []

    artifacts["reward"] = mean_reward_history
    artifacts["loss"] = loss_history
    artifacts["grad_norm"] = grad_norm_history

    # Train on a per-step basis. If the episode ends, the environment is reset automatically.
    for step in trange(0, max_steps):
        opt.zero_grad()

        agent.epsilon = epsilon_decay.get(step)

        state = _play_into_buffer(env, agent, buffer, state, timesteps_per_epoch)

        (s, a, r, s_dash, is_done) = buffer.sample(batch_size)

        loss = agent.compute_loss(s, a, r, s_dash, is_done, loss_func=loss_func)

        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), grad_clip)
        opt.step()
        lr_sched.step()

        if step % update_target_net_freq == 0:
            # Load agent weights into target_network
            agent.update_target_net()

        if step % gather_freq == 0:
            # Gather info.
            loss_history.append(loss.data.cpu().numpy())
            grad_norm_history.append(grad_norm.data.cpu().numpy())

        if step % evaluation_freq == 0:
            score = evaluate(env_factory(step), agent, n_episodes=3, max_episode_steps=1000)
            mean_reward_history.append(
                score
            )

            _update_graphs(axs, mean_reward_history, loss_history, grad_norm_history, gather_freq, evaluation_freq)

            tqdm.write(f"I[{step}], "
                       f"E[{agent.epsilon: .2f}], "
                       f"S[{score: .2f}], "
                       f"L[{loss:.2f}], "
                       f"LR[{opt.param_groups[0]['lr']}]")


def _init_graphing():
    if can_graph():
        fig, axs = plt.subplots(3, 1)
    else:
        fig, axs = None, None
    return axs


def _update_graphs(axs, mean_reward_history, loss_history, grad_norm_history, gather_freq, eval_freq):
    if can_graph() is False:
        return

    axs[0].cla()
    axs[1].cla()
    axs[2].cla()

    axs[0].set_title("Mean Reward")
    axs[1].set_title("Loss")
    axs[2].set_title("Grad Norm")

    eval_x = np.array(range(len(mean_reward_history))) * eval_freq
    gather_x = np.array(range(len(loss_history))) * gather_freq
    axs[0].plot(eval_x, mean_reward_history)
    axs[1].plot(gather_x, smoothen(loss_history))
    axs[2].plot(gather_x, smoothen(grad_norm_history))

    axs[0].set_xlabel("Steps")
    axs[1].set_xlabel("Steps")
    axs[2].set_xlabel("Steps")

    plt.tight_layout()
    plt.pause(0.05)


def dqn_graphing_wrapper(artifacts, config: ProfileConfig):
    _update_graphs(
        _init_graphing(),
        mean_reward_history=artifacts["reward"],
        loss_history=artifacts["loss"],
        grad_norm_history=artifacts["grad_norm"],
        eval_freq=config.getint("eval_freq"),
        gather_freq=config.getint("gather_freq")
    )


def dqn_training_wrapper(
        env_factory: Callable[[int], gym.Env],
        agent: DQNAgent,
        dqn_config: ProfileConfig,
        artifacts: Dict,
        save_callback
):
    train_dqn_agent(
        env_factory=env_factory,
        agent=agent,
        hidden_layers=dqn_config.getlist("hidden_layers"),
        artifacts=artifacts,
        learning_rate=dqn_config.getfloat("learning_rate"),
        epsilon_decay=StaticLinearDecay(
            start_value=dqn_config.getfloat("init_epsilon"),
            final_value=dqn_config.getfloat("final_epsilon"),
            max_steps=dqn_config.getint("decay_steps")
        ),
        max_steps=dqn_config.getint("total_steps"),
        timesteps_per_epoch=dqn_config.getint("timesteps_per_epoch"),
        batch_size=dqn_config.getint("batch_size"),
        update_target_net_freq=dqn_config.getint("refresh_target_network_freq"),
        evaluation_freq=dqn_config.getint("eval_freq"),
        gather_freq=dqn_config.getint("gather_freq"),
        replay_buffer_size=dqn_config.getint("replay_buffer_size"),
        grad_clip=dqn_config.getint("grad_clip"),
        loss_name=dqn_config.getstr("loss")
    )
