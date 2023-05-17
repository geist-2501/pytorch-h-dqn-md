from collections import deque
from typing import Dict, List, Callable

import gym
import numpy as np
import torch.optim.lr_scheduler
from scipy.ndimage.filters import uniform_filter1d
from crete import Agent

try:
    import tkinter
    tkinter_available = True
except ModuleNotFoundError:
    tkinter_available = False


def can_graph():
    global tkinter_available
    return tkinter_available


class StaticLinearDecay:
    def __init__(self, start_value, final_value, max_steps):
        self.start_value = start_value
        self.final_value = final_value
        self.max_steps = max_steps

    def get(self, step):
        step = min(step, self.max_steps)
        upper = self.start_value * (self.max_steps - step)
        lower = self.final_value * step
        return (upper + lower) / self.max_steps


class MeteredLinearDecay:
    def __init__(self, start_value, final_value, max_steps):
        self._decay = StaticLinearDecay(start_value, final_value, max_steps)
        self._tick = 0

    def next(self):
        v = self._decay.get(self._tick)
        self._tick += 1
        return v

    def get_epsilon(self):
        return self._decay.get(self._tick)


class KMemory:
    """Memory for success rate based decay that only considers the last K attempts."""

    def __init__(self, size: int) -> None:
        self._mem = deque(maxlen=size)

    def get_success_rate(self) -> float:
        n_total_attempts = len(self._mem)
        if n_total_attempts == 0:
            return 0

        return sum(self._mem) / n_total_attempts

    def add_attempt(self, was_successful: bool):
        self._mem.append(was_successful)


class PermanentMemory:
    """Memory for success rate based decay that considers all attempts."""

    def __init__(self) -> None:
        self._n_successful_attempts = 0
        self._n_failed_attempts = 0

    def get_success_rate(self) -> float:
        n_total_attempts = self._n_successful_attempts + self._n_failed_attempts
        if n_total_attempts == 0:
            return 0

        return self._n_successful_attempts / n_total_attempts

    def add_attempt(self, was_successful: bool):
        if was_successful:
            self._n_successful_attempts += 1
        else:
            self._n_failed_attempts += 1


class SuccessRateBasedDecay:
    """Epsilon decay based on the success rate of the agent."""

    def __init__(self, start_value, final_value, min_steps, memory_size: int = None):
        """
        :param memory_size: Size of the memory to use for the success rate. If None, a permanent memory is used.
        """
        self._decay_limit = StaticLinearDecay(start_value, final_value, min_steps)
        self.start_value = start_value
        self.final_value = final_value
        self._last_step = 0
        self._mem = KMemory(memory_size) if memory_size else PermanentMemory()

    def get_success_rate(self) -> float:
        return self._mem.get_success_rate()

    def next(self, step: int, was_successful: bool) -> float:
        self._last_step = step
        self._mem.add_attempt(was_successful)
        return self.get_epsilon()

    def get_epsilon(self):
        inv_success_rate = (1 - self.get_success_rate()) * self.start_value
        minimum_decay = self._decay_limit.get(self._last_step)
        return max(inv_success_rate, minimum_decay)


class SuccessRateWithTimeLimitDecay:
    """Same as SuccessRateBasedDecay but considers actions beyond a time limit unsuccessful."""
    def __init__(self, start_value, final_value, min_steps, max_t: int, memory_limit: int = None):
        self._base_decay = SuccessRateBasedDecay(start_value, final_value, min_steps, memory_limit)
        self._max_t = max_t

    def get_success_rate(self) -> float:
        return self._base_decay.get_success_rate()

    def next(self, step: int, was_successful: bool, duration: int) -> float:
        was_successful &= duration <= self._max_t
        return self._base_decay.next(step, was_successful)

    def get_epsilon(self):
        return self._base_decay.get_epsilon()


def smoothen(data):
    return uniform_filter1d(data, size=30)


def evaluate(
        env: gym.Env,
        agent: Agent,
        n_episodes=1,
        max_episode_steps=10000
) -> float:
    total_ep_rewards = []
    for _ in range(n_episodes):
        s, _ = env.reset()
        total_ep_reward = 0
        extra_state = None
        for _ in range(max_episode_steps):
            a, extra_state = agent.get_action(s, extra_state)
            s, r, done, _, _ = env.step(a)
            total_ep_reward += r

            if done:
                break

        total_ep_rewards.append(total_ep_reward)
    return np.mean(total_ep_rewards).item()


def flatten(obs: Dict) -> List:
    return [
        *obs["agent_loc"],
        *obs["agent_obs"],
        *obs["agent_inv"],
        *obs["depot_locs"],
        *obs["output_loc"],
        *obs["depot_queues"],
        *obs["depot_ages"],
    ]


def parse_int_list(raw: str) -> List[int]:
    parts = raw.split(",")
    return list(map(lambda part: int(part), parts))


class BetterReduceLROnPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
    def get_last_lr(self):
        return self.optimizer.param_groups[0]['lr']  # ew.


def label_values(
        values: np.ndarray,
        name_func: Callable[[int], str] = None,
        name_list: List[str] = None
) -> List[str]:
    out = []
    if name_func is not None:
        names = [name_func(action) for action in range(len(values))]
    else:
        names = name_list

    for name, value in zip(names, values):
        out.append(f"{name}: {value:.2f}")

    return out
