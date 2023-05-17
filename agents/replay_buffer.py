from abc import ABC, abstractmethod
from collections import deque
from typing import Tuple

import numpy as np
import random


class Buffer(ABC):
    """Interface for allowing agents to pull samples while using specific `add()` methods."""
    @abstractmethod
    def sample(self, batch_size: int) -> Tuple:
        raise NotImplementedError


class _ReplayBuffer:
    """Datastructure for a replay buffer that supports arbitrary tuples."""

    def __init__(self, size: int):
        self._storage = deque(maxlen=size)
        self.max_size = size
        self._entry_size = None

    def __len__(self):
        return len(self._storage)

    def __getitem__(self, item):
        return self._storage[item]

    def add(self, *args):
        if self._entry_size is None:
            self._entry_size = len(args)
        elif len(args) != self._entry_size:
            raise RuntimeError(f"Inconsistent entry size ({self._entry_size} != {len(args)})")

        self._storage.append(args)

    def sample(self, batch_size: int) -> Tuple:
        idxes = [
            random.randint(0, len(self._storage) - 1)
            for _ in range(batch_size)
        ]
        return self._encode(idxes)

    def _encode(self, idxes) -> Tuple:
        samples = tuple([] for _ in range(self._entry_size))
        for i in idxes:
            data = self._storage[i]
            for s, d in enumerate(data):
                samples[s].append(d)
        return tuple(np.array(s) for s in samples)


class ReplayBuffer(Buffer):
    """Replay buffer that supports (s, a, r, s', is_done) tuples."""
    def __init__(self, size):
        self._buffer = _ReplayBuffer(size)

    def __len__(self):
        return len(self._buffer)

    def add(self, obs_t, action, reward, obs_tp1, done):
        self._buffer.add(obs_t, action, reward, obs_tp1, done)

    def sample(self, batch_size):
        return self._buffer.sample(batch_size)


class ReplayBufferWithStats(ReplayBuffer):
    """
    Replay buffer that supports (s, a, r, s', is_done) tuples.
    Also tracks statistics on goals via the `contents` variable.
    """

    def __init__(self, size, n_goals):
        super().__init__(size)
        self.n_goals = n_goals
        self.contents = np.zeros(n_goals)

    def add(self, obs_t, action, reward, obs_tp1, done):
        goal = self._get_goal(obs_t)

        self.contents[goal] += 1

        if len(self._buffer) >= self._buffer.max_size:
            to_be_discarded = self._buffer[0]
            old_goal = self._get_goal(to_be_discarded[0])
            self.contents[old_goal] -= 1

        self._buffer.add(obs_t, action, reward, obs_tp1, done)

    def _get_goal(self, obs):
        goal_onehot = obs[-self.n_goals:]
        return np.argmax(goal_onehot).item()


class ReplayBufferWithDelta(Buffer):
    """Replay buffer that supports (s, a, r, s', δ, is_done) tuples."""

    def __init__(self, size):
        self._buffer = _ReplayBuffer(size)

    def __len__(self):
        return len(self._buffer)

    def add(self, obs_t, action, reward, obs_tp1, delta, done):
        self._buffer.add(obs_t, action, reward, obs_tp1, delta, done)

    def sample(self, batch_size) -> Tuple:
        return self._buffer.sample(batch_size)
