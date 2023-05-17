from typing import Optional


class TimeKeeper:

    def __init__(self) -> None:
        super().__init__()
        self._in_pretrain_mode = False
        self._pretrain_steps = 0
        self._q1_steps = 0
        self._q2_steps = 0
        self._env_steps = 0
        self._n_episodes = 0

    def should_train_q1(self) -> bool:
        return True

    def should_train_q2(self) -> bool:
        return True

    def step_env(self):
        self._env_steps += 1

    def get_env_steps(self):
        return self._env_steps

    def step_q1(self):
        if self._in_pretrain_mode:
            self._pretrain_steps += 1
        else:
            self._q1_steps += 1

    def get_q1_steps(self):
        if self._in_pretrain_mode:
            return self._pretrain_steps
        else:
            return self._q1_steps

    def step_q2(self):
        self._q2_steps += 1

    def get_q2_steps(self):
        return self._q2_steps

    def step_episode(self):
        self._n_episodes += 1

    def get_episode_steps(self):
        return self._n_episodes

    def train_mode(self):
        self._in_pretrain_mode = False

    def pretrain_mode(self):
        self._in_pretrain_mode = True


class KCatchUpTimeKeeper(TimeKeeper):
    """Timekeeper that implements K-catch-up training for h-DQN."""

    def __init__(self) -> None:
        super().__init__()
        self.k = None
        self._k_end = 0

    def set_k_catch_up(self, k: Optional[int]):
        if k is None:
            return
        self.k = k
        self._k_end = self._q1_steps + self.k

    def should_train_q1(self) -> bool:
        if self.k is None:
            return True
        else:
            return not self._q1_steps > self._k_end

    def should_train_q2(self) -> bool:
        return not self._in_pretrain_mode

    def step_q2(self):
        self._q2_steps += 1
        if self.k is not None and self._q2_steps >= self._k_end:
            self._k_end = self._q2_steps + self.k


class SerialTimekeeper(TimeKeeper):
    """Timekeeper class that implements a seperate 2-stage training program."""

    def should_train_q1(self) -> bool:
        return self._in_pretrain_mode

    def should_train_q2(self) -> bool:
        return not self._in_pretrain_mode

