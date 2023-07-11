import math
from collections import defaultdict
from typing import Optional, Tuple, List, Union, Dict

import numpy as np
from gym import spaces
from gym.core import ActType, ObsType
from matplotlib import pyplot as plt

from factory_machines.envs.fm_env_base import FactoryMachinesEnvBase
from factory_machines.envs.order_generators import OrderGenerator, GaussianOrderGenerator, StaticOrderGenerator
from factory_machines.envs.pygame_utils import draw_lines


class FactoryMachinesEnvMulti(FactoryMachinesEnvBase):

    _reward_per_order = 10  # The amount of reward for a fulfilled order.
    _item_pickup_reward = 1  # The amount of reward for picking up a needed item.
    _item_dropoff_reward = 0  # The amount of reward for dropping off a needed item.
    _item_pickup_punishment = -2  # The amount of reward for picking up an item it shouldn't.
    _collision_punishment = -1
    _timestep_punishment = -0.5
    _episode_reward = 100

    _age_bands = 3  # The number of stages of 'oldness'.
    _max_age_reward = 3  # The max reward that can be gained from an early order completion.
    _age_max_timesteps = 50  # The amount of timesteps that can elapse before an order is considered old.

    def __init__(
            self,
            render_mode: Optional[str] = None,
            map_id="0",
            num_orders=10,
            agent_capacity=10,
            order_generator: OrderGenerator = None,
            order_override: str = None,
            verbose=False,
            correct_obs=True
    ) -> None:
        super().__init__(render_mode, map_id, agent_capacity, verbose, correct_obs)

        if order_override is not None:
            self._order_generator = StaticOrderGenerator(np.fromstring(order_override, sep=',', dtype=int))
        elif order_generator is None:
            self._order_generator = GaussianOrderGenerator(self._map.p, self._map.max_order_size)
        else:
            self._order_generator = order_generator

        self.observation_space = spaces.Dict(
            {
                "agent_loc": spaces.Box(0, np.array([self._len_x, self._len_y]) - 1, shape=(2,), dtype=int),
                "agent_obs": spaces.Box(0, 1, shape=(9,), dtype=int),
                "agent_inv": spaces.Box(0, 10, shape=(len(self._depot_locs),), dtype=int),
                "depot_locs": spaces.Box(0, max(self._len_x, self._len_y), shape=(len(self._depot_locs) * 2,), dtype=int),
                "depot_queues": spaces.Box(0, num_orders, shape=(len(self._depot_locs),), dtype=int),
                "output_loc": spaces.Box(0, max(self._len_x, self._len_y), shape=(2,), dtype=int),
                "depot_ages": spaces.Box(0, self._age_bands, shape=(len(self._depot_locs),), dtype=int),
            }
        )

        num_orders = int(num_orders)

        self._total_num_orders = num_orders
        self._num_orders_pending = num_orders
        self._open_orders: List[Tuple[int, np.ndarray]] = []

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[ObsType, dict]:
        obs, _ = super().reset(seed=seed, options=options)

        self._num_orders_pending = self._total_num_orders
        self._open_orders = []

        # Immediately create an open order.
        self._num_orders_pending -= 1
        order = self._order_generator.make_order(self._num_depots)
        self._open_orders.append((self._timestep, order))

        return obs, {}

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        obs, reward, terminated, _, info = super().step(action)

        # Process orders.
        should_create_order = self._order_generator.should_make_order(len(self._open_orders))
        if should_create_order and self._num_orders_pending > 0:
            self._num_orders_pending -= 1
            order = self._order_generator.make_order(self._num_depots)
            self._open_orders.append((self._timestep, order))

        terminated = self._num_orders_pending == 0 and len(self._open_orders) == 0
        reward += self._episode_reward if terminated else 0

        info = {
            **info,
            "orders per minute": self._get_num_completed_orders() / self._timestep * 60
        }

        return obs, reward, terminated, False, info

    def get_info(self):
        ages = self._get_depot_ages()
        orders = self._open_orders
        queues = self._get_depot_queues()
        return ages, orders, queues

    def _render_info(self, font, header_origin, screen_width, spacing):

        # Draw table header.
        table_rows = [
            "   | " + ' '.join([f"{f'D{x}':>3}" for x in range(self._num_depots)]),
            "INV| " + self._add_table_padding(self._agent_inv),
            "DEP| " + self._add_table_padding(self._get_depot_queues()),
            "AGE| " + self._add_table_padding(self._get_depot_ages()),
            "   |",
        ]

        for order in self._open_orders:
            order_t, order_items = order
            table_rows.append(f"{order_t:>3}| " + self._add_table_padding(order_items))

        draw_lines(table_rows, self.screen, header_origin, font, self.colors["text"])

    @staticmethod
    def _add_table_padding(arr):
        return ' '.join(map(lambda i: f"{i:3}", arr))

    def _depot_drop_off(self) -> int:
        # Go through each open order and strike off items the agent holds.
        # Then, if an order has been completed, move it to the completed pile.
        reward = 0
        for i, order in enumerate(self._open_orders.copy()):
            order_t, order_items = order
            items_fulfilled = np.minimum(self._agent_inv, order_items)
            reward += sum(items_fulfilled) * self._item_dropoff_reward

            new_order = order_items - items_fulfilled
            self._open_orders[i] = (order_t, new_order)

            self._agent_inv -= items_fulfilled

            if sum(new_order) == 0:
                # Order is complete!
                order_age = self._timestep - order_t
                reward += self._reward_per_order + self._sample_age_reward(order_age)

        # Remove complete orders.
        self._open_orders[:] = [order for order in self._open_orders if sum(order[1]) != 0]

        return reward

    def _get_obs(self):
        return {
            **super()._get_obs(),
            "depot_ages": self._get_depot_ages()
        }

    def _get_depot_queues(self):
        depot_queues = np.zeros(self._num_depots, dtype=int)
        for order in self._open_orders:
            _, order_items = order
            depot_queues += order_items

        return depot_queues

    def _get_depot_ages(self):
        """Get the age of the oldest open order on the depots."""
        depot_ages = np.zeros(self._num_depots, dtype=int)
        for order in self._open_orders:
            order_t, order_items = order
            order_age = self._timestep - order_t
            mask = order_items * order_age
            depot_ages = np.maximum(depot_ages, mask)

        # Cutoff.
        depot_ages = self._get_age(depot_ages)

        return depot_ages

    def _get_age(self, order_t: Union[int, np.ndarray]) -> Union[int, np.ndarray]:
        compression_factor = self._age_max_timesteps / self._age_bands

        if type(order_t) is int:
            # Cutoff.
            order_t = np.minimum(order_t, self._age_max_timesteps).item()
            return math.floor(order_t / compression_factor)
        elif type(order_t) is np.ndarray:
            order_t = np.minimum(order_t, self._age_max_timesteps)
            return np.floor(order_t / compression_factor).astype(int)

        raise RuntimeError("Times are neither int nor ndarray.")

    def _sample_age_reward(self, age: int) -> float:
        compressed_age = self._get_age(age)
        reward_ratio = (self._age_bands - compressed_age) / self._age_bands
        return reward_ratio * self._max_age_reward

    def _get_num_completed_orders(self):
        return self._total_num_orders - (self._num_orders_pending + len(self._open_orders))


def _swizzle(data: List[Dict]) -> Dict:
    """
    Converts a list of dictionaries to a dictionary of lists.
    No, I'm not giving it a better name.
    """
    swizzled_data = defaultdict(lambda: [])
    for entry in data:
        for k, v in entry.items():
            swizzled_data[k].append(v)
    return swizzled_data


def fm_multi_graphing_wrapper(env_args: Dict, agent_names: List[str], scores: Tuple):
    map_id = env_args["map_id"] if "map_id" in env_args else "0"
    title = f"Performance on map {map_id}"

    rewards, reward_errs, infos, info_errs = scores

    num_plots = len(info_errs[0]) + 1
    fig, axs = plt.subplots(num_plots, 1, figsize=(3, 2 * num_plots))
    fig.suptitle(title)

    swizzled_infos = _swizzle(infos)
    swizzled_infos["reward"] = rewards

    swizzled_errs = _swizzle(info_errs)
    swizzled_errs["reward"] = reward_errs

    cmap = plt.get_cmap("tab10")
    colours = [cmap(i) for i in range(len(agent_names))]

    def _graph_bar(ax, metric):
        values = swizzled_infos[metric]
        errs = swizzled_errs[metric]
        y_pos = np.arange(len(agent_names))
        ax.barh(y_pos, values, color=colours, xerr=errs, label=agent_names)
        ax.set_yticks(y_pos, labels=agent_names)
        ax.set_title(metric)

    _graph_bar(axs[0], "reward")

    _graph_bar(axs[1], "orders per minute")
    axs[1].set_xlabel("Orders per minute")

    _graph_bar(axs[2], "distance")
    axs[2].set_xlabel("Distance travelled (m)")

    _graph_bar(axs[3], "timesteps")
    axs[3].set_xlabel("Time taken (s)")

    plt.tight_layout()
    plt.show()
