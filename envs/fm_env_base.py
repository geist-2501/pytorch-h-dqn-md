from typing import Optional, Union, List, Tuple, Dict
from abc import ABC, abstractmethod

import gym
import numpy as np
import pygame
from gym import spaces
from gym.core import RenderFrame, ActType, ObsType

from factory_machines.envs.pygame_utils import History
from factory_machines.envs.route_tracer import RouteTracer
from factory_machines.envs.warehouse import Map
from talos import get_cli_state


def _opt_bool(opt: Union[str, bool]) -> bool:
    if type(opt) is str:
        return opt == "True"
    else:
        return opt


class FactoryMachinesEnvBase(gym.Env, ABC):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 4}
    maps: Dict[str, Map] = {
        "0": Map([
            'o.d',
            '...',
            'd.d',
        ],
            [0.1, 0.2, 0.6],
            3
        ),
        "0-longer": Map([
            'o.d.d',
            '.....',
            'd.d.d',
        ],
            [1, 1, 1, 1, 1],
            3
        ),
        "1": Map([
            'o.w.d',
            '..w..',
            '.....',
            'd...d',
        ],
            [1, 1, 1],
            3
        ),
        "2": Map([
            '...o...',
            '.d...d.',
            '.d.w.d.',
            '.d...d.',
            '.......',
        ],
            [0.2, 0.1, 0.5, 0.3, 0.2, 0.3],
            4
        ),
        "3": Map([
            '....o....',
            '.........',
            '.dwd.dwd.',
            '.dwd.dwd.',
            '.dwd.dwd.',
            '.dwd.dwd.',
            '.dwd.dwd.',
            '.........',
            '.........',
        ],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            5
        ),
        "slam-example": Map([
            '....o.ww.',
            'd.ww.....',
            'wwwwwww..',
            '....d....',
        ],
            [1, 2],
            2
        ),
        "aisled-example": Map([
            '...o...',
            'dwd.dwd',
            'dwd.dwd',
            'dwd.dwd',
            '.......',
        ],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            3
        ),
    }

    colors = {
        'background': (255, 255, 255),
        'foreground': (50, 50, 50),
        'gridlines': (214, 214, 214),
        'text': (10, 10, 10),
        'agent': (43, 79, 255),
        'agent-light': (7, 35, 176),
        "route": (255, 166, 166),
        "black": (0, 0, 0)
    }

    _item_pickup_reward = 1
    _item_pickup_punishment = -0.5
    _item_dropoff_reward = 1
    _collision_punishment = -0.1
    _timestep_punishment = -0.1

    up, left, down, right, grab = range(5)

    def __init__(
            self,
            render_mode: Optional[str] = None,
            map_id="0",
            agent_capacity=10,
            verbose=False,
            correct_obs=True
    ) -> None:

        self.debug_mode = get_cli_state().debug_mode

        agent_capacity = int(agent_capacity)
        self._verbose = _opt_bool(verbose)
        self._correct_obs = _opt_bool(correct_obs)

        self._map = self.maps[map_id]

        self._output_loc = self._map.output_loc
        self._depot_locs = self._map.depot_locs
        self._num_depots = len(self._depot_locs)

        self._len_x = self._map.len_x
        self._len_y = self._map.len_y

        self._agent_cap = agent_capacity
        self._agent_loc = np.array(self._output_loc, dtype=int)
        self._agent_inv = np.zeros(self._num_depots, dtype=int)

        self._last_action = 0

        self._history = History(size=8)

        self.observation_space = spaces.Dict(
            {
                "agent_loc": spaces.Box(0, np.array([self._len_x, self._len_y]) - 1, shape=(2,), dtype=int),
                "agent_obs": spaces.Box(0, 1, shape=(9,), dtype=int),
                "agent_inv": spaces.Box(0, 10, shape=(len(self._depot_locs),), dtype=int),
                "depot_locs": spaces.Box(0, max(self._len_x, self._len_y), shape=(len(self._depot_locs) * 2,), dtype=int),
                "depot_queues": spaces.Box(0, 10, shape=(len(self._depot_locs),), dtype=int),
                "output_loc": spaces.Box(0, max(self._len_x, self._len_y), shape=(2,), dtype=int),
            }
        )

        self.action_space = spaces.Discrete(5)  # Up, down, left, right, grab.

        # Utility vectors for moving the agent.
        self._action_to_direction = {
            0: np.array([0, -1], dtype=int),  # w 0, -1
            1: np.array([-1, 0], dtype=int),  # a -1, 0
            2: np.array([0, 1], dtype=int),  # s 0, 1
            3: np.array([1, 0], dtype=int),  # d 1, 0
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Used for human friendly rendering.
        self.screen = None
        self.clock = None
        self.route_tracer = RouteTracer()

        # Stats.
        self._dist_travelled = 0
        self._timestep = 0

    def _get_obs(self):

        local_obs = np.zeros((3, 3))

        a_x, a_y = self._agent_loc
        for x in range(3):
            for y in range(3):
                map_x = a_x + x - 1
                map_y = a_y + y - 1
                if self._is_oob(map_x, map_y) or self._map.layout[map_y][map_x] == 'w':
                    local_obs[y, x] = 1

        return {
            "agent_loc": self._agent_loc,
            "agent_obs": local_obs.flatten(),
            "agent_inv": self._agent_inv,
            "depot_locs": self._depot_locs.flatten(),
            "depot_queues": self._get_depot_queues(),
            "output_loc": self._output_loc,
        }

    @abstractmethod
    def _get_depot_queues(self):
        """Get the amount of items needed from each queue."""
        pass

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[ObsType, dict]:
        super().reset(seed=seed, options=options)

        if seed is not None:
            np.random.seed(seed)

        self._agent_loc = self._output_loc

        self._agent_inv = np.zeros(self._num_depots, dtype=int)

        obs = self._get_obs()

        self._timestep = 0
        self._dist_travelled = 0
        self.route_tracer = RouteTracer(k=20)
        self.route_tracer.trace(self._agent_loc)

        return obs, {}

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:

        self._last_action = action
        self._timestep += 1

        # Process actions.
        action_reward = 0
        if action < self.grab:
            # Action is a move op.
            direction = self._action_to_direction[action]
            new_pos = self._agent_loc + direction
            if self._is_oob(new_pos[0], new_pos[1]) or self._map.layout[new_pos[1]][new_pos[0]] == 'w':
                action_reward += self._collision_punishment
                self._history.log("Agent bumped into a wall.")
            else:
                self._agent_loc = new_pos
                self._dist_travelled += 1
        elif action == self.grab:
            # Action is a grab op.
            action_reward = self._try_grab()

        # Check depot drop off.
        drop_off_reward = 0
        if np.array_equal(self._agent_loc, self._output_loc):
            drop_off_reward = self._depot_drop_off()

        reward = action_reward + drop_off_reward + self._timestep_punishment

        obs = self._get_obs()
        info = {
            "timesteps": self._timestep,
            "distance": self._dist_travelled,
        }

        self.route_tracer.trace(self._agent_loc, action == self.grab)

        if self.render_mode == "human":
            self.render()

        if self._verbose:
            print(f"Reward: {reward}")
            print(obs)

        return obs, reward, False, False, info

    @abstractmethod
    def _depot_drop_off(self):
        """Handle what happens when the agent arrives at the depot."""
        pass

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        len_x = self._len_x
        len_y = self._len_y
        cell_size = 64
        spacing = 8

        pygame.font.init()
        font = pygame.font.SysFont("monospace", 13)

        header_width = 600
        header_origin = (cell_size * len_x + spacing, spacing)

        screen_width, screen_height = cell_size * len_x + header_width, cell_size * len_y

        pygame.init()
        if self.screen is None:
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((screen_width, screen_height))
            else:
                self.screen = pygame.Surface((screen_width, screen_height))

        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.screen.fill(self.colors["background"])

        # Add gridlines
        for x in range(len_x + 1):
            pygame.draw.line(
                self.screen,
                self.colors["gridlines"],
                (cell_size * x, 0),
                (cell_size * x, len_y * cell_size),
                width=3,
            )

        for y in range(len_y + 1):
            pygame.draw.line(
                self.screen,
                self.colors["gridlines"],
                (0, cell_size * y),
                (len_x * cell_size, cell_size * y),
                width=3,
            )

        # Draw depots
        output_text = font.render("O", True, self.colors["text"])
        self.screen.blit(output_text, self._output_loc * cell_size)
        depot_queues = self._get_depot_queues()
        for depot_num, depot_loc in enumerate(self._depot_locs):
            item_diff = depot_queues[depot_num] - self._agent_inv[depot_num]
            depot_text = font.render(f"D{depot_num} - {item_diff}", True, self.colors["text"])
            self.screen.blit(depot_text, depot_loc * cell_size)

        # Draw walls.
        for x in range(self._len_x):
            for y in range(self._len_y):
                if self._map.layout[y][x] == 'w':
                    pygame.draw.rect(
                        self.screen,
                        self.colors["foreground"],
                        pygame.Rect(
                            (x * cell_size, y * cell_size),
                            (cell_size, cell_size)
                        )
                    )

        if self.debug_mode:
            self.route_tracer.render(self.screen, cell_size, self.colors["black"], self.colors["route"])

        # Draw agent.
        pygame.draw.circle(
            self.screen,
            self.colors["agent"] if self._last_action != 4 else self.colors["agent-light"],
            (self._agent_loc + 0.5) * cell_size,
            cell_size / 3,
        )

        # Draw text.
        self._render_info(font, header_origin, screen_width, spacing)

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def _render_info(self, font, header_origin, screen_width, spacing):
        # Draw inventory.
        inv_text = font.render("INV: " + self._format_depots(self._agent_inv), True, self.colors["text"])
        inv_text_rect = self.screen.blit(inv_text, header_origin)

        # Draw depot queues.
        depot_text = font.render("DEP: " + self._format_depots(self._get_depot_queues()), True, self.colors["text"])
        depot_text_rect = self.screen.blit(depot_text, (header_origin[0], inv_text_rect.bottom + spacing))

        # Draw history log.
        history_text = self._history.render(font, self.colors["text"], width=screen_width)
        self.screen.blit(history_text, (header_origin[0], depot_text_rect.bottom + spacing))

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()

    @staticmethod
    def get_keys_to_action():
        return {
            'w': 0,
            'a': 1,
            's': 2,
            'd': 3,
            'g': 4,
        }

    def _try_grab(self) -> float:
        """
        Try and add the current depot resource to the agent inventory.
        Returns reward if agent needed the resource, punishment if not.
        """
        depot_queues = self._get_depot_queues()

        for depot_num, depot_loc in enumerate(self._depot_locs):
            if not np.array_equal(self._agent_loc, depot_loc):
                continue

            # Agent is on a depot.
            if self._agent_inv[depot_num] == self._agent_cap:
                # Agent at max capacity for item, administer punishment.
                self._history.log(f"Inv full for D{depot_num}.")

                return self._item_pickup_punishment

            elif self._agent_inv[depot_num] < depot_queues[depot_num]:
                # Agent picks up a needed resource.
                self._history.log(f"Picked up required item D{depot_num}.")
                self._agent_inv[depot_num] += 1

                return self._item_pickup_reward

            elif self._agent_inv[depot_num] >= depot_queues[depot_num]:
                # Agent picks up an unneeded resource.
                self._history.log(f"Picked up unnecessary item D{depot_num}.")
                self._agent_inv[depot_num] += 1

                # Punishing this behaviour is a matter of debate.
                # It may lock out interesting behaviours like stockpiling popular items.
                return self._item_pickup_punishment

        self._history.log("Agent tried to grab a blank tile.")
        return self._item_pickup_punishment

    def _is_oob(self, x: int, y: int):
        return x < 0 or x >= self._len_x or y < 0 or y >= self._len_y

    @staticmethod
    def _format_depots(arr):
        return ', '.join(map(lambda i: f"D{i[0]}: {i[1]}", enumerate(arr)))
