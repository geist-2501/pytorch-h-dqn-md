from typing import Tuple

import numpy as np


class Map:

    def __init__(self, layout, p, max_order_size) -> None:
        super().__init__()
        self.layout = layout
        self.p = p
        self.max_order_size = max_order_size

        output_loc, depot_locs, len_x, len_y = self._get_map_info()
        self.output_loc = output_loc
        self.depot_locs = depot_locs
        self.len_x = len_x
        self.len_y = len_y

        assert len(depot_locs) == len(p)

    def _get_map_info(self) -> Tuple[np.ndarray, np.ndarray, int, int]:
        output_loc = None
        depot_locs = []
        len_y = len(self.layout)
        len_x = len(self.layout[0])
        for y in range(len_y):
            for x in range(len_x):
                cell = self.layout[y][x]
                if cell == 'o':
                    # Get output loc.
                    output_loc = np.array([x, y], dtype=int)
                elif cell == 'd':
                    # Get depot locs.
                    depot_locs.append(np.array([x, y], dtype=int))

        return output_loc, np.array(depot_locs), len_x, len_y


warehouses = {
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