from collections import deque
from typing import Tuple, List

import pygame


def draw_lines(
        lines: List[str],
        surface: pygame.Surface,
        loc: Tuple[int, int],
        font: pygame.font.Font,
        color,
):
    _, font_size = font.size("test")
    x_loc, y_loc = loc
    y_offset = y_loc
    for line in lines:
        text = font.render(line, True, color)
        text_rect = surface.blit(text, (x_loc, y_offset))
        y_offset = text_rect.bottom


class History:
    def __init__(self, size=4):
        self.size = size
        self._logs = deque(maxlen=size)

    def log(self, message: str):
        self._logs.append(message)

    def render(
            self,
            font: pygame.font.Font,
            color,
            width=300
    ) -> pygame.Surface:
        _, font_height = font.size("test")
        surface = pygame.Surface((width, font_height * self.size))
        surface.fill((255, 255, 255))
        y_offset = 0
        for entry in self._logs:
            history_text = font.render(entry, True, color)
            history_text_rect = surface.blit(history_text, (0, y_offset))
            y_offset = history_text_rect.bottom

        return surface
