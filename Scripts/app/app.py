import pygame

import config
from app import render
from sim.track import Track


class App:
    def __init__(self) -> None:
        self.running = True

        pygame.init()
        pygame.display.set_caption(config.WINDOW_TITLE)

        self.window = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()

        self.track: Track = Track.load(config.DATA_DIR / "tracks" / "oval.json")

    def run(self) -> None:
        while self.running:
            self.clock.tick(config.FPS)

            self._handle_events()
            self._update()
            self._render()

            pygame.display.flip()
        pygame.quit()

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def _update(self) -> None:
        pass

    def _render(self) -> None:
        self.window.fill(config.BACKGROUND_COLOR)
        render.draw_track(self.window, self.track, 3)
