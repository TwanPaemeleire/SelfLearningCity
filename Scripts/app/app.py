import numpy as np
import pygame

import config
from ai.brain import Brain
from app import render
from sim.track import Track
from sim.world import World


class App:
    def __init__(self) -> None:
        self.running = True

        pygame.init()
        pygame.display.set_caption(config.WINDOW_TITLE)
        self.font: pygame.Font = pygame.font.Font(None, 24)

        self.window = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()

        self.show_sensors = False

        self.rng = np.random.default_rng(42)
        self.track = Track.load(config.DATA_DIR / "tracks" / "oval.json")
        self._spawn_random_cars()

        self.accumulator = 0.0

    def _spawn_random_cars(self) -> None:
        """Start a new World with 20 cars, each driven by a fresh random brain."""
        brains = [Brain.random(self.rng) for _ in range(20)]
        self.world = World(self.track, brains)

    def run(self) -> None:
        while self.running:
            self.accumulator += self.clock.tick(config.FPS) / 1000.0

            self._handle_events()
            while self.accumulator >= config.DT:
                self.accumulator -= config.DT
                self._update()
            self._render()

            pygame.display.flip()
        pygame.quit()

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_F1:
                self.show_sensors = not self.show_sensors
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                self._spawn_random_cars()

    def _update(self) -> None:
        self.world.step(config.DT)

    def _render(self) -> None:
        self.window.fill(config.BACKGROUND_COLOR)
        render.draw_track(self.window, self.world.track, 0)
        for car in self.world.cars:
            if self.show_sensors:
                render.draw_sensors(self.window, car, self.world.track.walls)
            render.draw_car(self.window, car)
