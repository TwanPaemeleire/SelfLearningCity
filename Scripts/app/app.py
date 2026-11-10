import pygame

import config
from ai.evolution import EvolutionConfig
from ai.trainer import Trainer
from app import render
from sim.track import Track


class App:
    def __init__(self) -> None:
        self.running = True

        pygame.init()
        pygame.display.set_caption(config.WINDOW_TITLE)
        self.font: pygame.Font = pygame.font.Font(None, 24)

        self.window = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()

        self.show_sensors = False

        self.track = Track.load(config.DATA_DIR / "tracks" / "circuit.json")
        self.trainer = Trainer(self.track, EvolutionConfig())

        self.accumulator = 0.0

    def run(self) -> None:
        while self.running:
            self.accumulator += self.clock.tick(config.FPS) / 1000.0
            # If steps are too slow to keep up, don't try to catch up forever (the "spiral of death").
            # The simulation then runs in slow motion instead of freezing the app.
            self.accumulator = min(self.accumulator, config.MAX_CATCH_UP_SECONDS)

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

    def _update(self) -> None:
        self.trainer.step()

    def _render(self) -> None:
        self.window.fill(config.BACKGROUND_COLOR)
        render.draw_track(self.window, self.trainer.world.track, 0)
        for car in self.trainer.world.cars:
            if self.show_sensors:
                render.draw_sensors(self.window, car, self.trainer.world.walls_near(car.position))
            render.draw_car(self.window, car)
