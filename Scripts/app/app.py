import pygame

import config
from app import render
from sim.car import Car
from sim.track import Track


class App:
    def __init__(self) -> None:
        self.running = True

        pygame.init()
        pygame.display.set_caption(config.WINDOW_TITLE)

        self.window = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()

        self.track: Track = Track.load(config.DATA_DIR / "tracks" / "oval.json")
        self.car: Car = Car(self.track.start_pos, self.track.start_angle)
        self.accumulator = 0.0

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

    def _update(self) -> None:

        keys = pygame.key.get_pressed()

        steer = keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]
        throttle = keys[pygame.K_UP] - keys[pygame.K_DOWN]

        if not self.car.alive:
            if keys[pygame.K_r]:
                self.car.respawn()
        else:
            self.car.update(steer, throttle, config.DT)
            self.car.check_collision(self.track.walls)

    def _render(self) -> None:
        self.window.fill(config.BACKGROUND_COLOR)
        render.draw_track(self.window, self.track, 3)
        render.draw_car(self.window, self.car)
