import pygame

import config
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
        self.world = World(Track.load(config.DATA_DIR / "tracks" / "oval.json"))

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
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_F1:
                self.show_sensors = not self.show_sensors

    def _update(self) -> None:
        keys = pygame.key.get_pressed()

        steer = keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]
        throttle = keys[pygame.K_UP] - keys[pygame.K_DOWN]

        self.world.step(config.DT, [(steer, throttle)])

        if not self.world.cars[0].alive:
            if keys[pygame.K_r]:
                self.world.cars[0].respawn()

    def _render(self) -> None:
        self.window.fill(config.BACKGROUND_COLOR)
        render.draw_track(self.window, self.world.track, self.world.cars[0].next_checkpoint_index)
        if self.show_sensors:
            render.draw_sensors(self.window, self.world.cars[0], self.world.track.walls)
        render.draw_car(self.window, self.world.cars[0])
        render.draw_fitness_and_kill_timer(self.window, self.world.cars[0], self.font)
