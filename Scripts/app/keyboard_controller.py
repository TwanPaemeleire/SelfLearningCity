import pygame


class KeyboardController:
    def decide(self, inputs: list[float]) -> tuple[float, float]:
        keys = pygame.key.get_pressed()

        steer = keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]
        throttle = keys[pygame.K_UP] - keys[pygame.K_DOWN]
        return float(steer), float(throttle)
