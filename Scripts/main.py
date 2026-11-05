import pygame

import config


def main() -> None:
    pygame.init()
    window = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
    pygame.display.set_caption(config.WINDOW_TITLE)

    clock = pygame.time.Clock()
    running = True
    while running:
        clock.tick(config.FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        window.fill(config.BACKGROUND_COLOR)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
