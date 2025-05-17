# renderer.py — Combines rendering and input handling logic
import config
import pygame
import math


class Renderer:
    # Class-level constants from config
    scale = config.METRES_TO_PIXELS
    cart_width = config.CART_WIDTH * scale
    cart_height = config.CART_HEIGHT * scale
    pole_length = 2 * config.POLE_LENGTH_HALF * scale
    pole_width = config.POLE_WIDTH * scale
    track_width = 2 * config.X_THRESHOLD * scale

    screen_width = round(1.5 * (2 * pole_width + track_width))
    screen_height = round(2.0 * (cart_height + pole_length))

    pivot_y = max(0.70 * screen_height, cart_height)
    cart_top_left_y = pivot_y

    def __init__(self):
        # Instance-level dynamic state
        self.pivot_x = self.screen_width / 2
        self.cart_top_left_x = self.pivot_x - self.cart_width / 2
        self.pole_x_tip = self.pivot_x
        self.pole_y_tip = self.pivot_y + self.pole_length

    def cart_render(self, surface, x):
        x_pixels = x * self.scale
        self.pivot_x = x_pixels + self.screen_width / 2
        self.cart_top_left_x = self.pivot_x - self.cart_width / 2
        pygame.draw.rect(
            surface,
            (0, 0, 0),
            (
                round(self.cart_top_left_x),
                round(self.cart_top_left_y),
                round(self.cart_width),
                round(self.cart_height)
            )
        )

    def pole_render(self, surface, x, theta):
        self.pole_x_tip = self.pivot_x + self.pole_length * math.sin(theta)
        self.pole_y_tip = self.pivot_y - self.pole_length * math.cos(theta)
        pygame.draw.line(
            surface,
            (0, 0, 0),
            (round(self.pivot_x), round(self.pivot_y)),
            (round(self.pole_x_tip), round(self.pole_y_tip)),
            round(self.pole_width)
        )


# Factory to generate rendering function
def make_render_fn(mode, player=None, episode_num_fn=None):
    pygame.init()
    renderer = Renderer()

    screen = pygame.display.set_mode((renderer.screen_width, renderer.screen_height))
    pygame.display.set_caption('CartPole Simulation')
    font = pygame.font.SysFont(None, 30)

    def render_fn(state, time_elapsed, score):
        screen.fill((255, 255, 255))

        # Draw cart and pole
        renderer.cart_render(screen, state[0])
        renderer.pole_render(screen, state[0], state[1])

        # Line 0: Episode number at the top
        if episode_num_fn:
            episode_text = font.render(f"Episode: {episode_num_fn()}", True, (0, 0, 0))
            screen.blit(episode_text, episode_text.get_rect(center=(renderer.screen_width // 2, int(0.07 * renderer.screen_height))))

        # Line 1: Player type
        mode_text = font.render(f"Player: {mode}", True, (0, 0, 0))
        screen.blit(mode_text, mode_text.get_rect(center=(renderer.screen_width // 2, int(0.82 * renderer.screen_height))))

        # Line 2: Epsilon + training/eval status (for AI player)
        if mode == "ai" and player and hasattr(player, "epsilon"):
            epsilon_display = f"{player.epsilon:.3f}"
            mode_status = "Evaluating" if player.epsilon <= player.epsilon_min + 1e-3 else "Training"
            train_eval_text = font.render(f"{mode_status} (ε = {epsilon_display})", True, (0, 0, 0))
            screen.blit(train_eval_text, train_eval_text.get_rect(center=(renderer.screen_width // 2, int(0.86 * renderer.screen_height))))

        # Line 3: Time and score
        score_text = font.render(f"Time: {round(time_elapsed, 3)} | Score: {round(score)}", True, (0, 0, 0))
        banner_text = font.render("Press 'q' to quit or 'p' to pause", True, (0, 0, 0))

        screen.blit(score_text, score_text.get_rect(center=(renderer.screen_width // 2, int(0.15 * renderer.screen_height))))
        screen.blit(banner_text, banner_text.get_rect(center=(renderer.screen_width // 2, int(0.95 * renderer.screen_height))))

        pygame.display.flip()

    return render_fn

# Factory to generate input function
def make_input_fn():
    def input_fn():
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    pygame.quit()
                    exit()
                elif event.key == pygame.K_p:
                    _pause_game()
    return input_fn

# Pause logic with overlay text
def _pause_game():
    font = pygame.font.SysFont(None, 30)
    screen = pygame.display.get_surface()
    renderer = Renderer()
    paused = True

    while paused:
        screen.fill((255, 255, 255))
        pause_text = font.render("Paused. Press 'p' to resume or 'q' to quit.", True, (0, 0, 0))
        screen.blit(pause_text, pause_text.get_rect(center=(renderer.screen_width // 2, renderer.screen_height // 2)))
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    paused = False
                elif event.key == pygame.K_q:
                    pygame.quit()
                    exit()
