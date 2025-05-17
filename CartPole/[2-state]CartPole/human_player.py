# human_player.py
import pygame

class HumanPlayer():
    
    # Initialise
    def __init__(self):
        # Cumulative reward
        self.cumulative_reward = 0

    # Update cumulative reward
    def update_cumulative_reward(self, reward):
        self.cumulative_reward += reward
        return self.cumulative_reward
    
    # Reset cumulative reward
    def reset_cumulative_reward(self):
        self.cumulative_reward = 0
        return self.cumulative_reward

    # Choose action from keyboard input 
    def choose_action(self, state, verbose):
        # Get current state of all keyboard keys
        keys = pygame.key.get_pressed()
        # If left arrow key is pressed
        if keys[pygame.K_LEFT]:
            # Move left (action = -1)
            return -1
        # If right arrow key is pressed
        elif keys[pygame.K_RIGHT]:
            # Move right (action = 1)
            return 1
        else:
            # If no arrow key is pressed, stay still (action = 0)
            return 0


