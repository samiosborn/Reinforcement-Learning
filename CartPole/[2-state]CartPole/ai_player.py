# ai_player.py
import random
import torch
import numpy as np
import config
from replay_buffer import ReplayBuffer

class AIPlayer:
    def __init__(self, model=None):
        # DQN model (policy network)
        self.model = model

        # Epsilon-greedy exploration parameters
        self.epsilon = config.EPS_START
        self.epsilon_min = config.EPS_MIN
        self.epsilon_decay = config.EPS_DECAY
        self.total_steps = 0
        self.warmup_steps = config.WARMUP_STEPS
        # Cumulative reward tracker
        self.cumulative_reward = 0
        # Discrete action space: -1 = LEFT, +1 = RIGHT
        self.actions = [-1, 1]
        # Maps action values to output indices
        self.index_map = {-1: 0, 1: 1}
        # Used to alternate during warmup
        self.last_warmup_action_index = 1
        # Tracks how often each action was chosen
        self.action_counter = {-1: 0, 1: 0}
        # Thresholds for normalising state
        self.thresholds = [config.THETA_THRESHOLD, config.THETA_DOT_THRESHOLD]
        # Experience replay buffer
        self.replay_buffer = ReplayBuffer(capacity=config.REPLAY_CAPACITY)

    # Convert action value to output index
    def to_index(self, action_value):
        return self.index_map[action_value]

    # Convert output index to action value
    def to_action(self, index):
        return self.actions[index]

    # Normalise [theta, theta_dot] by their thresholds
    def normalise_state(self, state):
        theta, theta_dot = state
        return np.array([
            theta / self.thresholds[0],
            theta_dot / self.thresholds[1],
        ])

    # Store a transition tuple in the replay buffer
    def store_experience(self, state, action, reward, next_state, done):
        index = self.to_index(action)
        transition = (state, index, reward, next_state, done)
        self.replay_buffer.push(transition)

    # Select an action given a state
    def choose_action(self, state, verbose=False):
        # Extract only theta and theta_dot from full state
        theta, theta_dot = state[2], state[3]
        norm_state = self.normalise_state((theta, theta_dot))
        self.total_steps += 1

        # Warmup phase
        if self.total_steps <= self.warmup_steps:
            # Alternate left and right
            index = 1 - self.last_warmup_action_index
            self.last_warmup_action_index = index
            decision_type = "warmup"

        # Exploration phase
        elif random.random() < self.epsilon or self.model is None:
            # Most of the time, try to balance left / right actions
            if sum(self.action_counter.values()) > 0 and random.random() < 0.7:
                ratio = self.action_counter[-1] / sum(self.action_counter.values())
                index = 1 if ratio > 0.5 else 0
            else:
                index = random.randint(0, 1)
            decision_type = "explore"

        # Exploitation phase
        else:
            # Convert state to input tensor
            state_tensor = torch.tensor(norm_state, dtype=torch.float32).unsqueeze(0).to(next(self.model.parameters()).device)
            with torch.no_grad():
                q_values = self.model(state_tensor).cpu().numpy().flatten()

            # Add small Gaussian noise to Q-values to avoid rigid bias
            noisy_q_values = q_values + np.random.normal(0, 0.01, size=2)
            index = int(np.argmax(noisy_q_values))
            decision_type = "exploit"

        # Convert index to action and track frequency
        action = self.to_action(index)
        self.action_counter[action] += 1

        # Logging
        if verbose:
            print(f"[{decision_type.upper()}] Chose action {action} (index {index}) at epsilon={self.epsilon:.3f}")
            if decision_type == "exploit":
                print(f"Q[LEFT] = {q_values[0]:.4f}, Q[RIGHT] = {q_values[1]:.4f}, diff = {q_values[1] - q_values[0]:.4f}")
            total = sum(self.action_counter.values())
            left_pct = 100 * self.action_counter[-1] / total if total else 0
            right_pct = 100 * self.action_counter[1] / total if total else 0
            print(f"Action history: LEFT: {left_pct:.1f}%, RIGHT: {right_pct:.1f}%")

        return action

    # Decay epsilon
    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # Accumulate reward within episode
    def update_cumulative_reward(self, reward):
        self.cumulative_reward += reward
        return self.cumulative_reward

    # Reset cumulative reward
    def reset_cumulative_reward(self):
        self.cumulative_reward = 0

    # Reset action counter
    def reset_action_counter(self):
        self.action_counter = {-1: 0, 1: 0}

    # Clear memory buffer
    def clear_memory(self):
        self.replay_buffer = ReplayBuffer(capacity=config.REPLAY_CAPACITY)
