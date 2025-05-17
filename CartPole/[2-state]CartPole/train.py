# train.py — DQN Trainer for CartPole with configurable updates

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import config

class Train:
    # Initialises the trainer with the model, target model, optimiser, and device setup
    def __init__(self, model):
        self.gamma = config.GAMMA
        self.lr = config.LR
        self.tau = config.SOFT_UPDATE_TAU
        self.step_count = 0

        # Initialise policy and target networks
        self.model = model
        self.target_model = type(model)(model.input_dim, model.output_dim)
        self.target_model.load_state_dict(model.state_dict())

        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)

        self.device = next(self.model.parameters()).device
        self.model.to(self.device)
        self.target_model.to(self.device)

        self.track_q_values_flag = config.TRACK_Q_VALUES
        self.q_value_history = {'left': [], 'right': [], 'diff': []} if self.track_q_values_flag else None

        self.replay_buffer = None

        # Thresholds used for normalisation of 2D state
        self.thresholds = [
            config.THETA_THRESHOLD,
            config.THETA_DOT_THRESHOLD,
        ]

    # Assigns the replay buffer used for training
    def set_replay_buffer(self, buffer):
        self.replay_buffer = buffer

    # Performs a single training step on a sampled batch
    def train_step(self, batch_size=32):
        if self.replay_buffer is None or len(self.replay_buffer) == 0:
            return

        # Use balanced sampling if action distribution is skewed
        if hasattr(self.replay_buffer, 'get_action_distribution'):
            action_dist = self.replay_buffer.get_action_distribution()
            skewed = max(action_dist.values()) > 0.7
        else:
            skewed = False

        if skewed and hasattr(self.replay_buffer, 'sample_balanced'):
            batch = self.replay_buffer.sample_balanced(batch_size)
        else:
            sample_result = self.replay_buffer.sample(batch_size)
            batch = sample_result[0] if isinstance(sample_result, tuple) else sample_result

        if len(batch) == 0:
            return

        # Unpack transitions
        states, actions, rewards, next_states, dones = zip(*batch)
        states_tensor = torch.from_numpy(np.stack([self.normalise_state(s) for s in states])).float().to(self.device)
        next_states_tensor = torch.from_numpy(np.stack([self.normalise_state(s) for s in next_states])).float().to(self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # Q-values for chosen actions
        q_values_all = self.model(states_tensor)
        q_values = q_values_all.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

        # Double DQN: use online model to select best next action
        with torch.no_grad():
            next_actions = self.model(next_states_tensor).argmax(dim=1)
            next_q_values = self.target_model(next_states_tensor)
            next_q_selected = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)

            targets = rewards_tensor + self.gamma * next_q_selected * (1.0 - dones_tensor)
            targets = torch.clamp(targets, min=config.TD_TARGET_CLIP_MIN, max=config.TD_TARGET_CLIP_MAX)

        # Compute Huber loss and update model
        loss = F.smooth_l1_loss(q_values, targets)
        self.optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimiser.step()

        # Target network update strategy (soft or hard)
        if config.HARD_UPDATE:
            if self.step_count % config.HARD_UPDATE_N == 0:
                self.target_model.load_state_dict(self.model.state_dict())
        else:
            self.soft_target_update()

        self.step_count += 1

        # Record Q-value statistics
        if self.track_q_values_flag:
            self.track_q_values(q_values_all)

    # Performs soft Polyak averaging to update the target model
    def soft_target_update(self):
        for target_param, local_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    # Normalises the 2D input state (theta, theta_dot)
    def normalise_state(self, state):
        theta, theta_dot = state[-2], state[-1]
        return np.tanh([
            theta / self.thresholds[0],
            theta_dot / self.thresholds[1]
        ])

    # Tracks and records average Q-values for each action
    def track_q_values(self, q_values_all):
        mean_q_left = q_values_all[:, 0].mean().item()
        mean_q_right = q_values_all[:, 1].mean().item()
        diff = mean_q_right - mean_q_left
        self.q_value_history['left'].append(mean_q_left)
        self.q_value_history['right'].append(mean_q_right)
        self.q_value_history['diff'].append(diff)

    # Saves Q-value trends to PNG
    def plot_q_history(self):
        if not self.track_q_values_flag:
            return

        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        plt.plot(self.q_value_history['left'], label='LEFT')
        plt.plot(self.q_value_history['right'], label='RIGHT')
        plt.title('Average Q-values during Training')
        plt.xlabel('Training Step')
        plt.ylabel('Average Q-value')
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(self.q_value_history['diff'], color='purple')
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.title('Q-value Difference (RIGHT - LEFT)')
        plt.xlabel('Training Step')
        plt.ylabel('Difference')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('q_value_history.png')
        plt.close()
