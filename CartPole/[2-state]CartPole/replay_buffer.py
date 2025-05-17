import torch
import numpy as np
import random

class ReplayBuffer:
    # Uniform Experience Replay Buffer with optional balanced action sampling
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.action_counts = {0: 0, 1: 0}

    def push(self, transition):
        s, a, r, s_next, done = transition
        self.action_counts[a] += 1

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return []
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def sample_balanced(self, batch_size):
        # Sample a balanced batch with equal LEFT (0) and RIGHT (1) actions, if possible.
        left = [t for t in self.buffer if t[1] == 0]
        right = [t for t in self.buffer if t[1] == 1]

        half = batch_size // 2

        if len(left) >= half and len(right) >= (batch_size - half):
            sample_left = random.sample(left, half)
            sample_right = random.sample(right, batch_size - half)
            return sample_left + sample_right
        else:
            # Fallback to uniform if not enough samples on one side
            return self.sample(batch_size)

    def get_action_distribution(self):
        total = sum(self.action_counts.values())
        return {k: v / total if total else 0.0 for k, v in self.action_counts.items()}

    def __len__(self):
        return len(self.buffer)
