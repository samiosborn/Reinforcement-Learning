# environment.py
import numpy as np
import math
import config

class Environment:
    # Simulates CartPole dynamics using Euler integration
    def __init__(self):
        # Physics constants
        self.cart_force = config.CART_FORCE
        self.m_cart = config.MASS_CART
        self.m_pole = config.MASS_POLE
        self.length = config.POLE_LENGTH_HALF
        self.g = config.GRAVITY
        self.dt = config.TIME_STEP

        # Termination thresholds
        self.x_threshold = config.X_THRESHOLD
        self.theta_threshold = config.THETA_THRESHOLD
        self.theta_dot_threshold = config.THETA_DOT_THRESHOLD

        # Reward shaping thresholds
        self.theta_reward = config.THETA_REWARD
        self.theta_dot_reward = config.THETA_DOT_REWARD

        # Random perturbations on reset
        self.x_perturb = config.X_PERTURBATIONS
        self.x_dot_perturb = config.X_DOT_PERTURBATIONS
        self.theta_perturb = config.THETA_PERTURBATIONS
        self.theta_dot_perturb = config.THETA_DOT_PERTURBATIONS

        self.reset()

    @property
    def state(self):
        return np.array([self.x, self.theta, self.x_dot, self.theta_dot], dtype=np.float32)

    @state.setter
    def state(self, new_state):
        self.x, self.theta, self.x_dot, self.theta_dot = new_state

    def reset(self):
        self.x = np.random.uniform(-self.x_perturb, self.x_perturb)
        self.x_dot = np.random.uniform(-self.x_dot_perturb, self.x_dot_perturb)
        self.theta = np.random.uniform(-self.theta_perturb, self.theta_perturb)
        self.theta_dot = np.random.uniform(-self.theta_dot_perturb, self.theta_dot_perturb)
        self.done = False
        return self.state

    def step(self, action):
        force = action * self.cart_force
        x_ddot, theta_ddot = self.dynamics(force)

        # Euler integration
        self.x_dot += x_ddot * self.dt
        self.theta_dot += theta_ddot * self.dt
        self.x += self.x_dot * self.dt
        self.theta += self.theta_dot * self.dt

        # Check termination
        self.done = (
            abs(self.x) > self.x_threshold or
            abs(self.theta) > self.theta_threshold
        )

        # Reward shaping
        if self.done:
            reward = config.REWARD_CLIP_MIN
            return self.state, self.done, reward
        else:
            # Normalised penalties
            angle_penalty = abs(self.theta) / self.theta_threshold
            angle_vel_penalty = min(abs(self.theta_dot) / self.theta_dot_threshold, 1.0)

            # Base shaped reward
            reward = 1.0 - 0.7 * angle_penalty - 0.3 * angle_vel_penalty

            # Sharp penalty if nearing terminal state
            if abs(self.theta) > 0.75 * self.theta_threshold:
                reward *= 0.2
            elif abs(self.theta) > 0.5 * self.theta_threshold:
                reward *= 0.5

            # Penalise high angular velocity
            reward -= 0.1 * abs(self.theta_dot)

            # Final clipping
            reward = min(max(reward, config.REWARD_CLIP_MIN), config.REWARD_CLIP_MAX)

        return self.state, self.done, reward

    # Physical dynamics
    def dynamics(self, force):
        m_c = self.m_cart
        m_p = self.m_pole
        l = self.length
        total_mass = m_c + m_p
        theta = self.theta
        theta_dot = self.theta_dot
        g = self.g

        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)

        temp = (force + m_p * l * theta_dot**2 * sin_theta) / total_mass
        numerator = g * sin_theta - cos_theta * temp
        denominator = l * (4.0 / 3.0 - m_p * cos_theta**2 / total_mass)
        theta_ddot = numerator / denominator
        x_ddot = temp - m_p * l * theta_ddot * cos_theta / total_mass

        return x_ddot, theta_ddot