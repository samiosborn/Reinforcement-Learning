# main.py - Main entry point for CartPole DQN training (2D state version)
import os
from datetime import datetime
import torch
import matplotlib.pyplot as plt
import numpy as np
import config
from renderer import make_render_fn, make_input_fn
from train import Train
from ai_player import AIPlayer
from model import Model
from environment import Environment
from human_player import HumanPlayer

def run_game(mode=None,max_game_time=config.MAX_GAME_TIME,render_fn=None,input_fn=None,player=None,step_callback=None,verbose=config.DEBUG):
    env = Environment()
    state = env.reset()
    if verbose:
        print(f"[DEBUG] Initial state: {state}")

    # Player
    player = player or (AIPlayer() if mode == "ai" else HumanPlayer())

    done = False
    current_time = 0.0
    step_counter = 0

    # Main loop
    while not done and current_time <= max_game_time:
        # Apply next state
        prev_state = state.copy()
        action = player.choose_action(state, verbose=verbose)
        state, done, reward = env.step(action)

        # Debugging
        if verbose: 
            print(f"[STEP {step_counter}] Action: {action}, Reward: {reward}, Done: {done}")
            print(f"    From state: {prev_state}\n    To state: {state}")
        
        step_counter += 1

        # Store experience if ai
        if mode == "ai":
            player.store_experience(prev_state, action, reward, state, done)
        
        player.update_cumulative_reward(reward)

        if step_callback:
            step_callback()

        if render_fn:
            render_fn(state, current_time, player.cumulative_reward)

        if input_fn:
            input_fn()

        current_time += env.dt

# Runs evaluation episodes using a greedy policy (epsilon = 0)
def evaluate_policy(player, episodes, render=False, verbose=False):
    rewards = []
    episode_num = [0]
    episode_fn = lambda: episode_num[0]

    for ep in range(episodes):
        episode_num[0] = ep + 1
        run_game(
            mode="ai",
            max_game_time=config.MAX_GAME_TIME,
            render_fn=make_render_fn("ai", player=player, episode_num_fn=episode_fn) if render else None,
            input_fn=None,
            player=player,
            step_callback=None,
            verbose=verbose
        )
        rewards.append(player.cumulative_reward)
        player.reset_cumulative_reward()

    avg_reward = np.mean(rewards)
    if verbose: 
        print(f"\n[EVAL] Average reward over {episodes} episodes: {avg_reward:.2f}")

# Plots the reward received in each episode and saves to file
def plot_training_rewards(rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Rewards over Episodes")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_rewards.png")
    plt.close()

# Warn if action sampling in replay buffer is too skewed
def check_sampling_skew(buffer, verbose=False):
    dist = buffer.get_action_distribution()
    max_action = max(dist, key=dist.get)
    if verbose and dist[max_action] > 0.7:
        print(f"[WARN] Action '{max_action}' dominates buffer: {dist}")
    return dist

# Save model with timestamp
def save_model(model, verbose=False):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
    save_path = os.path.join(os.path.dirname(config.MODEL_SAVE_PATH), f"dqn_model_{timestamp}.pth")
    torch.save(model.state_dict(), save_path)
    if verbose: 
        print(f"Model saved to: {save_path}")

# Compute Q-value differences over a theta x theta_dot grid
def diagnose_q_values(model):
    # 2D array of Q[RIGHT] - Q[LEFT] values
    model.eval()
    with torch.no_grad():
        thetas = np.linspace(-config.THETA_THRESHOLD, config.THETA_THRESHOLD, 25)
        theta_dots = np.linspace(-config.THETA_DOT_THRESHOLD, config.THETA_DOT_THRESHOLD, 25)
        q_diffs = np.zeros((len(thetas), len(theta_dots)))

        for i, theta in enumerate(thetas):
            for j, theta_dot in enumerate(theta_dots):
                # Normalised 2D state
                norm_theta = theta / config.THETA_THRESHOLD
                norm_theta_dot = theta_dot / config.THETA_DOT_THRESHOLD
                state_tensor = torch.tensor([[norm_theta, norm_theta_dot]], dtype=torch.float32)
                q_values = model(state_tensor).cpu().numpy().flatten()
                q_diffs[i, j] = q_values[1] - q_values[0]

    return q_diffs

# Print min, max, and mean Q-difference stats
def print_diagnostic_results(q_diffs, verbose=False):
    if verbose: 
        print("[DIAGNOSTIC] Q-value difference stats (RIGHT - LEFT):")
        print(f"  Min diff:  {q_diffs.min():.4f}")
        print(f"  Max diff:  {q_diffs.max():.4f}")
        print(f"  Mean diff: {q_diffs.mean():.4f}")

# Training loop
def train_loop(player, trainer, episodes, batch_size, train_every_n_steps, episode_num, render_fn, input_fn, verbose=False):
    episode_rewards = []
    step_counter = 0

    def step_callback():
        nonlocal step_counter
        step_counter += 1
        if step_counter % train_every_n_steps == 0 and len(player.replay_buffer) >= batch_size:
            batch = player.replay_buffer.sample_balanced(batch_size)
            trainer.train_step(batch_size)

    for i in range(episodes):
        episode_num[0] = i + 1
        step_counter = 0
        player.reset_action_counter()

        run_game(
            mode="ai",
            max_game_time=config.MAX_GAME_TIME,
            render_fn=render_fn,
            input_fn=input_fn,
            player=player,
            step_callback=step_callback, 
            verbose=verbose
        )

        episode_rewards.append(player.cumulative_reward)
        player.reset_cumulative_reward()
        player.decay_epsilon()

        if (i + 1) % 10 == 0:
            check_sampling_skew(player.replay_buffer, verbose)
        if (i + 1) % 50 == 0:
            results = diagnose_q_values(trainer.model)
            print_diagnostic_results(results, verbose=verbose)

    return episode_rewards

def main():
    episodes = config.EPISODES
    batch_size = config.BATCH_SIZE
    train_every_n_steps = config.TRAIN_EVERY_N_STEPS
    render = config.RENDER
    evaluate_flag = config.EVALUATE_AFTER_TRAINING
    eval_episodes = config.EVAL_EPISODES
    verbose = config.DEBUG

    model = Model(config.INPUT_DIM, config.OUTPUT_DIM, verbose=verbose)

    if config.LOAD_MODEL and os.path.exists(config.MODEL_LOAD_PATH):
        model.load_state_dict(torch.load(config.MODEL_LOAD_PATH))
        if verbose: 
            print(f"[LOAD] Loaded model from {config.MODEL_LOAD_PATH}")
    elif config.LOAD_MODEL:
        print(f"[WARN] File not found to load model: {config.MODEL_LOAD_PATH}")

    player = AIPlayer(model=model)
    trainer = Train(model)
    trainer.set_replay_buffer(player.replay_buffer)

    with torch.no_grad():
        q_init = model(torch.zeros(1, config.INPUT_DIM))
        if verbose: 
            print(f"Initial Q-values: {q_init}")

    episode_num = [0]
    episode_fn = lambda: episode_num[0]
    render_fn = make_render_fn("ai", player=player, episode_num_fn=episode_fn) if render else None
    input_fn = make_input_fn() if render else None

    if config.TRAIN_MODEL:
        rewards = train_loop(
            player=player,
            trainer=trainer,
            episodes=episodes,
            batch_size=batch_size,
            train_every_n_steps=train_every_n_steps,
            episode_num=episode_num,
            render_fn=render_fn,
            input_fn=input_fn, 
            verbose=verbose
        )

        save_model(model, verbose)

        if config.TRACK_Q_VALUES:
            trainer.plot_q_history()

        plot_training_rewards(rewards)
        if verbose: 
            avg_training_reward = np.mean(rewards)
            print(f"\n[TRAIN] Average reward over {len(rewards)} episodes: {avg_training_reward:.2f}")

    if evaluate_flag:
        if verbose: 
            print("\nFinal Evaluation")
        player.epsilon = 0.0
        evaluate_policy(player, episodes=eval_episodes, render=render, verbose=verbose)

if __name__ == "__main__":
    main()
