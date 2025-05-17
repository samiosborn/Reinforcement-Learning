# config.py
import math

# Model Paths
MODEL_SAVE_PATH = "checkpoints/dqn_model.pth"
MODEL_LOAD_PATH = "checkpoints/dqn_model_2025-05-15_17-27-51.pth"

# Flags
RENDER = True
TRAIN_MODEL = True
TRACK_Q_VALUES = False
EVALUATE_AFTER_TRAINING = True
LOAD_MODEL = False
DEBUG = True
HARD_UPDATE = True

# Target Update Settings
HARD_UPDATE_N = 250

# Training Parameters
MAX_GAME_TIME = 7.5
EPISODES = 10
BATCH_SIZE = 128
GAMMA = 0.97
LR = 0.0005
WARMUP_STEPS = 1000
SOFT_UPDATE_TAU = 0.005
REPLAY_CAPACITY = 50000
TRAIN_EVERY_N_STEPS = 2

# Epsilon-Greedy Exploration
EPS_START = 1.0
EPS_MIN = 0.05
EPS_DECAY = 0.999

# Evaluation Settings
EVAL_EPISODES = 10

# Render Settings
METRES_TO_PIXELS = 100

# Cart Settings
CART_WIDTH = 0.75
CART_HEIGHT = 0.2
MASS_CART = 10.0
CART_FORCE = 50.0

# Pole Settings
MASS_POLE = 1.0
POLE_LENGTH_HALF = 1.0
POLE_WIDTH = 0.1

# Physics Thresholds
X_THRESHOLD = 2.0
X_DOT_THRESHOLD = 3.0
THETA_THRESHOLD = math.radians(60)
THETA_DOT_THRESHOLD = 5.0

# Reward Shaping Thresholds
THETA_REWARD = math.radians(5)
THETA_DOT_REWARD = 0.75
REWARD_CLIP_MIN = -5.0
REWARD_CLIP_MAX = 1.0
TD_TARGET_CLIP_MIN = -20.0
TD_TARGET_CLIP_MAX = 100.0

# Gravity and Simulation Settings
GRAVITY = 4.0
TIME_STEP = 0.01

# State / Action Space
INPUT_DIM = 2
OUTPUT_DIM = 2

# Initial State Perturbations
X_PERTURBATIONS = 0.10
X_DOT_PERTURBATIONS = 0.05
THETA_PERTURBATIONS = 0.05
THETA_DOT_PERTURBATIONS = 0.05

# Model Architecture
HIDDEN1 = 256
HIDDEN2 = 128
