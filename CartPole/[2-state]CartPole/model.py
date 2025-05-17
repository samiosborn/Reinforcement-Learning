# model.py
import torch
import torch.nn as nn
import numpy as np
import config

class Model(nn.Module):
    # Feedforward NN
    def __init__(self, input_dim, output_dim, hidden_dim_1=None, hidden_dim_2=None, verbose=False):
        super(Model, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim_1 = hidden_dim_1 if hidden_dim_1 else config.HIDDEN1
        self.hidden_dim_2 = hidden_dim_2 if hidden_dim_2 else config.HIDDEN2
        self.verbose = verbose

        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim_1)
        self.fc2 = nn.Linear(self.hidden_dim_1, self.hidden_dim_2)
        self.fc3 = nn.Linear(self.hidden_dim_2, self.output_dim)

        self._init_weights()

    def _init_weights(self):
        # Initialise hidden weights with He init and output with Xavier
        for layer in [self.fc1, self.fc2]:
            nn.init.kaiming_normal_(layer.weight, nonlinearity='leaky_relu')
            nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

        # Check symmetry at zero input
        with torch.no_grad():
            zero_input = torch.zeros(100, self.input_dim)
            test_output = self.forward(zero_input)
            diff = test_output[:, 0] - test_output[:, 1]
            mean_diff = diff.mean().item()
            if self.verbose: 
                print(f"[Init Symmetry Check] Mean Q[LEFT - RIGHT] diff at zero input: {mean_diff:.6f}")
                if abs(mean_diff) > 1e-3:
                    print("[Init Warning] Q-values may be initialised asymmetrically!")

    def forward(self, x):
        # Forward pass through network
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        x = x.to(next(self.parameters()).device)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        return self.fc3(x)
