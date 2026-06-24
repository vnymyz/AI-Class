"""
Same MLP architecture as mlp_numpy.py, but using PyTorch.
Compare against the numpy version: forward/backward/update here are
all handled automatically by autograd + optimizer.
"""
import torch
import torch.nn as nn


class TorchMLP(nn.Module):
    def __init__(self, input_size=2, hidden_size=4, output_size=1):
        super().__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.hidden(x))
        x = self.activation(self.output(x))
        return x
