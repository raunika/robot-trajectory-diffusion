"""
model.py

Minimal MLP denoiser for diffusion-style trajectory smoothing.
"""

import torch
import torch.nn as nn


class TrajectoryDenoiser(nn.Module):
    """
    Simple MLP that predicts noise added to a joint trajectory.
    """

    def __init__(self, traj_dim: int, hidden_dim: int = 256):
        """
        Args:
            traj_dim: flattened trajectory dimension (T * 2)
            hidden_dim: size of hidden layers
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(traj_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, traj_dim),
        )

    def forward(self, x):
        return self.net(x)


def test_model():
    """
    Simple sanity check.
    """
    T = 200
    traj_dim = T * 2

    model = TrajectoryDenoiser(traj_dim)

    dummy_input = torch.randn(4, traj_dim)  # batch size 4
    output = model(dummy_input)

    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)


if __name__ == "__main__":
    test_model()
