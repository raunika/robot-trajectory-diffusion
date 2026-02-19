"""
diffusion.py

Minimal diffusion-style utilities for trajectory denoising:
- linear beta schedule
- forward noising: x0 -> xt
- loss target: predict epsilon (noise)

This is intentionally lightweight (enough for a solid day-1 repo).
"""

from __future__ import annotations
import torch


def linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    """
    Linear schedule from beta_start to beta_end over T steps.
    Returns: betas shape (T,)
    """
    return torch.linspace(beta_start, beta_end, T)


class Diffusion:
    def __init__(self, T: int = 100, beta_start: float = 1e-4, beta_end: float = 2e-2, device: str | None = None):
        self.T = int(T)
        self.device = device or "cpu"

        self.betas = linear_beta_schedule(self.T, beta_start, beta_end).to(self.device)              # (T,)
        self.alphas = (1.0 - self.betas).to(self.device)                                            # (T,)
        self.alpha_bars = torch.cumprod(self.alphas, dim=0).to(self.device)                          # (T,)

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        """
        Sample integer timesteps t in [0, T-1] for each sample.
        """
        return torch.randint(low=0, high=self.T, size=(batch_size,), device=self.device)

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward noising: xt = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps

        Args:
            x0: (B, D) clean trajectories
            t:  (B,) integer timesteps

        Returns:
            xt: (B, D) noised trajectories
            eps: (B, D) the noise that was added (training target)
        """
        if x0.ndim != 2:
            raise ValueError(f"x0 must be (B, D). Got {tuple(x0.shape)}")
        if t.ndim != 1 or t.shape[0] != x0.shape[0]:
            raise ValueError(f"t must be (B,) with same batch as x0. Got t={tuple(t.shape)}, x0={tuple(x0.shape)}")

        eps = torch.randn_like(x0)

        alpha_bar_t = self.alpha_bars[t].unsqueeze(1)  # (B, 1)
        xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * eps
        return xt, eps


def test_diffusion():
    B, D = 4, 400
    diff = Diffusion(T=50)
    x0 = torch.randn(B, D, device=diff.device)
    t = diff.sample_timesteps(B)
    xt, eps = diff.add_noise(x0, t)
    print("device:", diff.device)
    print("x0:", x0.shape, "t:", t.shape, "xt:", xt.shape, "eps:", eps.shape)


if __name__ == "__main__":
    test_diffusion()
