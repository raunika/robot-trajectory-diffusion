"""
train.py

End-to-end training script:
- generate joint-space trajectories (2-DOF)
- apply diffusion forward noise
- train MLP denoiser to predict epsilon
- reconstruct x0 estimate and visualize smoothing

Run:
  source .venv/bin/activate
  python train.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from arm import Planar2DOF, generate_jerky_joint_trajectory
from model import TrajectoryDenoiser
from diffusion import Diffusion

import matplotlib.pyplot as plt


# ----------------------------
# Config
# ----------------------------

@dataclass
class Config:
    # trajectory
    T_steps: int = 200         # timesteps per trajectory
    n_traj: int = 2048         # dataset size
    angle_scale: float = 1.0

    # diffusion
    diffusion_T: int = 100
    beta_start: float = 1e-4
    beta_end: float = 2e-2

    # training
    batch_size: int = 64
    epochs: int = 8
    lr: float = 1e-3
    hidden_dim: int = 256
    seed: int = 42

    # outputs
    results_dir: str = "results"
    out_prefix: str = "results/diffusion_denoise"


# ----------------------------
# Dataset
# ----------------------------

class TrajectoryDataset(Dataset):
    def __init__(self, cfg: Config):
        rng = np.random.default_rng(cfg.seed)

        trajs = []
        for i in range(cfg.n_traj):
            # Different seed per trajectory
            seed_i = int(rng.integers(0, 1_000_000))
            thetas = generate_jerky_joint_trajectory(T=cfg.T_steps, seed=seed_i, scale=cfg.angle_scale)
            trajs.append(thetas.astype(np.float32))

        self.trajs = np.stack(trajs, axis=0)  # (N, T, 2)

    def __len__(self):
        return self.trajs.shape[0]

    def __getitem__(self, idx):
        # return flattened (T*2,)
        x0 = self.trajs[idx].reshape(-1)
        return x0


# ----------------------------
# Helpers
# ----------------------------

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def estimate_x0_from_eps(xt: torch.Tensor, eps_hat: torch.Tensor, alpha_bar_t: torch.Tensor) -> torch.Tensor:
    """
    One-step reconstruction of x0 (not full reverse diffusion):
      x0_hat = (xt - sqrt(1 - alpha_bar_t) * eps_hat) / sqrt(alpha_bar_t)

    alpha_bar_t should be shape (B, 1).
    """
    return (xt - torch.sqrt(1.0 - alpha_bar_t) * eps_hat) / torch.sqrt(alpha_bar_t)


def plot_before_after(
    x0_true: np.ndarray,
    xt_noisy: np.ndarray,
    x0_hat: np.ndarray,
    cfg: Config,
):
    """
    Save:
      - joints_before_after.png
      - ee_path_before_after.png
    """
    os.makedirs(cfg.results_dir, exist_ok=True)

    T = cfg.T_steps
    true_traj = x0_true.reshape(T, 2)
    noisy_traj = xt_noisy.reshape(T, 2)
    hat_traj = x0_hat.reshape(T, 2)

    # Joint plots
    t = np.arange(T)

    plt.figure()
    plt.plot(t, true_traj[:, 0], label="theta1 true")
    plt.plot(t, noisy_traj[:, 0], label="theta1 noisy")
    plt.plot(t, hat_traj[:, 0], label="theta1 denoised")
    plt.xlabel("timestep")
    plt.ylabel("rad")
    plt.title("Joint 1: true vs noisy vs denoised")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{cfg.out_prefix}_joint1.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(t, true_traj[:, 1], label="theta2 true")
    plt.plot(t, noisy_traj[:, 1], label="theta2 noisy")
    plt.plot(t, hat_traj[:, 1], label="theta2 denoised")
    plt.xlabel("timestep")
    plt.ylabel("rad")
    plt.title("Joint 2: true vs noisy vs denoised")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{cfg.out_prefix}_joint2.png", dpi=200)
    plt.close()

    # End-effector paths
    arm = Planar2DOF(1.0, 1.0)
    xy_true = arm.fk_batch(true_traj)
    xy_noisy = arm.fk_batch(noisy_traj)
    xy_hat = arm.fk_batch(hat_traj)

    plt.figure()
    plt.plot(xy_true[:, 0], xy_true[:, 1], label="true")
    plt.plot(xy_noisy[:, 0], xy_noisy[:, 1], label="noisy")
    plt.plot(xy_hat[:, 0], xy_hat[:, 1], label="denoised")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("End-effector path: true vs noisy vs denoised")
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{cfg.out_prefix}_ee_path.png", dpi=200)
    plt.close()


# ----------------------------
# Main training
# ----------------------------

def main():
    cfg = Config()
    set_seed(cfg.seed)

    os.makedirs(cfg.results_dir, exist_ok=True)

    device = "cpu"  # keep this repo simple + deterministic
    print("Using device:", device)

    # Data
    ds = TrajectoryDataset(cfg)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    traj_dim = cfg.T_steps * 2
    model = TrajectoryDenoiser(traj_dim=traj_dim, hidden_dim=cfg.hidden_dim).to(device)
    diff = Diffusion(T=cfg.diffusion_T, beta_start=cfg.beta_start, beta_end=cfg.beta_end, device=device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(cfg.epochs):
        pbar = tqdm(dl, desc=f"epoch {epoch+1}/{cfg.epochs}")
        running = 0.0
        for x0 in pbar:
            x0 = x0.to(device)  # (B, D)
            t = diff.sample_timesteps(x0.shape[0])  # (B,)

            xt, eps = diff.add_noise(x0, t)  # both (B, D)

            eps_hat = model(xt)
            loss = loss_fn(eps_hat, eps)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running += float(loss.item())
            pbar.set_postfix(loss=running / (pbar.n + 1))

    # ----------------------------
    # Demo: pick one trajectory and show denoise effect at a moderately noisy t
    # ----------------------------
    model.eval()
    with torch.no_grad():
        x0_true = ds[0]  # (D,) numpy float32
        x0_true_t = torch.from_numpy(x0_true).unsqueeze(0).to(device)  # (1, D)

        # choose a mid/high noise level (e.g., 70% into schedule)
        t_demo = torch.tensor([int(cfg.diffusion_T * 0.7)], device=device)
        xt, eps = diff.add_noise(x0_true_t, t_demo)

        eps_hat = model(xt)

        alpha_bar = diff.alpha_bars[t_demo].unsqueeze(1)  # (1, 1)
        x0_hat = estimate_x0_from_eps(xt, eps_hat, alpha_bar)  # (1, D)

        # convert to numpy for plotting
        xt_np = xt.squeeze(0).cpu().numpy()
        x0_hat_np = x0_hat.squeeze(0).cpu().numpy()

        plot_before_after(x0_true, xt_np, x0_hat_np, cfg)

    print("Saved:")
    print(f"  {cfg.out_prefix}_joint1.png")
    print(f"  {cfg.out_prefix}_joint2.png")
    print(f"  {cfg.out_prefix}_ee_path.png")


if __name__ == "__main__":
    main()
