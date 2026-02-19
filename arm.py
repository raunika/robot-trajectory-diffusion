"""
arm.py

2-DOF planar manipulator utilities:
- forward kinematics
- trajectory -> end-effector path
- simple plotting helpers

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Planar2DOF:
    """2-link planar arm."""
    l1: float = 1.0
    l2: float = 1.0

    def fk(self, theta1: float, theta2: float) -> Tuple[float, float]:
        """
        Forward kinematics for 2-DOF planar arm.

        Args:
            theta1, theta2: joint angles [rad]

        Returns:
            (x, y): end-effector position
        """
        x = self.l1 * np.cos(theta1) + self.l2 * np.cos(theta1 + theta2)
        y = self.l1 * np.sin(theta1) + self.l2 * np.sin(theta1 + theta2)
        return float(x), float(y)

    def fk_batch(self, thetas: np.ndarray) -> np.ndarray:
        """
        Batch forward kinematics.

        Args:
            thetas: array of shape (T, 2) with columns [theta1, theta2]

        Returns:
            xy: array of shape (T, 2)
        """
        thetas = np.asarray(thetas, dtype=np.float64)
        if thetas.ndim != 2 or thetas.shape[1] != 2:
            raise ValueError(f"thetas must have shape (T, 2). Got {thetas.shape}")

        t1 = thetas[:, 0]
        t2 = thetas[:, 1]

        x = self.l1 * np.cos(t1) + self.l2 * np.cos(t1 + t2)
        y = self.l1 * np.sin(t1) + self.l2 * np.sin(t1 + t2)

        return np.stack([x, y], axis=1)


def generate_jerky_joint_trajectory(
    T: int = 200,
    seed: int = 0,
    scale: float = 1.0,
) -> np.ndarray:
    """
    Generate a deliberately jerky joint trajectory for testing.

    This is NOT physically optimal; it's meant to be "ugly" so denoising later is obvious.

    Returns:
        thetas: (T, 2) in radians
    """
    rng = np.random.default_rng(seed)

    # Random walk + occasional jumps
    thetas = np.cumsum(rng.normal(0.0, 0.08 * scale, size=(T, 2)), axis=0)

    jump_idx = rng.integers(low=20, high=T - 20, size=6)
    for idx in jump_idx:
        thetas[idx:] += rng.normal(0.0, 0.8 * scale, size=(1, 2))

    # Keep angles within a sane range for display
    thetas = np.clip(thetas, -np.pi, np.pi)
    return thetas


def plot_trajectory_and_path(
    thetas: np.ndarray,
    arm: Planar2DOF,
    save_path_prefix: str = "results/arm_demo",
) -> None:
    """
    Plot joint angles over time and end-effector path.

    Saves:
      - {prefix}_joints.png
      - {prefix}_path.png
    """
    thetas = np.asarray(thetas, dtype=np.float64)
    xy = arm.fk_batch(thetas)

    t = np.arange(thetas.shape[0])

    # Plot joints vs time
    plt.figure()
    plt.plot(t, thetas[:, 0], label="theta1 (rad)")
    plt.plot(t, thetas[:, 1], label="theta2 (rad)")
    plt.xlabel("timestep")
    plt.ylabel("angle (rad)")
    plt.title("Joint Trajectory (Jerky Example)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}_joints.png", dpi=200)
    plt.close()

    # Plot end-effector path
    plt.figure()
    plt.plot(xy[:, 0], xy[:, 1])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("End-Effector Path")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}_path.png", dpi=200)
    plt.close()


def main() -> None:
    arm = Planar2DOF(l1=1.0, l2=1.0)
    thetas = generate_jerky_joint_trajectory(T=250, seed=42, scale=1.0)
    plot_trajectory_and_path(thetas, arm, save_path_prefix="results/arm_demo")
    print("Saved plots to results/arm_demo_joints.png and results/arm_demo_path.png")


if __name__ == "__main__":
    main()
