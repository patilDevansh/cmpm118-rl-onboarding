"""Train and evaluate a PPO baseline on Gymnasium CarRacing-v3.

The script writes TensorBoard scalars, periodic evaluation results, a trained
model, and a reward plot suitable for CMPM 118 Task 5.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import gymnasium as gym
import numpy as np

# Keep Matplotlib's cache inside the project when the user home is read-only.
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage


def make_env(render_mode: str | None = None):
    def _init():
        env = gym.make("CarRacing-v3", continuous=True, render_mode=render_mode)
        return Monitor(env)

    return _init


def build_env(render_mode: str | None = None):
    env = DummyVecEnv([make_env(render_mode)])
    env = VecTransposeImage(env)
    return VecFrameStack(env, n_stack=4)


def plot_evaluations(csv_path: Path, plot_path: Path) -> None:
    steps, rewards = [], []
    with csv_path.open(newline="") as file:
        for row in csv.DictReader(file):
            steps.append(int(row["timesteps"]))
            rewards.append(float(row["mean_reward"]))

    plt.figure(figsize=(8, 4.5))
    plt.plot(steps, rewards, marker="o", color="#2563eb")
    plt.title("PPO evaluation reward on CarRacing-v3")
    plt.xlabel("Training timesteps")
    plt.ylabel("Mean reward (5 episodes)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=180)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--output", type=Path, default=Path("artifacts/task5_carracing"))
    args = parser.parse_args()

    output = args.output
    logs = output / "tensorboard"
    evaluations = output / "evaluations"
    output.mkdir(parents=True, exist_ok=True)
    evaluations.mkdir(exist_ok=True)

    train_env = build_env()
    eval_env = build_env()
    callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output / "best_model"),
        log_path=str(evaluations),
        eval_freq=args.eval_freq,
        n_eval_episodes=5,
        deterministic=True,
    )
    model = PPO(
        "CnnPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        gamma=0.99,
        verbose=1,
        tensorboard_log=str(logs),
        device="auto",
    )
    model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=False)
    model.save(output / "ppo_carracing")

    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=5, deterministic=True)
    (output / "final_evaluation.txt").write_text(
        f"mean_reward={mean_reward:.2f}\nstd_reward={std_reward:.2f}\ntimesteps={args.timesteps}\n"
    )

    # EvalCallback stores NumPy archives. Save a portable CSV and plot from it.
    archive_path = evaluations / "evaluations.npz"
    if not archive_path.exists():
        raise RuntimeError(
            "No periodic evaluations were recorded. Set --eval-freq lower than --timesteps."
        )
    archive = np.load(archive_path)
    csv_path = output / "evaluation_rewards.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["timesteps", "mean_reward"])
        writer.writeheader()
        for timestep, rewards in zip(archive["timesteps"], archive["results"]):
            writer.writerow({"timesteps": int(timestep), "mean_reward": float(rewards.mean())})
    plot_evaluations(csv_path, output / "reward_curve.png")


if __name__ == "__main__":
    main()
