# CMPM 118 Task 5 — RL benchmarks

PPO trained on Gymnasium CarRacing-v3 on a Nautilus V100. At 100,000 requested steps, five deterministic evaluation episodes averaged 369.32 reward (standard deviation 191.40). A separate final evaluation averaged 218.11 (standard deviation 162.00). Training stopped at 100,352 steps because PPO collects complete 1,024-step rollouts. See `ppo_reward_curve.png` and `ppo_evaluation_rewards.csv`.

The course CARLA Gym example was adapted for CARLA 0.9.15 on port 4000 and Town01. A short SAC run recorded four episodes and a mean reward of -1,689.63 at 208 steps. It ended at the five-minute timeout before the planned 1,000 steps. The single logged mean is shown in `carla_gym_reward.png`; it does not establish a learning trend. `carla_gym_run.py` requires the course repository `cruz-control/carla-gym-env` and a running CARLA server.

Run the CarRacing experiment with `python train_carracing_ppo.py --timesteps 100000`. TensorBoard logs and evaluation plots are written to the output directory. The benchmark used Python 3.10, Stable Baselines 3, Gymnasium and a V100-compatible PyTorch CUDA build.
