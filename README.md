# Task 5 — PPO benchmark on CarRacing-v3

## Setup

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install swig wheel
PATH="$VIRTUAL_ENV/bin:$PATH" pip install --no-build-isolation box2d-py==2.3.5
pip install -r requirements.txt
```

## Run

```sh
python train_carracing_ppo.py --timesteps 100000
tensorboard --logdir artifacts/task5_carracing/tensorboard
```

Outputs include `reward_curve.png`, `evaluation_rewards.csv`, the saved PPO
model, and TensorBoard event files. Run this on a Nautilus GPU desktop once a
healthy GPU allocation is available. The `artifacts/smoke_test/` folder holds
a verified 1,024-step CPU validation run. It is not a completed 100,000-step
benchmark. The CARLA Gym portion remains blocked by the documented cluster GPU
failure.
