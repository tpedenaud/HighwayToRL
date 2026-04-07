# HighwayToRL

Reinforcement learning project for the `highway-v0` benchmark.

## Core task
- Environment: `highway-v0`
- Config: `config.py`
- Observation: `Kinematics`
- Actions: `DiscreteMetaAction`
- Comparison: custom DQN vs Stable-Baselines3 DQN

## Project structure
- `config.py`: shared benchmark config
- `rl/`
  - `network.py`: Q-network
  - `replay_buffer.py`: replay buffer
  - `dqn.py`: custom DQN agent
  - `train.py`: custom DQN training
  - `evaluate.py`: evaluation and rollout helpers
  - `visualize.py`: plotting and notebook display helpers
  - `utils.py`: preprocessing helpers
- `nb.ipynb`: notebook for training, evaluation, and analysis

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Main workflow
The main workflow is in nb.ipynb:

- train the custom DQN on 3 seeds
- train SB3 DQN on the same benchmark
- evaluate both on the same evaluation seeds
- compare metrics, rollouts, and failure cases

## Notes
Generated outputs are written to artifacts/ and are not tracked by git.