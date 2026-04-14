# HighwayToRL

Reinforcement learning project for the `highway-v0` benchmark in `highway-env`.

## Project overview
The repository contains:

- the shared core task required for the course
- one integrated extension notebook built on top of the same custom DQN codebase

The code stays centered in `rl/`, with notebooks used only to orchestrate experiments and present results.

## Core task
- Environment: `highway-v0`
- Shared benchmark config: `config.py`
- Observation: `Kinematics`
- Action space: `DiscreteMetaAction`
- Comparison: custom DQN vs Stable-Baselines3 DQN
- Main notebook: `nb.ipynb`

## Extension question
The extension asks:

- how does the observation representation affect learning?
- how much do Double DQN and Prioritized Experience Replay help over the baseline custom DQN?
- can explicit reward shaping and training on more diverse traffic settings reduce crashes while preserving useful driving behavior?

The extension notebook is `extension.ipynb`.

## Project structure
- `config.py`: shared benchmark config used in the core task
- `rl/`
  - `network.py`: MLP and CNN Q-networks
  - `replay_buffer.py`: uniform replay and prioritized replay
  - `dqn.py`: integrated custom DQN agent
  - `train.py`: training entry point
  - `evaluate.py`: evaluation, rollout, and failure-case helpers
  - `extension_configs.py`: extension experiment definitions
  - `reward_shaping.py`: explicit safety reward wrapper
  - `visualize.py`: plotting and notebook display helpers
  - `utils.py`: observation preprocessing and observation-mode helpers
- `nb.ipynb`: core-task notebook
- `extension.ipynb`: final extension notebook

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the notebooks
Core task:

- train the custom DQN on 3 seeds
- train SB3 DQN on the same benchmark
- evaluate both on the same evaluation seeds
- compare metrics, rollouts, and failure cases

Extension task:

- compare `Kinematics` + MLP against `OccupancyGrid` + CNN
- compare baseline DQN, Double DQN, PER, and PER+Double DQN on the kinematics setup
- select a safe reward preset
- test whether reward shaping and mixed-traffic training improve safety and generalization

## Notes
- Generated outputs are written under `artifacts/` and are ignored by git.
- The core task and the extension share the same `rl/` module so the implementation stays compact and reproducible.
