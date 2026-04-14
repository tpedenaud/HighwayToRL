# HighwayToRL: Deep Q-Network for Highway Driving

A reinforcement learning project implementing and comparing custom DQN agents with Stable-Baselines3 on the `highway-v0` environment, with advanced extensions for algorithmic improvements and observation representation exploration.

## Project Overview

This project completes the core RL task and three substantial algorithmic and architectural extensions:

1. **Core Task**: Implement a DQN agent from scratch, train on the `highway-v0` benchmark, compare against Stable-Baselines3, and thoroughly evaluate both approaches.
2. **Extension 1**: Double DQN (DDQN) - addressing Q-value overestimation through decoupled action selection
3. **Extension 2**: Double DQN + Prioritized Experience Replay (DDQN+PER) - combining algorithmic stability with sample efficiency improvements
4. **Extension 3**: CNN-based observation representation - effect of feature learning with convolutional neural networks

## Core Task

### Environment Configuration

The benchmark uses the `highway-v0` environment with standardized settings defined in `config.py`:

- **Observation**: Kinematics representation (presence, x, y, vx, vy for 10 vehicles)
- **Action**: DiscreteMetaAction with target speeds [20, 25, 30] km/h
- **Lanes**: 4 lanes
- **Vehicles**: 45 surrounding vehicles, 1 controlled agent
- **Duration**: 30 second episodes
- **Reward**: Composed of collision penalty (-1.5), high speed reward (0.7), and lane change penalty (-0.02)

### Core Implementations

**Custom DQN Agent** (`rl/dqn.py`):
- Standard DQN with epsilon-greedy exploration
- Replay buffer for experience storage (capacity: 10,000)
- Target network updated every 1,000 steps
- Adam optimizer with learning rate 0.001
- Gamma discount factor: 0.99

**Stable-Baselines3 Baseline**:
- Official SB3 DQN implementation trained on same environment
- Same hyperparameter tuning when possible
- Used for comparison to validate custom implementation

### Core Results

**Custom DQN (3 seeds evaluation)**:
- Mean reward across 50 evaluation episodes per seed
- Standard deviation tracking
- Crash rate and episode duration metrics

**Stable-Baselines3 Comparison**:
- Relative performance analysis
- Training convergence speed
- Robustness across seeds

## Extension 1: Double DQN (DDQN)

### Motivation

Standard DQN tends to overestimate Q-values due to using the same network for both action selection and evaluation. This can lead to suboptimal policies and instability during training.

### Implementation (`rl/double_dqn.py`, `rl/ddqn_train.py`)

Double DQN addresses this by decoupling action selection from Q-value estimation:
- **Action selection**: Use the online (current) network to select the best next action
- **Q-value estimation**: Use the target network to evaluate the Q-value of the selected action
- This reduces overestimation bias and improves stability

### Results

Results stored in `artifacts/ddqn/double_dqn/`:
- Three training runs with different random seeds (0, 1, 2)
- Per-seed checkpoint and metrics:
  - `checkpoint.pt`: Trained model weights
  - `metrics.npz`: Training dynamics (rewards, losses, epsilon evolution)

## Extension 2: Double DQN + Prioritized Experience Replay (DDQN+PER)

### Motivation

While Double DQN improves stability, Prioritized Experience Replay (PER) enhances sample efficiency. Combining both approaches provides both stability and data efficiency improvements.

### Implementation (`rl/ddqn_per.py`, `rl/prioritized_replay_buffer.py`)

**Double DQN Component** (same as Extension 1):
- Decouples action selection from Q-value evaluation
- Uses current network to select actions, target network to evaluate them

**Prioritized Experience Replay Component**:
- Samples transitions based on Temporal Difference (TD) error magnitude
- Higher TD-error transitions are sampled more frequently

### Architecture

- Combines Double DQN update rule with PER sampling
- `prioritized_replay_buffer.py` maintains priority scores and samples accordingly
- Importance weights applied during loss computation

### Results

Results stored in `artifacts/custom_ddqn_per/`:
- Three training runs with different random seeds (0, 1, 2)
- Per-seed checkpoint and metrics:
  - `checkpoint.pt`: Trained model weights
  - `metrics.npz`: Training dynamics (rewards, losses, epsilon evolution, priorities)

## Extension 3: CNN-based Observation Representation

### Motivation

Standard kinematics observations provide explicit vehicle features but don't leverage spatial structure. Convolutional Neural Networks (CNNs) can learn hierarchical spatial features from grid-based representations, potentially improving long-term planning and generalization.

### Approach

Explores the effect of learned feature representation through CNN architectures:

1. **Kinematics** (baseline): Vehicle state vectors (presence, position, velocity)
   - Observation shape: (10 vehicles × 5 features)
   - Compact, interpretable feature representation

2. **Occupancy Grid with CNN feature learning**: Spatial grid-based input processed by CNN
   - Grid configuration: [-27.5, 27.5] × [-27.5, 27.5] with 5×5 spacing
   - Convolutional layers learn spatial feature hierarchies
   - Captures relative spatial relationships and patterns

### Implementation (`rl/observation_extension_config.py`, `rl/train_cnn_observation_extension.py`, `rl/common_cnn_observation_extension.py`)

- Dynamic environment configuration supporting multiple observation modes
- Modified network architecture to support different input shapes
- Training pipeline for fair comparison across representations
- Same hyperparameters and random seeds for reproducibility

### Analysis (`artifacts/observation_extension/`)

**Comparison Summary** (`comparison_summary.csv`, `comparison_summary.md`):
- Per-seed performance metrics:
  - Mean reward, reward standard deviation, median reward
  - Crash rate and episode duration
  - Mean speed achieved

**Visualization Artifacts**:
- `observation_mode_summary_50eval.png`: Performance comparison overview
- `reward_per_eval_seed_50eval.png`: Per-seed reward distributions
- `episode_*.gif`: Visualization of learned behaviors with each representation
- `reward_distribution_boxplot_50eval.png`: Reward distribution across runs

## Project Structure

```
HighwayToRL/
├── config.py                          # Shared environment configuration
├── requirements.txt                   # Python dependencies
├── train_ddqn_per.py                 # CLI training script for DDQN+PER
├── rl/                               # Core implementation module
│   ├── dqn.py                        # Standard DQN agent
│   ├── double_dqn.py                 # Double DQN agent (Extension 1)
│   ├── ddqn_train.py                 # DDQN training utilities
│   ├── ddqn_per.py                   # Double DQN + PER agent (Extension 2)
│   ├── network.py                    # Neural network architecture
│   ├── replay_buffer.py              # Standard replay buffer
│   ├── prioritized_replay_buffer.py  # PER implementation (Extension 2)
│   ├── train.py                      # Training loop utilities
│   ├── evaluate.py                   # Evaluation and rollout utilities
│   ├── visualize.py                  # Plotting and analysis helpers
│   ├── utils.py                      # Preprocessing utilities
│   ├── observation_extension_config.py # Multi-representation config (Extension 3)
│   ├── train_cnn_observation_extension.py # CNN training (Extension 3)
│   ├── evaluate_cnn_observation_extension.py # CNN evaluation (Extension 3)
│   ├── common_cnn_observation_extension.py  # CNN utilities (Extension 3)
│   ├── run_cnn_observation_extension.py # CNN execution script (Extension 3)
│   └── __init__.py
├── nb.ipynb                          # Core task + DDQN_PER extension notebook
├── ddqn.ipynb                        # DDQN training notebook
├── CNN_notebook.ipynb                # CNN observation extension notebook
├── artifacts/                        # Generated results (non-tracked)
│   ├── custom_dqn/                   # Custom DQN training results
│   ├── ddqn/                         # Extension 1: DDQN training results
│   ├── custom_ddqn_per/              # Extension 2: DDQN+PER training results (3 seeds)
│   ├── sb3/                          # Stable-Baselines3 results
│   ├── observation_extension/        # Extension 3: CNN observation comparison results
│   ├── rollouts/                     # Episode visualizations
│   ├── failures/                     # Failure mode analysis
│   ├── failures_ddqn_per/            # Extension 2 specific failures
│   ├── reports/                      # Analysis reports
│   └── evaluation/                   # Evaluation run results
```

## Essential Files and Contents

### Training & Core Logic

| File | Purpose |
|------|---------|
| [`config.py`](config.py) | Defines standardized highway-v0 configuration for reproducibility |
| [`rl/dqn.py`](rl/dqn.py) | Core DQN agent implementation with epsilon-greedy exploration |
| [`rl/double_dqn.py`](rl/double_dqn.py) | Double DQN agent (Extension 1) |
| [`rl/ddqn_per.py`](rl/ddqn_per.py) | DDQN+PER agent (Extension 2) |
| [`rl/network.py`](rl/network.py) | Neural network architecture for Q-value approximation |
| [`rl/replay_buffer.py`](rl/replay_buffer.py) | Standard experience replay buffer (FIFO) |
| [`rl/prioritized_replay_buffer.py`](rl/prioritized_replay_buffer.py) | PER-based sampling with TD-error priorities |
| [`rl/train.py`](rl/train.py) | Training loop with epsilon decay and target network sync |
| [`rl/ddqn_train.py`](rl/ddqn_train.py) | Extension training utilities for DDQN variants |

### Analysis & Evaluation

| File | Purpose |
|------|---------|
| [`rl/evaluate.py`](rl/evaluate.py) | Evaluation protocols and rollout collection (50 runs per seed) |
| [`rl/utils.py`](rl/utils.py) | Observation preprocessing and normalization |
| [`rl/visualize.py`](rl/visualize.py) | Plotting metrics, learning curves, and failure modes |
| [`nb.ipynb`](nb.ipynb) | Main workflow: train, evaluate, and compare DQN vs SB3 |
| [`ddqn.ipynb`](ddqn.ipynb) | DDQN+PER training and results analysis |

### Extension: Observation Representation (Extension 3)

| File | Purpose |
|------|---------|
| [`rl/observation_extension_config.py`](rl/observation_extension_config.py) | Builder for Kinematics and Occupancy Grid configurations |
| [`rl/train_cnn_observation_extension.py`](rl/train_cnn_observation_extension.py) | Training logic for different observation modes |
| [`rl/evaluate_cnn_observation_extension.py`](rl/evaluate_cnn_observation_extension.py) | Fair evaluation across representations |
| [`rl/common_cnn_observation_extension.py`](rl/common_cnn_observation_extension.py) | Shared CNN utilities and helpers |
| [`rl/run_cnn_observation_extension.py`](rl/run_cnn_observation_extension.py) | Execution script for extension study |
| [`CNN_notebook.ipynb`](CNN_notebook.ipynb) | Observation representation analysis and visualization |

## Main Results Summary

The main findings and analysis can be reviewed in the following notebooks:
- `nb.ipynb`: Main analysis comparing custom DQN vs Stable-Baselines3
- `ddqn.ipynb`: DDQN and DDQN+PER training and comparison
- `CNN_notebook.ipynb`: CNN observation representation analysis

All outputs and results are stored in the `artifacts/` directory and documented in the README sections above.

## Key Hyperparameters

### Training Configuration
- **Total training steps**: 20,000
- **Learning starts**: 1,000 (warm-up)
- **Batch size**: 32
- **Replay buffer capacity**: 10,000
- **Learning rate**: 0.001
- **Gamma (discount)**: 0.99
- **Target network update**: Every 1,000 steps

### Exploration
- **Initial epsilon**: 1.0
- **Final epsilon**: 0.05
- **Decay schedule**: Linear over training

### Double DQN + PER (when enabled)
- **Prioritization alpha**: (Configurable, typically 0.6)
- **Importance sampling beta**: Anneals from 0.4 to 1.0
- **Min priority**: 1e-6 (for stability)

## Evaluation Protocol

**Thorough Evaluation (50 runs per seed)**:
- 50 independent evaluation episodes per trained model
- No exploration (epsilon=0.0), deterministic actions
- Metrics collected:
  - Mean episode reward
  - Reward standard deviation
  - Episode length statistics
  - Crash rate (early termination rate)
  - Mean velocity achieved

**Multi-Seed Training**:
- Multiple random seeds (0, 1, 2) to assess stability
- Statistical significance via mean/std across seeds
- Reproducibility through seed control

## Failure Mode Analysis

Stored in `artifacts/failures` and `artifacts/failures_ddqn_per`:
- Collected episodes where agent crashes or performs poorly
- Analyzed through replay buffers and reward traces
- Visualized as GIFs showing learned behaviors
- Common failure patterns:
  - Sudden lane changes into collision
  - Slow recovery from suboptimal positions
  - Insufficient lookahead in high-density traffic

## Notes

- **Generated artifacts** are NOT tracked by git (see `.gitignore`)
- **Results reproducibility**: Use fixed seeds for deterministic behavior
- **Device optimization**: GPU acceleration via MPS (Mac) or CUDA (Linux/Windows) when available
- **Environment dynamics**: All runs use the same `highway-v0` environment setup
- **Hyperparameter tuning**: Can be extended in training scripts

## References

- Highway-Env: https://github.com/eleurent/highway-env
- Stable-Baselines3: https://github.com/DLR-RM/stable-baselines3
- DQN Paper: Mnih et al. (2015) - "Human-level control through deep reinforcement learning"
- Double DQN: van Hasselt et al. (2015) - "Deep Reinforcement Learning with Double Q-learning"
- PER Paper: Schaul et al. (2015) - "Prioritized Experience Replay"