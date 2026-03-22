# HighwayToRL

## Environment
- **Environment**: `highway-v0`
- **Configuration**: `shared_core_config.py`
- **Observations**: `Kinematics`
- **Actions**: `DiscreteMetaAction`

---

## Objectives
- Implement and train a **DQN agent** on the provided benchmark.
- Train a **Stable-Baselines model** on the same setup.
- Perform a rigorous comparison between both approaches.

---

## Evaluation Protocol
- Save model checkpoints and training metrics.

- Evaluate each model over **50 episodes** by computing the mean reward and the standard deviation.

- Run experiments on **at least 3 different seeds**.

---

## Expected Results
- Training curves  
- Evaluation table (at least 3 seeds)  
- Fair comparison between DQN and Stable-Baselines:
  - Same configuration  
  - Same evaluation seeds  
  - Same metrics  

---

## Analysis
- Qualitative analysis (recorded rollout or observed behavior)  
- At least **one failure case** with explanation  
- Brief discussion of design choices  