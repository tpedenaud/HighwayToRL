# HighwayToRL

Small reinforcement-learning project for the course benchmark on `highway-v0`.

## Benchmark
- Environment: `highway-v0`
- Config: `shared_core_config.py`
- Observation: `Kinematics`, flattened to 50 numbers
- Actions: `DiscreteMetaAction`
- Comparison: self-written DQN vs `stable_baselines3.DQN`

## Files
- `common.py`: shared helpers, paths, env creation
- `custom_dqn.py`: custom DQN training code
- `train_sb3_dqn.py`: SB3 training code
- `evaluate_models.py`: evaluation over fixed seeds
- `report_results.py`: plots, tables, rollout analysis
- `run_smoke.py`: quick end-to-end check
- `run_core_experiment.py`: pilot + final training + evaluation + report

## Setup
Create the virtual environment:

```bash
python -m venv .venv
```

Activate it:

- Windows (PowerShell)
	```powershell
	.\.venv\Scripts\Activate.ps1
	```

	If PowerShell blocks the script, run this once in the same shell and retry activation:
	```powershell
	Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
	.\.venv\Scripts\Activate.ps1
	```

- Windows (cmd.exe)
	```bat
	.venv\Scripts\activate.bat
	```

- macOS/Linux
	```bash
	source .venv/bin/activate
	```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick check
```bash
python run_smoke.py
```

This runs a tiny training job for both models, evaluates them on 3 episodes, and builds the report files.

## Main commands
Train the custom DQN on seeds `0 1 2`:
```bash
python custom_dqn.py --seeds 0 1 2 --timesteps 200000 --device cpu
```

Train the SB3 DQN on seeds `0 1 2`:
```bash
python train_sb3_dqn.py --seeds 0 1 2 --timesteps 200000 --device cpu
```

Evaluate both models on the shared evaluation seeds:
```bash
python evaluate_models.py --seeds 0 1 2 --device cpu
```

Build plots and markdown summaries:
```bash
python report_results.py --seeds 0 1 2 --device cpu
```

Run the whole core pipeline in one script:
```bash
python run_core_experiment.py
```

## Outputs
Everything is written under `artifacts/`:
- `artifacts/custom_dqn/seed_<n>/...`
- `artifacts/sb3_dqn/seed_<n>/...`
- `artifacts/evaluation/comparison_summary.csv`
- `artifacts/evaluation/comparison_summary.md`
- `artifacts/reports/training_curves.png`
- `artifacts/reports/core_task_summary.md`
- `artifacts/reports/behavior_analysis.md`

## Tests
```bash
python -m pytest
```
