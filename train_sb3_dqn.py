import argparse

import pandas as pd

from common import (
    DEFAULT_DQN_SETTINGS,
    DEFAULT_TRAIN_SEEDS,
    SB3_MODEL_NAME,
    SHARED_CORE_CONFIG,
    SHARED_CORE_ENV_ID,
    dump_json,
    ensure_run_dirs,
    make_env,
    resolve_device,
    seed_env,
    set_seed,
)


def convert_monitor_log(monitor_path, training_log_path):
    monitor_df = pd.read_csv(monitor_path, comment="#")
    if monitor_df.empty:
        monitor_df = pd.DataFrame(columns=["r", "l", "t", "crashed", "speed"])

    monitor_df["episode_index"] = range(1, len(monitor_df) + 1)
    monitor_df["global_step"] = monitor_df["l"].cumsum()
    monitor_df = monitor_df.rename(columns={"r": "episode_reward", "l": "episode_length"})
    if "crashed" not in monitor_df:
        monitor_df["crashed"] = 0
    if "speed" not in monitor_df:
        monitor_df["speed"] = 0.0
    monitor_df["epsilon"] = pd.NA
    monitor_df["mean_speed"] = monitor_df["speed"]
    monitor_df[
        ["episode_index", "global_step", "episode_reward", "episode_length", "epsilon", "crashed", "mean_speed"]
    ].to_csv(training_log_path, index=False)


def train_one_seed(seed, total_timesteps, device="cpu", settings=None):
    from stable_baselines3 import DQN
    from stable_baselines3.common.callbacks import CheckpointCallback

    settings = dict(DEFAULT_DQN_SETTINGS if settings is None else settings)
    device = resolve_device(device)
    set_seed(seed)
    paths = ensure_run_dirs(SB3_MODEL_NAME, seed)

    env = make_env(monitor_path=paths["monitor_path"])
    seed_env(env, seed)

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=settings["learning_rate"],
        buffer_size=settings["buffer_size"],
        learning_starts=settings["learning_starts"],
        batch_size=settings["batch_size"],
        gamma=settings["gamma"],
        train_freq=settings["train_freq"],
        gradient_steps=1,
        target_update_interval=settings["target_update_interval"],
        exploration_initial_eps=settings["epsilon_start"],
        exploration_final_eps=settings["epsilon_end"],
        exploration_fraction=min(1.0, settings["epsilon_decay_steps"] / float(total_timesteps)),
        policy_kwargs={"net_arch": settings["hidden_sizes"]},
        seed=seed,
        verbose=0,
        device=device,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=settings["checkpoint_interval"],
        save_path=str(paths["checkpoints_dir"]),
        name_prefix="step_model",
    )
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback, progress_bar=False)
    model.save(paths["final_model_path"])
    env.close()

    convert_monitor_log(paths["monitor_path"], paths["training_log_path"])
    dump_json(
        paths["metadata_path"],
        {
            "model_name": SB3_MODEL_NAME,
            "seed": seed,
            "total_timesteps": total_timesteps,
            "device": device,
            "environment_id": SHARED_CORE_ENV_ID,
            "shared_core_config": SHARED_CORE_CONFIG,
            "dqn_settings": settings,
        },
    )

    return paths


def train_many(seeds, total_timesteps, device="cpu", settings=None):
    for seed in seeds:
        train_one_seed(seed, total_timesteps, device=device, settings=settings)


def main():
    parser = argparse.ArgumentParser(description="Train the SB3 DQN baseline.")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_TRAIN_SEEDS)
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "auto"])
    args = parser.parse_args()
    train_many(args.seeds, args.timesteps, device=args.device)


if __name__ == "__main__":
    main()
