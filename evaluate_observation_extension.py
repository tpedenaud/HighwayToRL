import argparse

import numpy as np
import pandas as pd

from common import dump_json, resolve_device, seed_env
from common_observation_extension import (
    DEFAULT_EXTENSION_EVAL_SEEDS,
    DEFAULT_EXTENSION_TRAIN_SEEDS,
    ensure_extension_run_dirs,
    extension_comparison_csv_path,
    extension_comparison_markdown_path,
    get_extension_run_paths,
    make_extension_env,
)
from custom_dqn_observation_extension import ExtensionDQNPolicy


def rollout_episode(policy, observation_mode, episode_seed):
    env = make_extension_env(observation_mode)
    try:
        observation, _ = seed_env(env, episode_seed)
        total_reward = 0.0
        episode_length = 0
        crashed = False
        speeds = []

        while True:
            action = policy.predict(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += float(reward)
            episode_length += 1
            speeds.append(float(info.get("speed", 0.0)))
            crashed = crashed or bool(info.get("crashed", False))

            if done:
                return {
                    "episode_seed": int(episode_seed),
                    "reward": total_reward,
                    "episode_length": episode_length,
                    "crashed": int(crashed),
                    "mean_speed": float(np.mean(speeds)) if speeds else 0.0,
                }
    finally:
        env.close()


def summarize_episodes(episodes_df, observation_mode, train_seed):
    return {
        "observation_mode": observation_mode,
        "train_seed": train_seed,
        "num_eval_episodes": int(len(episodes_df)),
        "mean_reward": float(episodes_df["reward"].mean()),
        "reward_std": float(episodes_df["reward"].std(ddof=0)),
        "reward_median": float(episodes_df["reward"].median()),
        "crash_rate": float(episodes_df["crashed"].mean()),
        "mean_episode_length": float(episodes_df["episode_length"].mean()),
        "mean_speed": float(episodes_df["mean_speed"].mean()),
    }


def evaluate_one_run(observation_mode, train_seed, eval_seeds=None, device="cpu"):
    eval_seeds = DEFAULT_EXTENSION_EVAL_SEEDS if eval_seeds is None else eval_seeds
    device = resolve_device(device)
    paths = ensure_extension_run_dirs(observation_mode, train_seed)

    model_path = get_extension_run_paths(observation_mode, train_seed)["final_model_path"]
    policy = ExtensionDQNPolicy.load(model_path, device=device)

    print(
        (
            f"[eval] mode={observation_mode} train_seed={train_seed} "
            f"episodes={len(eval_seeds)}"
        ),
        flush=True,
    )

    rows = []
    for index, eval_seed in enumerate(eval_seeds, start=1):
        rollout = rollout_episode(policy, observation_mode, int(eval_seed))
        rows.append(rollout)
        print(
            (
                f"[eval_episode] mode={observation_mode} train_seed={train_seed} "
                f"{index}/{len(eval_seeds)} eval_seed={int(eval_seed)} "
                f"reward={rollout['reward']:.3f} len={rollout['episode_length']} "
                f"crashed={rollout['crashed']}"
            ),
            flush=True,
        )

    episodes_df = pd.DataFrame(rows)
    episodes_df.to_csv(paths["evaluation_episodes_path"], index=False)
    summary = summarize_episodes(episodes_df, observation_mode, train_seed)
    dump_json(paths["evaluation_summary_path"], summary)
    print(
        (
            f"[eval_summary] mode={observation_mode} train_seed={train_seed} "
            f"mean_reward={summary['mean_reward']:.3f} crash_rate={summary['crash_rate']:.3f}"
        ),
        flush=True,
    )
    return summary


def evaluate_all(observation_modes=None, train_seeds=None, eval_seeds=None, device="cpu"):
    observation_modes = ["kinematics", "occupancy_grid"] if observation_modes is None else observation_modes
    train_seeds = DEFAULT_EXTENSION_TRAIN_SEEDS if train_seeds is None else train_seeds
    eval_seeds = DEFAULT_EXTENSION_EVAL_SEEDS if eval_seeds is None else eval_seeds

    rows = []
    for observation_mode in observation_modes:
        for train_seed in train_seeds:
            rows.append(evaluate_one_run(observation_mode, train_seed, eval_seeds=eval_seeds, device=device))

    summary_df = pd.DataFrame(rows).sort_values(["observation_mode", "train_seed"]).reset_index(drop=True)
    grouped_df = (
        summary_df.groupby("observation_mode", as_index=False)
        .agg(
            mean_reward=("mean_reward", "mean"),
            reward_seed_std=("mean_reward", "std"),
            mean_eval_std=("reward_std", "mean"),
            median_reward=("reward_median", "mean"),
            crash_rate=("crash_rate", "mean"),
            mean_episode_length=("mean_episode_length", "mean"),
            mean_speed=("mean_speed", "mean"),
        )
        .fillna(0.0)
    )

    summary_df.to_csv(extension_comparison_csv_path(), index=False)
    extension_comparison_markdown_path().write_text(
        "\n".join(
            [
                "# Observation Extension Comparison",
                "",
                "## Per-run results",
                "",
                summary_df.to_markdown(index=False),
                "",
                "## Grouped by observation mode",
                "",
                grouped_df.to_markdown(index=False),
                "",
            ]
        ),
        encoding="utf-8",
    )

    return summary_df


def main():
    parser = argparse.ArgumentParser(description="Evaluate kinematics vs occupancy-grid custom DQN checkpoints.")
    parser.add_argument("--observation-modes", nargs="+", choices=["kinematics", "occupancy_grid"], default=["kinematics", "occupancy_grid"])
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_EXTENSION_TRAIN_SEEDS)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "auto"])
    parser.add_argument("--eval-seeds-start", type=int, default=DEFAULT_EXTENSION_EVAL_SEEDS[0])
    parser.add_argument("--num-eval-episodes", type=int, default=len(DEFAULT_EXTENSION_EVAL_SEEDS))
    args = parser.parse_args()

    eval_seeds = list(range(args.eval_seeds_start, args.eval_seeds_start + args.num_eval_episodes))
    evaluate_all(
        observation_modes=args.observation_modes,
        train_seeds=args.seeds,
        eval_seeds=eval_seeds,
        device=args.device,
    )


if __name__ == "__main__":
    main()
