import argparse

import numpy as np
import pandas as pd

from common import (
    ACTION_LABELS,
    CUSTOM_MODEL_NAME,
    DEFAULT_EVAL_SEEDS,
    DEFAULT_TRAIN_SEEDS,
    SB3_MODEL_NAME,
    comparison_csv_path,
    comparison_markdown_path,
    dump_json,
    ensure_run_dirs,
    get_run_paths,
    make_env,
    resolve_device,
    seed_env,
)
from custom_dqn import CustomDQNPolicy


class SB3Policy:
    def __init__(self, model):
        self.model = model

    def predict(self, observation):
        action, _ = self.model.predict(observation, deterministic=True)
        return int(action)


def load_policy(model_name, seed, device="cpu"):
    device = resolve_device(device)
    paths = get_run_paths(model_name, seed)
    if model_name == CUSTOM_MODEL_NAME:
        return CustomDQNPolicy.load(paths["final_model_path"], device=device)
    if model_name == SB3_MODEL_NAME:
        from stable_baselines3 import DQN

        return SB3Policy(DQN.load(paths["final_model_path"], device=device))
    raise ValueError(f"Unknown model name: {model_name}")


def rollout_episode(policy, episode_seed, capture_frames=False):
    env = make_env(render_mode="rgb_array" if capture_frames else None)
    try:
        observation, _ = seed_env(env, episode_seed)
        frames = []
        observations = [observation.copy()]
        actions = []
        rewards = []
        speeds = []
        crashed = False
        total_reward = 0.0
        episode_length = 0

        if capture_frames:
            frames.append(env.render())

        while True:
            action = policy.predict(observation)
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            actions.append(action)
            rewards.append(float(reward))
            observations.append(next_observation.copy())
            speeds.append(float(info.get("speed", 0.0)))
            crashed = crashed or bool(info.get("crashed", False))
            total_reward += float(reward)
            episode_length += 1

            if capture_frames:
                frames.append(env.render())

            observation = next_observation
            if done:
                return {
                    "episode_seed": episode_seed,
                    "reward": total_reward,
                    "episode_length": episode_length,
                    "crashed": crashed,
                    "mean_speed": float(np.mean(speeds)) if speeds else 0.0,
                    "frames": frames,
                    "observations": observations,
                    "actions": actions,
                    "action_labels": [ACTION_LABELS.get(action, str(action)) for action in actions],
                    "reward_trace": rewards,
                }
    finally:
        env.close()


def summarize_episodes(episodes_df, model_name, train_seed):
    return {
        "model_name": model_name,
        "train_seed": train_seed,
        "num_eval_episodes": int(len(episodes_df)),
        "mean_reward": float(episodes_df["reward"].mean()),
        "reward_std": float(episodes_df["reward"].std(ddof=0)),
        "crash_rate": float(episodes_df["crashed"].mean()),
        "mean_episode_length": float(episodes_df["episode_length"].mean()),
        "mean_speed": float(episodes_df["mean_speed"].mean()),
    }


def evaluate_one_run(model_name, train_seed, eval_seeds=None, device="cpu"):
    eval_seeds = DEFAULT_EVAL_SEEDS if eval_seeds is None else eval_seeds
    policy = load_policy(model_name, train_seed, device=device)
    paths = ensure_run_dirs(model_name, train_seed)

    rows = []
    for eval_seed in eval_seeds:
        rollout = rollout_episode(policy, int(eval_seed), capture_frames=False)
        rows.append(
            {
                "eval_seed": int(eval_seed),
                "reward": rollout["reward"],
                "episode_length": rollout["episode_length"],
                "crashed": int(rollout["crashed"]),
                "mean_speed": rollout["mean_speed"],
            }
        )

    episodes_df = pd.DataFrame(rows)
    episodes_df.to_csv(paths["evaluation_episodes_path"], index=False)
    summary = summarize_episodes(episodes_df, model_name, train_seed)
    dump_json(paths["evaluation_summary_path"], summary)
    return summary


def evaluate_all(train_seeds=None, eval_seeds=None, device="cpu", model_names=None):
    train_seeds = DEFAULT_TRAIN_SEEDS if train_seeds is None else train_seeds
    eval_seeds = DEFAULT_EVAL_SEEDS if eval_seeds is None else eval_seeds
    model_names = [CUSTOM_MODEL_NAME, SB3_MODEL_NAME] if model_names is None else model_names

    rows = []
    for model_name in model_names:
        for train_seed in train_seeds:
            rows.append(evaluate_one_run(model_name, train_seed, eval_seeds=eval_seeds, device=device))

    summary_df = pd.DataFrame(rows).sort_values(["model_name", "train_seed"]).reset_index(drop=True)
    grouped_df = (
        summary_df.groupby("model_name", as_index=False)
        .agg(
            mean_reward=("mean_reward", "mean"),
            reward_seed_std=("mean_reward", "std"),
            mean_eval_std=("reward_std", "mean"),
            crash_rate=("crash_rate", "mean"),
            mean_episode_length=("mean_episode_length", "mean"),
            mean_speed=("mean_speed", "mean"),
        )
        .fillna(0.0)
    )

    summary_df.to_csv(comparison_csv_path(), index=False)
    comparison_markdown_path().write_text(
        "\n".join(
            [
                "# Evaluation Comparison",
                "",
                "## Per-run results",
                "",
                summary_df.to_markdown(index=False),
                "",
                "## Grouped by model",
                "",
                grouped_df.to_markdown(index=False),
                "",
            ]
        ),
        encoding="utf-8",
    )
    return summary_df


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained custom and SB3 DQN checkpoints.")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_TRAIN_SEEDS)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "auto"])
    parser.add_argument("--eval-seeds-start", type=int, default=DEFAULT_EVAL_SEEDS[0])
    parser.add_argument("--num-eval-episodes", type=int, default=len(DEFAULT_EVAL_SEEDS))
    args = parser.parse_args()

    eval_seeds = list(range(args.eval_seeds_start, args.eval_seeds_start + args.num_eval_episodes))
    evaluate_all(train_seeds=args.seeds, eval_seeds=eval_seeds, device=args.device)


if __name__ == "__main__":
    main()
