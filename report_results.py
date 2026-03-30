import argparse
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (
    CUSTOM_MODEL_NAME,
    DEFAULT_TRAIN_SEEDS,
    DEFAULT_EVAL_SEEDS,
    SHARED_CORE_CONFIG,
    SHARED_CORE_ENV_ID,
    behavior_analysis_path,
    comparison_csv_path,
    comparison_markdown_path,
    core_summary_path,
    failure_contact_sheet_path,
    get_run_paths,
    success_contact_sheet_path,
    training_curve_path,
)
from evaluate_models import load_policy, rollout_episode


def load_training_logs(train_seeds):
    logs = []
    for model_name in [CUSTOM_MODEL_NAME, "sb3_dqn"]:
        for train_seed in train_seeds:
            training_log_path = get_run_paths(model_name, train_seed)["training_log_path"]
            if training_log_path.exists():
                df = pd.read_csv(training_log_path)
                if not df.empty:
                    df["model_name"] = model_name
                    df["train_seed"] = train_seed
                    logs.append(df)
    if not logs:
        raise FileNotFoundError("No training logs found. Train the models first.")
    return pd.concat(logs, ignore_index=True)


def plot_training_curves(train_seeds):
    training_df = load_training_logs(train_seeds)
    color_map = {CUSTOM_MODEL_NAME: "#1f77b4", "sb3_dqn": "#d62728"}

    plt.figure(figsize=(10, 6))
    for (model_name, train_seed), run_df in training_df.groupby(["model_name", "train_seed"]):
        run_df = run_df.sort_values("global_step").copy()
        run_df["smoothed_reward"] = run_df["episode_reward"].rolling(window=20, min_periods=1).mean()
        plt.plot(
            run_df["global_step"],
            run_df["smoothed_reward"],
            label=f"{model_name} seed {train_seed}",
            color=color_map.get(model_name, "#555555"),
            alpha=0.8,
        )

    plt.title("Training Curves")
    plt.xlabel("Environment steps")
    plt.ylabel("Smoothed episode reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(training_curve_path(), dpi=200)
    plt.close()


def sample_frame_indices(num_frames, max_frames=6):
    if num_frames <= max_frames:
        return list(range(num_frames))
    return sorted({int(round(index)) for index in np.linspace(0, num_frames - 1, max_frames)})


def save_contact_sheet(frames, output_path, title):
    indices = sample_frame_indices(len(frames))
    columns = min(3, len(indices))
    rows = math.ceil(len(indices) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(4.5 * columns, 3.5 * rows))
    axes = np.atleast_1d(axes).ravel()

    for axis, frame_index in zip(axes, indices):
        axis.imshow(frames[frame_index])
        axis.set_title(f"Step {frame_index}")
        axis.axis("off")

    for axis in axes[len(indices) :]:
        axis.axis("off")

    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def explain_failure(rollout):
    if not rollout["crashed"]:
        return "No crash happened in this rollout."

    final_observation = np.array(rollout["observations"][-1], dtype=np.float32).reshape(-1, 5)
    nearby = final_observation[1:][final_observation[1:, 0] > 0.5]
    latest_action = rollout["action_labels"][-1] if rollout["action_labels"] else "IDLE"

    if len(nearby) == 0:
        return (
            "The crash happens without a clearly visible nearby vehicle in the clipped observation, "
            "which suggests that the policy reacted too late to traffic outside the immediate view."
        )

    frontal = nearby[(nearby[:, 1] >= 0.0) & (np.abs(nearby[:, 2]) < 0.25)]
    lateral = nearby[np.abs(nearby[:, 2]) >= 0.25]

    if len(frontal) > 0 and latest_action in {"FASTER", "IDLE"}:
        return (
            "The final observation still contains a nearby vehicle ahead, and the last action does not create more space. "
            "This looks like a late reaction to slower traffic."
        )

    if len(lateral) > 0 and latest_action in {"LANE_LEFT", "LANE_RIGHT"}:
        return (
            "The terminal observation still contains nearby traffic in an adjacent lane while the last action is lateral. "
            "This looks like an unsafe lane change."
        )

    return "The policy reaches a risky local state and only reacts once safe spacing is already gone."


def make_behavior_analysis(train_seeds, device):
    summary_df = pd.read_csv(comparison_csv_path())
    summary_df = summary_df[summary_df["train_seed"].isin(train_seeds)]
    best_run = summary_df.sort_values("mean_reward", ascending=False).iloc[0]
    best_model = best_run["model_name"]
    best_seed = int(best_run["train_seed"])

    episodes_df = pd.read_csv(get_run_paths(best_model, best_seed)["evaluation_episodes_path"])
    success_rows = episodes_df[episodes_df["crashed"] == 0]
    success_row = success_rows.sort_values("reward", ascending=False).iloc[0] if not success_rows.empty else episodes_df.sort_values("reward", ascending=False).iloc[0]
    failure_rows = episodes_df[episodes_df["crashed"] == 1]
    failure_row = failure_rows.sort_values("eval_seed").iloc[0] if not failure_rows.empty else episodes_df.sort_values("reward").iloc[0]

    policy = load_policy(best_model, best_seed, device=device)
    success_rollout = rollout_episode(policy, int(success_row["eval_seed"]), capture_frames=True)
    failure_rollout = rollout_episode(policy, int(failure_row["eval_seed"]), capture_frames=True)

    save_contact_sheet(
        success_rollout["frames"],
        success_contact_sheet_path(),
        f"Success rollout: {best_model}, seed {best_seed}, eval seed {int(success_row['eval_seed'])}",
    )
    save_contact_sheet(
        failure_rollout["frames"],
        failure_contact_sheet_path(),
        f"Failure rollout: {best_model}, seed {best_seed}, eval seed {int(failure_row['eval_seed'])}",
    )

    behavior_analysis_path().write_text(
        "\n".join(
            [
                "# Behavior Analysis",
                "",
                f"- Best run used for qualitative analysis: `{best_model}` with training seed `{best_seed}`.",
                f"- Success case eval seed: `{int(success_row['eval_seed'])}`.",
                f"- Failure case eval seed: `{int(failure_row['eval_seed'])}`.",
                "",
                "## Failure mode",
                "",
                explain_failure(failure_rollout),
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def make_core_summary(train_seeds):
    summary_df = pd.read_csv(comparison_csv_path())
    summary_df = summary_df[summary_df["train_seed"].isin(train_seeds)]
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
    best_model_row = grouped_df.sort_values("mean_reward", ascending=False).iloc[0]

    core_summary_path().write_text(
        "\n".join(
            [
                "# Core Task Summary",
                "",
                f"- Environment: `{SHARED_CORE_ENV_ID}`",
                "- Observation: flattened Kinematics (50 features)",
                "- Actions: `DiscreteMetaAction`",
                f"- Training seeds: `{list(train_seeds)}`",
                f"- Evaluation seeds: `{DEFAULT_EVAL_SEEDS[0]}` to `{DEFAULT_EVAL_SEEDS[-1]}`",
                "",
                "## Main result",
                "",
                f"The strongest average mean reward comes from `{best_model_row['model_name']}` with `{best_model_row['mean_reward']:.3f}`.",
                "",
                "## Design choices",
                "",
                "- The custom agent is a plain DQN with replay buffer, target network, epsilon-greedy exploration, Huber loss, and gradient clipping.",
                "- The SB3 baseline uses the same benchmark, same flattened input, and matched high-level hyperparameters where possible.",
                "- Both models are evaluated on the same saved evaluation seeds with the same metrics.",
                "",
                "## Shared benchmark config",
                "",
                f"```python\n{SHARED_CORE_CONFIG}\n```",
                "",
                "## Result table",
                "",
                summary_df.to_markdown(index=False),
                "",
                "Grouped summary:",
                "",
                grouped_df.to_markdown(index=False),
                "",
                f"Training curves: `{training_curve_path()}`",
                f"Evaluation table: `{comparison_markdown_path()}`",
                f"Behavior analysis: `{behavior_analysis_path()}`",
                "",
            ]
        ),
        encoding="utf-8",
    )


def make_reports(train_seeds=None, device="cpu"):
    train_seeds = DEFAULT_TRAIN_SEEDS if train_seeds is None else train_seeds
    plot_training_curves(train_seeds)
    make_behavior_analysis(train_seeds, device=device)
    make_core_summary(train_seeds)


def main():
    parser = argparse.ArgumentParser(description="Build plots and markdown reports from saved artifacts.")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_TRAIN_SEEDS)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "auto"])
    args = parser.parse_args()
    make_reports(train_seeds=args.seeds, device=args.device)


if __name__ == "__main__":
    main()
