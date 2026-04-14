from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Image, display


def plot_training_curve(metrics, window=20, label="custom dqn"):
    episode_steps = metrics["episode_end_steps"]
    episode_returns = metrics["episode_returns"]

    rolling = pd.Series(episode_returns).rolling(window, min_periods=1).mean()

    plt.figure(figsize=(8, 4))
    plt.plot(episode_steps, episode_returns, alpha=0.3, label=f"{label} raw")
    plt.plot(episode_steps, rolling, label=f"{label} rolling mean ({window})")
    plt.xlabel("environment step")
    plt.ylabel("episode return")
    plt.title("Training curve")
    plt.legend()
    plt.tight_layout()
    plt.show()


def make_results_table(rows):
    df = pd.DataFrame(rows)

    summary = (
        df.groupby("algo")[["mean_return", "std_return", "mean_length", "crash_rate"]]
        .agg(["mean", "std"])
        .round(3)
    )

    return df, summary


def plot_metric_comparison(df):
    metrics = ["mean_return", "std_return", "mean_length", "crash_rate"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 4))

    for ax, metric in zip(axes, metrics):
        grouped = df.groupby("algo")[metric].mean()
        ax.bar(grouped.index, grouped.values)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=20)

    plt.tight_layout()
    plt.show()


def make_extension_summary_table(results_df):
    summary = (
        results_df.groupby(["variant", "eval_env"])[
            [
                "mean_return",
                "std_return",
                "mean_length",
                "crash_rate",
                "mean_speed",
                "actual_lane_change_rate",
            ]
        ]
        .agg(["mean", "std"])
        .round(3)
    )
    return summary


def plot_extension_metric(results_df, metric):
    summary = (
        results_df.groupby(["eval_env", "variant"])[metric]
        .agg(["mean", "std"])
        .reset_index()
    )
    mean_table = summary.pivot(index="eval_env", columns="variant", values="mean")
    std_table = summary.pivot(index="eval_env", columns="variant", values="std").fillna(0.0)

    ax = mean_table.plot(
        kind="bar",
        yerr=std_table,
        capsize=3,
        figsize=(10, 4),
    )
    ax.set_title(metric.replace("_", " ").title())
    ax.set_ylabel(metric)
    ax.set_xlabel("evaluation environment")
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.show()


def plot_observation_comparison(results_df, metric):
    plot_extension_metric(results_df, metric)


def plot_generalization_gap(results_df, metric, reference_env="shared"):
    reference = (
        results_df[results_df["eval_env"] == reference_env][["variant", "train_seed", metric]]
        .rename(columns={metric: "reference_value"})
    )
    merged = results_df.merge(reference, on=["variant", "train_seed"], how="left")
    merged = merged[merged["eval_env"] != reference_env].copy()
    merged["gap"] = merged[metric] - merged["reference_value"]

    summary = merged.groupby(["eval_env", "variant"])["gap"].agg(["mean", "std"]).reset_index()
    mean_table = summary.pivot(index="eval_env", columns="variant", values="mean")
    std_table = summary.pivot(index="eval_env", columns="variant", values="std").fillna(0.0)

    ax = mean_table.plot(
        kind="bar",
        yerr=std_table,
        capsize=3,
        figsize=(10, 4),
    )
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title(f"{metric} gap vs {reference_env}")
    ax.set_ylabel("delta")
    ax.set_xlabel("evaluation environment")
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.show()


def plot_pilot_safe_results(pilot_df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    for ax, metric in zip(axes, ["mean_return", "crash_rate"]):
        summary = pilot_df.groupby(["eval_env", "variant"])[metric].mean().reset_index()
        table = summary.pivot(index="eval_env", columns="variant", values=metric)
        table.plot(kind="bar", ax=ax)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_xlabel("evaluation environment")
        ax.tick_params(axis="x", rotation=20)

    plt.tight_layout()
    plt.show()


def show_gif(gif_path):
    gif_path = Path(gif_path)
    display(Image(filename=str(gif_path)))


def show_side_by_side_gifs(left_path, right_path, left_title="custom", right_title="sb3"):
    from IPython.display import HTML

    left_path = Path(left_path)
    right_path = Path(right_path)

    html = f"""
    <div style="display:flex; gap:24px; align-items:flex-start;">
        <div>
            <div style="font-weight:bold; margin-bottom:8px;">{left_title}</div>
            <img src="{left_path}" style="max-width:100%;">
        </div>
        <div>
            <div style="font-weight:bold; margin-bottom:8px;">{right_title}</div>
            <img src="{right_path}" style="max-width:100%;">
        </div>
    </div>
    """
    display(HTML(html))


def print_rollout_summary(result, label="model"):
    print(
        f"{label}: seed={result['seed']}, "
        f"return={result['return']:.2f}, "
        f"crashed={result['crashed']}"
    )


def print_failure_case(result, label="model"):
    if result is None:
        print(f"{label}: no failure case found in the tested seeds.")
    else:
        print(
            f"{label} failure case: seed={result['seed']}, "
            f"return={result['return']:.2f}, "
            f"crashed={result['crashed']}"
        )
        show_gif(result["gif_path"])
