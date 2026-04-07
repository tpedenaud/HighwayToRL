from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
