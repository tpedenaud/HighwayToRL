import numpy as np
import pandas as pd

from evaluate_models import summarize_episodes
from report_results import explain_failure


def test_summarize_episodes_returns_expected_values():
    episodes_df = pd.DataFrame(
        [
            {"reward": 1.0, "episode_length": 10, "crashed": 0, "mean_speed": 25.0},
            {"reward": 3.0, "episode_length": 12, "crashed": 1, "mean_speed": 24.0},
        ]
    )
    summary = summarize_episodes(episodes_df, "custom_dqn", 0)
    assert summary["mean_reward"] == 2.0
    assert summary["reward_std"] == 1.0
    assert summary["crash_rate"] == 0.5


def test_failure_explanation_prefers_front_vehicle_case():
    final_observation = np.zeros((10, 5), dtype=np.float32)
    final_observation[1] = [1.0, 0.1, 0.0, -0.1, 0.0]
    rollout = {
        "crashed": True,
        "observations": [final_observation.reshape(-1)],
        "action_labels": ["IDLE"],
    }
    explanation = explain_failure(rollout)
    assert "vehicle ahead" in explanation or "late reaction" in explanation
