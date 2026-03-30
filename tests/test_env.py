import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("highway_env")

from common import make_env, seed_env
from shared_core_config import SHARED_CORE_CONFIG, SHARED_CORE_ENV_ID


def test_shared_core_config_loads():
    assert SHARED_CORE_ENV_ID == "highway-v0"
    assert SHARED_CORE_CONFIG["observation"]["type"] == "Kinematics"


def test_flattened_observation_shape_and_seed():
    env = make_env()
    try:
        observation_a, _ = seed_env(env, 7)
        observation_b, _ = seed_env(env, 7)
        assert observation_a.shape == (50,)
        np.testing.assert_allclose(observation_a, observation_b)
        assert env.action_space.n == 5
    finally:
        env.close()
