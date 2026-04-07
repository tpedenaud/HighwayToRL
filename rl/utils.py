import numpy as np

def preprocess_observation(obs):
    return np.asarray(obs, dtype=np.float32).flatten()