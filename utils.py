import numpy as np

def observ_table(pre_observation, observation, next_observation):
    """Map three consecutive threshold levels into a unique observation index (1–64)."""
    return (pre_observation - 1) * 16 + (observation - 1) * 4 + next_observation
