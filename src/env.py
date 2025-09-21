import numpy as np
import pandas as pd
from src.utils import observ_table
from gymnasium import Env
from gymnasium.spaces import Discrete
class DetectionEnv(Env):
    def __init__(self):
        self.action_space = Discrete(2)   # {0: continue, 1: stop}
        self.observation_space = Discrete(5, start=1)
        self.state = 0
        self.Detection_length = 200
        self.t = 0

    def step(self, df, action, tau, observ_num):
        """Perform one environment step."""
        self.t += 1

        if action == 1:  # stop
            self.state = 2
            reward = 1 if self.t <= tau else 0
            if self.t >= 12:
                next_obs = observ_table(
                    df['thr-lev'].iloc[self.t-1],
                    df['thr-lev'].iloc[self.t],
                    df['thr-lev'].iloc[self.t+1]
                )
            else:
                next_obs = 1
        else:  # continue
            reward = 0.02 if self.t >= tau else 0
            self.state = 1 if self.t >= tau else 0

            if self.t >= 12:
                next_obs = observ_table(
                    df['thr-lev'].iloc[self.t-2],
                    df['thr-lev'].iloc[self.t-1],
                    df['thr-lev'].iloc[self.t]
                )
            else:
                next_obs = 1

        done = self.t > self.Detection_length - 2
        return self.state, reward, done, self.t, next_obs, {}

    def reset(self, df):
        """Reset environment with new dataset."""
        self.state = 0
        self.observation_space = 1
        self.Detection_length = len(df) - 1
        self.t = 0
