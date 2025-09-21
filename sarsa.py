import numpy as np
import pandas as pd
from tqdm import tqdm
from src.env import DetectionEnv
from src.utils import observ_table

def policy(observ_num, action_values, epsilon):
    """ε-greedy policy: choose min action-value (exploit) or random (explore)."""
    if np.random.random() < epsilon:
        return np.random.randint(2)
    av = action_values[observ_num]
    return pd.Series(av).idxmin()

def SARSA(action_values, episodes, tau, df, epsilon=0.2, alpha=0.1, gamma=1):
    """Offline Q-learning (SARSA training Q-table)."""
    env = DetectionEnv()

    for _ in tqdm(range(1, episodes + 1)):
        env.reset(df)
        action = 0
        done = False
        observ_num = 1

        while not done:
            next_state, reward, done, t, next_obs, _ = env.step(df, action, tau, observ_num)
            if action == 1:
                action_values[observ_num][action] += alpha * (reward - action_values[observ_num][action])
                action = 0
            else:
                next_action = policy(next_obs, action_values, epsilon)
                qsa = action_values[observ_num][action]
                next_qsa = action_values[next_obs][next_action]
                action_values[observ_num][action] = qsa + alpha * (reward + gamma * next_qsa - qsa)
                action = next_action
                observ_num = next_obs
    return action_values

def SARSA_detection(action_values_learned, df):
    """Run anomaly detection on incoming data with pre-trained Q-table."""
    env = DetectionEnv()
    env.reset(df)
    t = 2
    packet_length = len(df)
    done = False

    while not done:
        obs3 = df['thr-lev'].iloc[t]
        obs2 = df['thr-lev'].iloc[t - 1]
        obs1 = df['thr-lev'].iloc[t - 2]
        observ_num = observ_table(obs1, obs2, obs3)

        av = action_values_learned[observ_num]
        action = pd.Series(av).idxmin()

        if action == 1 and t >= 12:
            state = 2
        else:
            action = 0
            state = 0

        done = t >= packet_length - 2 or action == 1
        t += 1

    return t, action, state
