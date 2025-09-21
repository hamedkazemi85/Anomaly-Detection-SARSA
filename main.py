import numpy as np
import pandas as pd
from src.sarsa import SARSA, SARSA_detection

if __name__ == "__main__":
    # Load attack dataset
    df_attack15 = pd.read_csv("data/QtableAttack_Onlyphi15.csv", header=None)
    df_attack15.columns = ["thr-lev", "time"]

    df_fault30 = pd.read_csv("data/QtableFault_sen30.csv", header=None)
    df_fault30.columns = ["thr-lev", "time"]

    # Load pre-trained Q-table
    df_action_values_attack_learned = pd.read_csv("data/action_values_Attack_theta.csv", header=None)
    action_values_attack_learned = df_action_values_attack_learned.iloc[:, 1:].to_numpy()

    # Run online detection
    t_detected, action, state = SARSA_detection(action_values_attack_learned, df_attack15)

    print("Anomaly detection completed.")
    print(f"Detection stopped at time step: {t_detected}")
    print(f"Final action: {action}, Final state: {state}")
