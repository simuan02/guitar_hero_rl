import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pickle
from pymongo import MongoClient
from guitar_hero.app.custom_env import GuitarHeroEnv
from parallel_qlearning_training import epsilon_greedy, moving_average, all_hyperparameters_combinations


""" Script per l'esecuzione parallela dell'addestramento di più modelli con configurazioni di iperparametri diverse 
dell'algoritmo SARSA"""


def train_single_config(params):
    alpha = params['alpha']
    gamma = params['gamma']
    epsilon_min = params['epsilon_min']
    epsilon_decay = params['epsilon_decay']
    num_episodes = params['num_episodes']

    client = MongoClient("mongodb://localhost:27017/")
    db = client["songs"]
    collection = db["songs"]

    env = GuitarHeroEnv(
        collection,
        screen=None,
        clock=None,
        render_mode=None,
        discrete_state=True
    )

    n_actions = env.action_space.n

    # SARSA
    q_table_sarsa = defaultdict(lambda: np.zeros(n_actions))
    precision_sarsa = []
    rewards_sarsa = []
    recall_sarsa = []
    f1_score_sarsa = []

    epsilon = 1.0
    max_steps = 20000

    print("Inizio addestramento agente SARSA...")
    for episode in range(num_episodes):
        state, _ = env.reset()
        action = epsilon_greedy(q_table_sarsa, state, epsilon, n_actions)
        total_reward = 0
        steps_episode = 0

        for step in range(max_steps):
            next_state, reward, terminated, _, _ = env.step(action)
            next_action = epsilon_greedy(q_table_sarsa, next_state, epsilon, n_actions)

            # Aggiornamento SARSA
            q_table_sarsa[state][action] += alpha * (
                        reward + gamma * q_table_sarsa[next_state][next_action] - q_table_sarsa[state][action])

            state = next_state
            action = next_action
            total_reward += reward
            steps_episode += 1
            if terminated:
                break

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        clicks = env.note_clicking_mode_counter["Perfect"] + env.note_clicking_mode_counter["Imperfect"] + \
                 env.note_clicking_mode_counter["Misclick"]
        precision = env.note_clicking_mode_counter["Perfect"] / clicks * 100 if clicks > 0 else 0
        precision_sarsa.append(precision)
        normalized_reward = total_reward / len(env.song_notes)
        rewards_sarsa.append(normalized_reward)
        recall = env.note_clicking_mode_counter["Perfect"] / len(env.song_notes) * 100
        recall_sarsa.append(recall)
        f1_score = (2 * ((precision / 100) * (recall / 100)) / (
                    (precision / 100) + (recall / 100))) * 100 if precision + recall > 0 else 0
        f1_score_sarsa.append(f1_score)
    print("Termine addestramento agente SARSA...")

    if num_episodes < 1000:
        window_size = 25
    else:
        window_size = 50

    precision_sarsa_smooth = moving_average(precision_sarsa, window_size)
    recall_sarsa_smooth = moving_average(recall_sarsa, window_size)
    f1_score_sarsa_smooth = moving_average(f1_score_sarsa, window_size)

    script_dir = Path(__file__).parent
    model_directory = script_dir / "models"

    q_table_qlearning_filename = "sarsa_a" + str(alpha) + "g" + str(gamma) + "ed" + str(epsilon_decay) + ".pkl"
    with open(model_directory / q_table_qlearning_filename, "wb") as f:
        pickle.dump(dict(q_table_sarsa), f)

    return {
        "params": {
            "alpha": alpha,
            "gamma": gamma,
            "epsilon_min": epsilon_min,
            "epsilon_decay": epsilon_decay,
            "num_episodes": num_episodes
        },
        "precision": precision_sarsa_smooth,
        "recall": recall_sarsa_smooth,
        "f1_score": f1_score_sarsa_smooth
    }


def main():
    """hyperparameter_dict_first_phase = {
        "alpha": [0.01, 0.05, 0.1, 0.3, 0.5],
        "gamma": [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1],
        "epsilon_min": [0.05],
        "epsilon_decay": [0.9, 0.95, 0.99, 0.999],
        "num_episodes": [3000]
    }

    param_samples = sample_hyperparameters(hyperparameter_dict_first_phase, n_samples=50)"""

    hyperparameter_dict_second_phase = {
            "alpha": [0.01, 0.05, 0.1],
            "gamma": [0, 0.25, 0.5, 0.75],
            "epsilon_min": [0.05],
            "epsilon_decay": [0.9, 0.95, 0.99],
            "num_episodes": [500]
        }

    param_samples = all_hyperparameters_combinations(hyperparameter_dict_second_phase)

    # baseline_param_samples = [[0.1, 0.95, 0.05, 0.99, 500]]

    results = []

    with ProcessPoolExecutor(max_workers=os.cpu_count() - 1) as executor:
        futures = [executor.submit(train_single_config, p) for p in param_samples]

        for future in as_completed(futures):
            results.append(future.result())

    script_dir = Path(__file__).parent
    output_metrics_path = script_dir / "training_metrics"
    with open(output_metrics_path / f"metrics_sarsa_second_phase.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
