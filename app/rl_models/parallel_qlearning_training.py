import itertools
import json
import os
from collections import defaultdict
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pickle
from pymongo import MongoClient
from guitar_hero.app.custom_env import GuitarHeroEnv


""" Script per l'esecuzione parallela dell'addestramento di più modelli con configurazioni di iperparametri diverse 
dell'algoritmo Q-Learning"""


# Funzione epsilon-greedy con leggero bias sul non fare nulla in caso di esplorazione
def epsilon_greedy(q, state, epsilon, n_actions):
    if random.random() < epsilon:
        if random.random() < 0.7:
            return 0
        else:
            return random.randint(1, n_actions - 1)
    return int(np.argmax(q[state]))


def moving_average(data, window=50):
    data_window_avg = list()
    for i in range(0, len(data), window):
        data_window = data[i:window + i]
        data_window_avg.append(sum(data_window) / len(data_window))
    return data_window_avg


def sample_hyperparameters(hyperparameter_dict, n_samples):
    samples = []
    for i in range(n_samples):
        sample = {
            "alpha": float(random.choice(hyperparameter_dict["alpha"])),
            "gamma": float(random.choice(hyperparameter_dict["gamma"])),
            "epsilon_min": float(random.choice(hyperparameter_dict["epsilon_min"])),
            "epsilon_decay": float(random.choice(hyperparameter_dict["epsilon_decay"])),
            "num_episodes": int(random.choice(hyperparameter_dict["num_episodes"]))
        }
        if sample not in samples:
            samples.append(sample)
        else:
            n_samples -= 1
    return samples


def all_hyperparameters_combinations(hyperparameter_dict):
    keys = ["alpha", "gamma", "epsilon_min", "epsilon_decay", "num_episodes"]

    combinations = list(itertools.product(hyperparameter_dict["alpha"], hyperparameter_dict["gamma"],
                                          hyperparameter_dict["epsilon_min"],
                                          hyperparameter_dict["epsilon_decay"],
                                          hyperparameter_dict["num_episodes"]))
    return [
        dict(zip(keys, values))
        for values in combinations
    ]


def train_single_config(params):
    client = MongoClient("mongodb://localhost:27017/")
    db = client["songs"]
    collection = db["songs"]

    print(params)

    alpha = params['alpha']
    gamma = params['gamma']
    epsilon_min = params['epsilon_min']
    epsilon_decay = params['epsilon_decay']
    num_episodes = params['num_episodes']

    env = GuitarHeroEnv(
        collection,
        screen=None,
        clock=None,
        render_mode=None,
        discrete_state=True
    )

    n_actions = env.action_space.n

    # Q-Learning
    q_table_qlearning = defaultdict(lambda: np.zeros(n_actions))
    precision_qlearning = []
    rewards_qlearning = []
    recall_qlearning = []
    f1_score_qlearning = []

    epsilon = 1.0

    print("Inizio addestramento Q-Learning...")
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        steps_episode = 0

        for step in range(20000):
            action = epsilon_greedy(q_table_qlearning, state, epsilon, n_actions)
            next_state, reward, terminated, _, _ = env.step(action)

            best_next = np.max(q_table_qlearning[next_state])
            q_table_qlearning[state][action] += alpha * (reward + gamma * best_next - q_table_qlearning[state][action])

            state = next_state
            total_reward += reward
            steps_episode += 1
            if terminated:
                break

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Precisione episodio
        clicks = env.note_clicking_mode_counter["Perfect"] + env.note_clicking_mode_counter["Imperfect"] + \
                 env.note_clicking_mode_counter["Misclick"]
        precision = env.note_clicking_mode_counter["Perfect"] / clicks * 100 if clicks > 0 else 0
        precision_qlearning.append(precision)
        normalized_reward = total_reward / len(env.song_notes)
        rewards_qlearning.append(normalized_reward)
        recall = env.note_clicking_mode_counter["Perfect"] / len(env.song_notes) * 100
        recall_qlearning.append(recall)
        f1_score = (2 * ((precision / 100) * (recall / 100)) / (
                (precision / 100) + (recall / 100))) * 100 if precision + recall > 0 else 0
        f1_score_qlearning.append(f1_score)

        """print(
            f"Episode {episode:4d} | "
            f"Reward: {total_reward:8.1f} | "
            f"Epsilon: {epsilon:.3f}"
        )"""
    print("Termine addestramento Q-Learning...")

    if num_episodes < 1000:
        window_size = 25
    else:
        window_size = 50

    precision_qlearning_smooth = moving_average(precision_qlearning, window_size)
    recall_qlearning_smooth = moving_average(recall_qlearning, window_size)
    f1_score_qlearning_smooth = moving_average(f1_score_qlearning, window_size)

    script_dir = Path(__file__).parent
    model_directory = script_dir / "models"

    q_table_qlearning_filename = "qlearning_a" + str(alpha) + "g" + str(gamma) + "ed" + str(epsilon_decay) + ".pkl"
    with open(model_directory / q_table_qlearning_filename, "wb") as f:
        pickle.dump(dict(q_table_qlearning), f)

    return {
        "params": {
            "alpha": alpha,
            "gamma": gamma,
            "epsilon_min": epsilon_min,
            "epsilon_decay": epsilon_decay,
            "num_episodes": num_episodes
        },
        "precision": precision_qlearning_smooth,
        "recall": recall_qlearning_smooth,
        "f1_score": f1_score_qlearning_smooth
    }


def main():
    # Fine-Tuning Phase 1
    """hyperparameter_dict_first_phase = {
        "alpha": [0.01, 0.05, 0.1, 0.3, 0.5],
        "gamma": [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1],
        "epsilon_min": [0.05],
        "epsilon_decay": [0.9, 0.95, 0.99, 0.999],
        "num_episodes": [3000]
    }

    param_samples = sample_hyperparameters(hyperparameter_dict_first_phase, n_samples=35)"""

    # Fine-Tuning Phase 2
    """
    hyperparameter_dict_second_phase = {
        "alpha": [0.01, 0.05],
        "gamma": [0, 0.25, 0.5, 0.75, 0.95],
        "epsilon_min": [0.05],
        "epsilon_decay": [0.95, 0.99],
        "num_episodes": [3000]
    }
    param_samples = all_hyperparameters_combinations(hyperparameter_dict_second_phase)"""

    # Baseline parameters combination
    param_samples = [{"alpha": 0.1,
                      "gamma": 0.75,
                      "epsilon_min": 0.05,
                      "epsilon_decay": 0.99,
                      "num_episodes": 500}]

    results = []

    with ProcessPoolExecutor(max_workers=os.cpu_count() - 1) as executor:
        futures = [executor.submit(train_single_config, p) for p in param_samples]

        for future in as_completed(futures):
            results.append(future.result())

    script_dir = Path(__file__).parent
    output_metrics_path = script_dir / "training_metrics"
    with open(output_metrics_path / f"metrics_qlearning_baseline.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
