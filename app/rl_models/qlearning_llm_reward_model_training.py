import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from pymongo import MongoClient

from guitar_hero.app.custom_env import GuitarHeroEnv
from guitar_hero.app.llm_variants.llm_reward_adapter import LLMRewardAdapter
from guitar_hero.app.rl_models.parallel_qlearning_training import epsilon_greedy, moving_average


"""Script per l'esecuzione dell'addestramento del modello Q-Learning nella variante che utilizza un LLM come Reward
Model"""


def main():
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

    # Hyperparametri
    epsilon_start = 1.0
    epsilon_min = 0.05

    hyperparameters = [
        {"model_label": "best_model", "alpha": 0.01, "gamma": 0.25, "epsilon_decay": 0.99},
        {"model_label": "max_f1_model", "alpha": 0.01, "gamma": 0.5, "epsilon_decay": 0.99},
        {"model_label": "baseline_model", "alpha": 0.1, "gamma": 0.75, "epsilon_decay": 0.99},
        {"model_label": "most_unstable_model", "alpha": 0.1, "gamma": 0.99, "epsilon_decay": 0.95},
        {"model_label": "worst_model", "alpha": 0.5, "gamma": 1.0, "epsilon_decay": 0.95}
    ]

    num_episodes = 500
    max_steps = 20000

    metrics = []

    for model in hyperparameters:

        # Q-Learning
        q_table_qlearning = defaultdict(lambda: np.zeros(n_actions))
        precision_qlearning = []
        rewards_qlearning = []
        recall_qlearning = []
        f1_score_qlearning = []

        alpha = model["alpha"]
        gamma = model["gamma"]
        epsilon_decay = model["epsilon_decay"]
        epsilon = epsilon_start

        reward_adapter = LLMRewardAdapter()

        print("Inizio addestramento Q-Learning...")
        for episode in range(num_episodes):
            state, _ = env.reset()
            total_reward = 0.0
            steps_episode = 0

            for step in range(max_steps):
                action = epsilon_greedy(q_table_qlearning, state, epsilon, n_actions)
                next_state, reward, terminated, _, _ = env.step(action)

                llm_reward = reward_adapter.get_reward(state, action)
                reward = 0.8 * float(reward) + 0.2 * llm_reward

                best_next = np.max(q_table_qlearning[next_state])
                q_table_qlearning[state][action] += alpha * (reward + gamma * best_next - q_table_qlearning[state][action])

                state = next_state
                total_reward += reward
                steps_episode += 1
                if terminated:
                    break

            epsilon = max(epsilon_min, epsilon * epsilon_decay)

            # Precisione episodio
            clicks = env.note_clicking_mode_counter["Perfect"] + env.note_clicking_mode_counter["Imperfect"] + env.note_clicking_mode_counter["Misclick"]
            precision = env.note_clicking_mode_counter["Perfect"] / clicks * 100 if clicks > 0 else 0
            precision_qlearning.append(precision)
            normalized_reward = total_reward / len(env.song_notes)
            rewards_qlearning.append(normalized_reward)
            recall = env.note_clicking_mode_counter["Perfect"] / len(env.song_notes) * 100
            recall_qlearning.append(recall)
            f1_score = (2 * ((precision / 100) * (recall / 100))/((precision / 100)+(recall / 100)))*100 if precision + recall > 0 else 0
            f1_score_qlearning.append(f1_score)

            print(
                f"Episode {episode:4d} | "
                f"Reward: {total_reward:8.1f} | "
                f"Epsilon: {epsilon:.3f}"
            )

            reward_adapter.save_cache()

        print("Termine addestramento Q-Learning...")

        precision_qlearning_smooth = moving_average(precision_qlearning, window=25)
        recall_qlearning_smooth = moving_average(recall_qlearning, window=25)
        f1_score_qlearning_smooth = moving_average(f1_score_qlearning, window=25)

        model_params = {
            "alpha": model["alpha"],
            "gamma": model["gamma"],
            "epsilon_decay": model["epsilon_decay"],
            "epsilon_min": epsilon_min,
            "num_episodes": num_episodes
        }

        qlearning_metrics = {
            "params": model_params,
            "precision": precision_qlearning_smooth,
            "recall": recall_qlearning_smooth,
            "f1_score": f1_score_qlearning_smooth,
        }

        metrics.append(qlearning_metrics)

        q_table_qlearning_filename = f"qlearning_{model['model_label']}_with_llm_reward_model.pkl"
        with open(q_table_qlearning_filename, "wb") as f:
            pickle.dump(dict(q_table_qlearning), f)

        print("Chiamate a LLM: ", reward_adapter.llm_calls)

        reward_adapter.save_cache()

    script_dir = Path(__file__).parent
    output_metrics_path = script_dir / "training_metrics"

    with open(output_metrics_path / f"metrics_qlearning_with_llm_reward_model.json", "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main()
