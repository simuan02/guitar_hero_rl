import pickle
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import gridfs
import numpy as np
import pygame
from pymongo import MongoClient

from guitar_hero.app.custom_env import GuitarHeroEnv
from guitar_hero.app.llm_variants.llm_agent_testing import save_test_results


def greedy_policy(q, state):
    return int(np.argmax(q[state]))


def main(model_path, model_name, render_mode):
    client = MongoClient("mongodb://localhost:27017/")
    db = client["songs"]
    collection = db["testing_songs"]
    collection_temp = db["testing_temp"]
    fs = gridfs.GridFS(db)
    rendering = render_mode if render_mode else ""

    if rendering == "human":
        pygame.init()
        screen = pygame.display.set_mode((700, 750))
        pygame.display.set_caption("Guitar Hero")
        clock = pygame.time.Clock()
    else:
        screen = None
        clock = None

    q_table_path = model_path / model_name

    print(model_path / model_name)

    with open(q_table_path, "rb") as f:
        loaded_q = pickle.load(f)
        q_table = defaultdict(lambda: np.zeros(env.action_space.n))
        q_table.update(loaded_q)

        tests = list()

        for song in collection.find():
            collection_temp.delete_many({})
            collection_temp.insert_one(song)

            env = GuitarHeroEnv(
                collection_temp,
                screen=screen,
                clock=clock,
                render_mode=rendering,
                discrete_state=True
            )

            state, _ = env.reset()
            terminated = False
            metrics = {}

            if rendering == "human":
                audio_id = env.song["audio_file_id"]
                if audio_id:
                    audio_file = fs.get(audio_id)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".ogg") as temp_audio:
                        temp_audio.write(audio_file.read())
                        temp_path = temp_audio.name
                    pygame.mixer.music.load(temp_path)
                    pygame.mixer.music.play()

            total_reward = 0.0

            while not terminated:
                if rendering == "human":
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit(0)

                action = greedy_policy(q_table, state)

                next_state, reward, terminated, _, info = env.step(action)
                if terminated:
                    metrics = info

                total_reward += reward
                state = next_state

                env.render()

            total_reward = round(total_reward, 1)
            normalized_reward = total_reward / len(env.song_notes)

            print(f"Normalized reward: {normalized_reward:.3f}")
            print(metrics)
            test_data = {"title": env.song['title'],
                         "precision": metrics['precision'],
                         "recall": metrics['recall'],
                         "f1_score": metrics['f1_score'],
                         "score": env.score_var}
            tests.append(test_data)

        print("testing_metrics" + model_name + "_test_results.json")
        save_test_results(tests, model=model_name, filename="testing_metrics/" + model_name + "_test_results.json")
        pygame.quit()


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    model_path = script_dir / "models"

    """models = ["qlearning_a0.1g0.75ed0.99.pkl",
                   "qlearning_a0.1g0.99ed0.95.pkl",
                   "qlearning_a0.01g0.5ed0.99.pkl",
                   "qlearning_a0.01g0.25ed0.99.pkl",
                   "qlearning_a0.5g1.0ed0.95.pkl"]"""

    """models = ["sarsa_a0.1g0.75ed0.99.pkl",
                   "sarsa_a0.01g0.5ed0.95.pkl",
                   "sarsa_a0.3g0.99ed0.95.pkl",
                   "sarsa_a0.05g0.25ed0.95.pkl"]"""

    models = ["q_table_qlearning_baseline_model_with_llm_reward_model.pkl",
              "q_table_qlearning_best_model_with_llm_reward_model.pkl",
              "q_table_qlearning_max_f1_model_with_llm_reward_model.pkl",
              "q_table_qlearning_most_unstable_model_with_llm_reward_model.pkl",
              "q_table_qlearning_worst_model_with_llm_reward_model.pkl"]
    for model in models:
        main(model_path, model_name=model, render_mode=None)
