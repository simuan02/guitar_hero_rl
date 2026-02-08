import os
import sys
import tempfile

import numpy as np
import pygame
import gridfs
from stable_baselines3 import PPO
from guitar_hero.app.custom_env import GuitarHeroEnv
from pymongo import MongoClient
from guitar_hero.app.llm_variants.llm_agent_testing import save_test_results


def main(render_mode):
    client = MongoClient("mongodb://localhost:27017/")
    db = client["songs"]
    collection = db["testing_songs"]
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

    env = GuitarHeroEnv(collection, screen, clock, render_mode=render_mode)

    model_path = os.path.join(os.getcwd(), "guitar_hero_ppo_model.zip")
    try:
        model = PPO.load(model_path, env=env)
    except Exception as e:
        print(f"Caricamento standard fallito: {e}")
        print("Tentativo di caricamento forzato dei parametri...")
        model = PPO("MlpPolicy", env)
        model.set_parameters(model_path)

    tests = list()
    for i in range(10):
        obs, info = env.reset()
        terminated = False
        total_reward = 0
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

        while not terminated:
            if rendering == "human":
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        sys.exit(0)

            action, _states = model.predict(obs, deterministic=True)

            if isinstance(action, np.ndarray):
                action = int(action)

            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                metrics = info

            total_reward += reward

            env.render()

        total_reward = round(total_reward, 1)
        test_data = {"title": env.song['title'],
                     "precision": metrics['precision'],
                     "recall": metrics['recall'],
                     "f1_score": metrics['f1_score'],
                     "reward": total_reward}
        tests.append(test_data)

    save_test_results(tests, model="ppo", filename="../app/rl_models/testing_metrics/ppo_test_results.json")

    print("Test completato!")


if __name__ == "__main__":
    main(render_mode=None)
