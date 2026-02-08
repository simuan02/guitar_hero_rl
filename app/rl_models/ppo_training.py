import json

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env

from guitar_hero.app.custom_env import GuitarHeroEnv
from pymongo import MongoClient
from parallel_qlearning_training import moving_average


class GuitarHeroCallback(BaseCallback):
    def __init__(self, n_max_episodes, verbose=0):
        super(GuitarHeroCallback, self).__init__(verbose)
        self.n_max_episodes = n_max_episodes
        self.episode_count = 0
        # Liste per memorizzare le metriche
        self.precision_list = []
        self.recall_list = []
        self.f1_score_list = []

    def _on_step(self) -> bool:
        if "dones" in self.locals and self.locals["dones"][0]:
            self.episode_count += 1

            metrics = self.locals["infos"][0]

            self.precision_list.append(metrics["precision"])
            self.recall_list.append(metrics["recall"])
            self.f1_score_list.append(metrics["f1_score"])

            if self.episode_count >= self.n_max_episodes:
                print("Limite di ", self.n_max_episodes, " episodi raggiunto. Interruzione training...")
                return False

        return True


def main():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["songs"]
    collection = db["songs"]

    # Crea l'ambiente in modalità "AI" (senza render_mode="human")
    env = GuitarHeroEnv(collection, screen=None, clock=None, render_mode=None)

    # Verifica se l'ambiente rispetta gli standard
    print("Verifica ambiente in corso...")
    check_env(env)
    print("Ambiente valido!")

    # Crea il modello AI (PPO)
    model = PPO("MlpPolicy", env, verbose=0)

    max_episodes = 50
    callback = GuitarHeroCallback(n_max_episodes=max_episodes)

    print("Inizio addestramento...")
    model.learn(total_timesteps=200000000, callback=callback)

    precision_smooth = moving_average(callback.precision_list, window=5)
    recall_smooth = moving_average(callback.recall_list, window=5)
    f1score_smooth = moving_average(callback.f1_score_list, window=5)

    ppo_metrics = {
        "precision": precision_smooth,
        "recall": recall_smooth,
        "f1_score": f1score_smooth
    }

    with open("training_metrics/metrics_ppo.json", "w") as f:
        json.dump(ppo_metrics, f)

    print("Addestramento completato!")

    model.save("guitar_hero_ppo_model")
    print("Modello salvato con successo!")


if __name__ == "__main__":
    main()
