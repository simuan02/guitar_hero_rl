import json

from guitar_hero.app.llm_variants.llm_agent_testing import get_llm_reward


"""Script per la gestione della cache richiesta dal LLM Reward Model e l'ottenimento della descrizione del risultato 
di una certa azione. La seconda funzione è necessaria al LLM Reward Model per fornire un valore qualitativo dell'azione"""


class LLMRewardAdapter:
    def __init__(self, use_cache=True):
        self.use_cache = use_cache
        self.cache = self.load_cache()
        self.llm_calls = 0

    @staticmethod
    def _make_key(obs, action):
        key = (
            tuple(obs),
            action
        )
        return str(key)

    def get_reward(self, obs, action):
        key = self._make_key(obs, action)
        if self.use_cache and key in self.cache:
            return self.cache[key]

        outcome_desc = self.get_outcome_desc(obs, action)
        reward = get_llm_reward(outcome_desc)
        self.llm_calls += 1

        if self.use_cache:
            self.cache[key] = reward

        return reward

    def save_cache(self, filename="llm_cache.json"):
        with open(filename, "w") as f:
            json.dump(self.cache, f)

    @staticmethod
    def load_cache(filename="llm_cache.json"):
        try:
            with open(filename, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    @staticmethod
    def get_outcome_desc(obs, action):
        lanes = ["v", "r", "g", "b", "a"]
        status_map = {0: "empty", 1: "far", 2: "near",
                      3: "TARGET", 4: "missed"}
        click_action_result = {0: "Missed", 1: "Imperfect", 2: "Perfect"}

        state_desc = " | ".join([f"{lanes[i]}: {status_map[obs[i]]}" for i in range(5)])

        note_outcome = []
        for i in range(5):
            if action == 0:
                if obs[i] == 3:
                    note_outcome.append(f"{lanes[i]}: Missed")
                else:
                    note_outcome.append(f"{lanes[i]}: No_Action")
            else:
                if i + 1 == action:
                    if obs[i] == 0 or obs[i] == 1 or obs[i] == 4:
                        result_index = 0
                    elif obs[i] == 2:
                        result_index = 1
                    else:
                        result_index = 2
                    note_outcome.append(f"{lanes[i]}: {click_action_result[result_index]}")
                else:
                    note_outcome.append(f"{lanes[i]}: not_targeted")

        if action == 0:
            action_desc = "no_click"
        else:
            action_desc = f"click_{lanes[action - 1]}"

        outcome_desc = (
                f"STA:\n{state_desc}\n\n"
                f"ACT:\n{action_desc}\n\n"
                f"OUT:\n- " + "\n- ".join(note_outcome) + "\n"
        )

        return outcome_desc
