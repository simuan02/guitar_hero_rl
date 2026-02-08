import json
import os
import re

import dotenv
from google import genai
from google.genai import types
from pymongo import MongoClient
from guitar_hero.app.custom_env import GuitarHeroEnv


""" Questo script contiene il codice di esecuzione del testing del LLM Decision Maker, il codice per l'ottenimento della 
reward del LLM Reward Model e una funzione di utilità per il salvataggio dei dati di testing"""


dotenv.load_dotenv()

gemini_client = genai.Client(api_key=os.getenv("GUITAR_HERO_API_KEY"))


# Funzione che consente di ottenere l'azione dal LLM (decision maker core)
def get_llm_action(description):
    try:
        response = gemini_client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=description,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "Sei un bot di Guitar Hero. Analizza lo stato e decidi l'azione.\n"
                    "ESEMPI:\n"
                    "- Stato: verde: NOTA NEL TARGET, rossa: vuota... -> Azione: 1\n"
                    "- Stato: verde: vuota, rossa: nota in avvicinamento... -> Azione: 0\n"
                    "- Stato: verde: nota in avvicinamento... -> Azione: 0\n"
                    "- Stato: tutte vuote -> Azione: 0\n"
                    "- Stato: verde: nota lontana, rossa: vuota... -> Azione: 0\n"
                    "- Stato: verde: nota lontana, rossa: NOTA NEL TARGET... -> Azione: 2\n"
                    "- Stato: verde: NOTA NEL TARGET, rossa: NOTA NEL TARGET... -> Azione: 1\n"
                    "REGOLE: Se vedi NOTA NEL TARGET, NON rispondere mai 0.\n"
                    "Rispondi SOLO con un numero da 0 a 5.\n"
                    "Rispondi con un numero da 1 a 5 solo se vedi NOTA NEL TARGET\n"
                ),
                temperature=0.5,
                max_output_tokens=1000,
            )
        )
        # response_text = response.choices[0].message.content
        response_text = response.text.strip()

        clean_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()

        match = re.search(r'\d', clean_text)
        action = int(match.group()) if match else 0

        # print(f"Azione estratta: {action}")
        # print("Stato corrente: ", description)
        return action if action else 0
    except Exception as e:
        print(f"Errore Gemini: {e}")
        return 0


# Funzione che restituisce un valore relativo alla bontà dell'azione scelta, cioè consente, a partire da una descrizione
# testuale relativa all'outcome dell'azione effettuata, di ottenere la reward prevista dal LLM (reward model core)
def get_llm_reward(outcome_desc):
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=outcome_desc,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "Sei un reward model per Guitar Hero. "
                    "Dati lo stato e l'esito dell'azione dell'agente, restituisci un numero reale che rappresenta la "
                    "bontà dell'azione ai fini dell'apprendimento dell'agente. "
                    "L'obiettivo di gioco è massimizzare la reward totale. "
                    
                    "Valuta la bontà dell’azione su una scala continua da -10 a +10, tenendo conto di: "
                    "- accuratezza temporale "
                    "- coerenza con le note visibili "
                    "- penalizzazione per inattività quando necessaria"
                    
                    "Il valore restituito deve essere un singolo numero reale. "
                    
                    "Restituisci SOLO il numero reale, senza testo aggiuntivo. "
                ),
                temperature=0,
                max_output_tokens=1000,
            )
        )

        # response_text = response.choices[0].message.content
        response_text = response.text.strip()
        clean_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()

        match = re.search(r'-?\d+(?:\.\d+)?', clean_text)
        reward = float(match.group()) if match else 0.0
        return reward

    except Exception as e:
        print(f"Errore LLM reward: {e}")
        return 0.0


# Funzione che consente di salvare i risultati della fase di testing dei modelli
def save_test_results(new_test_data, model, filename="test_results.json"):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            data = json.load(f)
    else:
        data = {"experiment_metadata": {"model": model}, "tests": []}

    data["tests"].append(new_test_data)

    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


def main():
    # Connessione DB
    mongo_client = MongoClient("mongodb://localhost:27017/")
    db = mongo_client["songs"]
    collection = db["testing_songs"]
    collection_temp = db["testing_temp"]

    tests = list()

    print("Inizio test Variante 1: LLM come Decision Maker...")

    for song in collection.find():
        collection_temp.delete_many({})
        collection_temp.insert_one(song)
        env = GuitarHeroEnv(
            collection_temp,
            screen=None,
            clock=None,
            render_mode=None,
            discrete_state=True,
            model_testing=True
        )
        obs, _ = env.reset()
        terminated = False
        total_reward = 0.0
        metrics = {}

        while not terminated:

            accumulated_reward = 0.0

            description = env.get_llm_state_description(obs)

            full_prompt = f"{description}\nScegli l'azione (1-5 per colpire, 0 se vuoto):"

            action = get_llm_action(full_prompt)

            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                metrics = info
            accumulated_reward += reward

            for _ in range(6):
                if not terminated:
                    obs, r, terminated, truncated, info = env.step(0)
                    accumulated_reward += r
                    if terminated:
                        metrics = info
            accumulated_reward = round(accumulated_reward, 1)
            # print("Azione: ", action, " | Reward: ", accumulated_reward)
            total_reward += accumulated_reward

            notes = env.note_clicking_mode_counter["Perfect"] + env.note_clicking_mode_counter["Imperfect"] + \
                env.note_clicking_mode_counter["Missed"]
            """if notes > 0 and notes % 10 == 0:
                print("Raggiunta nota n. ", notes)"""

        print(f"Test terminato. Score totale: {env.score_var}")
        test_data = {"title": env.song['title'],
                     "precision": metrics['precision'],
                     "recall": metrics['recall'],
                     "f1_score": metrics['f1_score'],
                     "score": env.score_var}
        tests.append(test_data)
        env.close()

    save_test_results(tests, model="Gemini_3_Flash_Preview", filename="llm_first_variant_test_results.json")


if __name__ == "__main__":
    main()
