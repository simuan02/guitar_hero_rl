import json
import numpy as np

""" Script per la modifica dei file .JSON risultanti dalla fase di training, al fine di aggiungere campi utili per 
il confronto, quali la f1-score media e la precision media"""

def main(path):

    WINDOW_SIZE = 8

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    unique_samples = list()
    unique_params = set()

    for i, sample in enumerate(data):
        sample["sample_number"] = i+1
        sample["precision"] = [round(x, 2) for x in sample["precision"]]
        sample["recall"] = [round(x, 2) for x in sample["recall"]]
        sample["f1_score"] = [round(x, 2) for x in sample["f1_score"]]

        f1 = np.array(sample["f1_score"], dtype=float)
        f1_tail = f1[-WINDOW_SIZE:]
        precision = np.array(sample["precision"], dtype=float)
        precision_tail = precision[-WINDOW_SIZE:]
        recall = np.array(sample["recall"], dtype=float)
        recall_tail = recall[-WINDOW_SIZE:]

        sample[f"f1_mean_last_episodes"] = round(float(np.mean(f1_tail)), 2)
        sample[f"f1_std_last_episodes"] = round(float(np.std(f1_tail)), 2)
        sample["best_f1_score"] = max(sample["f1_score"])
        sample[f"precision_mean_last_episodes"] = round(float(np.mean(precision_tail)), 2)
        sample[f"recall_mean_last_episodes"] = round(float(np.mean(recall_tail)), 2)

        sample_params = sample["params"]
        sample_params_tuple = (sample_params["alpha"], sample_params["gamma"], sample_params["epsilon_min"],
                               sample_params["epsilon_decay"], sample_params["num_episodes"])

        if sample_params_tuple not in unique_params:
            unique_params.add(sample_params_tuple)
            unique_samples.append(sample)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(unique_samples, f, indent=4)


if __name__ == "__main__":
    main("metrics_qlearning_with_llm_reward_model.json")
