import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt


"""Questo script consente di aggregare i risultati di testing di modelli con combinazioni di iperparametri differenti,
mostrare in forma tabellare i risultati di ogni singolo modello su ogni canzone e ottenere i risultati migliori 
ottenuti da tutte le combinazioni di iperparametri valutate su ogni canzone"""


def aggregation_func(algorithm, labels, metrics_path=None, algorithm_savename=None):
    base_directory = Path(__file__).parent
    testing_directory = base_directory / 'testing_metrics'
    models = {}
    for label in labels:
        if not metrics_path:
            model_path = testing_directory / f"{algorithm}_{label}_test_results.json"
        else:
            model_path = metrics_path
        with open(model_path, "r") as f:
            model = json.load(f)
            models[label] = model["tests"][0]
    if not algorithm_savename:
        algorithm_savename = algorithm
    with open(testing_directory / f"{algorithm_savename}_testing_metrics.json", "w") as f:
        json.dump(models, f, indent=4)
    return models


def show_model_stats(models, algorithm):
    base_directory = Path(__file__).parent
    testing_tables_directory = base_directory / 'testing_metrics' / f"{algorithm}_tables"
    testing_tables_directory.mkdir(parents=True, exist_ok=True)
    for label, tests in models.items():
        rows = []
        for song in tests:
            row = {
                ("Canzone"): song["title"],
                ("Precisione"): song["precision"],
                ("Recall"): song["recall"],
                ("F1-Score"): song["f1_score"],
                ("Score"): song["score"]
            }
            rows.append(row)
        data_frame = pd.DataFrame(rows)
        show_stats_table(data_frame, testing_tables_directory, algorithm, label)


def show_stats_table(data_frame, testing_tables_directory, algorithm, label):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")

    table = ax.table(
        cellText=data_frame.values,
        colLabels=list(data_frame.columns),
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    plt.savefig(testing_tables_directory / f"metrics_table_{algorithm}_{label}.png", dpi=300, bbox_inches="tight")
    plt.close()

    format_dict = {
        "Precisione": '{:.2f}',
        "Recall": '{:.2f}',
        "F1-Score": '{:.2f}',
        "Score": '{:.0f}'
    }

    latex_code = data_frame.style.format(format_dict).hide(axis='index').to_latex()
    print(latex_code)


def get_best_song_stats(models, algorithm):
    base_directory = Path(__file__).parent
    testing_tables_directory = base_directory / 'testing_metrics'

    max_score_per_song = {}

    for label, tests in models.items():
        for song in tests:
            if song["title"] not in max_score_per_song or song["score"] > max_score_per_song[song["title"]]["Score"]:
                max_score_per_song[song["title"]] = {"Score": int(song["score"]),
                                                     "Modello": label,
                                                     "Precisione": song["precision"],
                                                     "Recall": song["recall"],
                                                     "F1-Score": song["f1_score"]}

    data_frame = pd.DataFrame.from_dict(max_score_per_song, orient='index').reset_index().rename(columns={'index': 'Canzone'})
    show_stats_table(data_frame, testing_tables_directory=testing_tables_directory / f"{algorithm}_tables", algorithm=algorithm,
                     label="best_stats")

    with open(testing_tables_directory / f"{algorithm}_best_song_stats.json", "w") as outfile:
        json.dump(max_score_per_song, outfile, indent=4)


if __name__ == '__main__':
    # Q-Learning Standard
    model_labels = ['baseline', 'best_f1_score_mean', 'max_f1_score', 'worst', 'most_unstable']
    models = aggregation_func("qlearning", model_labels)
    show_model_stats(models, "qlearning")
    get_best_song_stats(models, "qlearning")

    # SARSA
    model_labels = ['baseline', 'best_f1_score_mean', 'stablest', 'worst']
    models = aggregation_func("sarsa", model_labels)
    show_model_stats(models, "sarsa")
    get_best_song_stats(models, "sarsa")

    # Q-Learning con LLM Reward Model
    model_labels = ['baseline_model_with_llm_reward_model', 'best_model_with_llm_reward_model',
                    'max_f1_model_with_llm_reward_model', 'worst_model_with_llm_reward_model',
                    'most_unstable_model_with_llm_reward_model']
    models = aggregation_func("qlearning", model_labels, algorithm_savename="qlearning_llm_reward_models")
    show_model_stats(models, "qlearning_llm_reward_models")
    get_best_song_stats(models, "qlearning_llm_reward_models")

    # LLM come Decision Maker
    base_directory = Path(__file__).parent.parent
    metrics_file = base_directory / "llm_variants/llm_first_variant_test_results.json"
    model_labels = ["llm_decision_model"]
    models = aggregation_func("llm_decision_model", model_labels, metrics_path=metrics_file)
    show_model_stats(models, "llm_decision_model")
    get_best_song_stats(models, "llm_decision_model")
