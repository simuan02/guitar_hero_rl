from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import json

import numpy as np
import pandas as pd


""" Questo script consente il confronto tabellare e grafico tra diversi modelli"""


def load_json(filename):
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(filename, " non trovato")
        return None


def legend_title(model_data):
    params = model_data["params"]
    return (f"A: {params['alpha']} - G: {params['gamma']} - E-Min: {params['epsilon_min']} - E-Decay: "
            f"{params['epsilon_decay']}")


def aggregate_f1(models_by_param):
    param_values = []
    f1_means = []
    f1_stds = []

    for param_value, models in sorted(models_by_param.items()):
        f1_values = [m["f1_mean_last_episodes"] for m in models]

        param_values.append(param_value)
        f1_means.append(np.mean(f1_values))
        f1_stds.append(np.std(f1_values))

    return param_values, f1_means, f1_stds


def plot_aggregate_f1(param_values, f1_mean, f1_std, x_label, param_name, algorithm, filename):
    plt.figure(figsize=(8, 5))
    plt.errorbar(param_values, f1_mean, yerr=f1_std, fmt="-o", capsize=6)

    plt.xlabel(x_label)
    plt.ylabel("F1-score medio")
    plt.title(f"Trend delle prestazioni dell'algoritmo {algorithm} al variare di {param_name}")

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()


def plot_metrics(data_to_plot, metrics, titles, savepaths):
    for metric, title, savepath in zip(metrics, titles, savepaths):
        fig, ax = plt.subplots(figsize=(14, 7))

        for data, label, color, style in data_to_plot:
            if data and metric in data:
                values = data[metric]
                x_axis = np.linspace(0, 100, len(values))

                ax.plot(x_axis, values, label=label, color=color, linestyle=style, linewidth=2)

        ax.set_title(title)
        ax.set_xlabel("Progresso Training (%)")
        ax.set_ylabel("Valore (%)")

        ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.01))

        ax.grid(True, linestyle=":", alpha=0.6)

        fig.subplots_adjust(left=0.05, right=0.7)

        fig.savefig(savepath, dpi=300, bbox_inches="tight")

        plt.show()
        plt.close(fig)


def hyperparameter_comparison(groups, hyperparameter, metrics, directory, colors):
    for model_group in groups.items():
        titles = [f"Confronto Precision ({hyperparameter} = {model_group[0]})",
                  f"Confronto Recall ({hyperparameter} = {model_group[0]})",
                  f"Confronto F1-Score ({hyperparameter} = {model_group[0]})"]
        savepaths = [(directory / f"q_learning_algorithms_{hyperparameter}_{model_group[0]}_"
                                  f"comparison_{metric}.png") for metric in metrics]
        plot_data = []
        for i, model in enumerate(model_group[1]):
            plot_data.append((model, legend_title(model), colors[i % len(colors)], "-"))
        plot_metrics(plot_data, metrics, titles, savepaths)


def best_models_per_hyperparameter_comp(models, hyperparameter, metrics, directory, colors, algorithm="", phase=""):
    best_models_per_param = []
    for i, model_group in enumerate(models.values()):
        best = max(model_group, key=lambda s: s["f1_mean_last_episodes"])
        best_models_per_param.append((best, legend_title(best), colors[i % len(colors)], "-"))
    # Confronto modelli migliori per ogni valore di alpha analizzato
    titles = [f"Confronto Precision (best models per {hyperparameter})",
              f"Confronto Recall (best models per {hyperparameter})",
              f"Confronto F1-Score (best models per {hyperparameter})"]
    savepaths = [(directory / f"{algorithm}_algorithm_best_models_per_{hyperparameter}_"
                              f"{phase}{metric}.png") for metric in metrics]
    plot_metrics(best_models_per_param, metrics, titles, savepaths)


def model_plot_comparison(data_path, algorithm="qlearning", phase="first"):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    models = data

    base_directory = Path(__file__).resolve().parent
    plots_directory = base_directory.parent / "plots" / f"{algorithm}_training"
    tables_directory = base_directory/ "tables"

    models.sort(
        key=lambda s: s["f1_mean_last_episodes"],
        reverse=True
    )
    for model in models:
        print(model["sample_number"], model["params"],
              " - F1-Score Max: ", model["best_f1_score"],
              " - F1-Score mean: ", model["f1_mean_last_episodes"],
              " - F1-Score SD: ", model["f1_std_last_episodes"],
              " - Precision: ", model["precision_mean_last_episodes"],
              " - Recall: ", model["recall_mean_last_episodes"])

    colors = plt.cm.tab20.colors
    latex_rows = []
    table_rows = []

    for model in models:
        row = {
            ("ID", ""): str(model["sample_number"]),
            ("Iperparametri", "alpha"): model["params"]["alpha"],
            ("Iperparametri", "gamma"): model["params"]["gamma"],
            ("Iperparametri", "Epsilon Decay"): model["params"]["epsilon_decay"],
            ("Iperparametri", "Epsilon Min"): model["params"]["epsilon_min"],
            ("Metriche", "Precisione Media"): model["precision_mean_last_episodes"],
            ("Metriche", "Recall Media"): model["recall_mean_last_episodes"],
            ("Metriche", "F1-Score media"): model["f1_mean_last_episodes"],
            ("Metriche", "F1-Score massima"): model["best_f1_score"],
            ("Metriche", "F1-Score SD"): model["f1_std_last_episodes"]
        }
        latex_rows.append(row)

    for model in models:
        row = {
            ("ID"): str(model["sample_number"]),
            ("α"): model["params"]["alpha"],
            ("γ"): model["params"]["gamma"],
            ("ε-Decay"): model["params"]["epsilon_decay"],
            ("ε Min"): model["params"]["epsilon_min"],
            ("Prec. Media"): model["precision_mean_last_episodes"],
            ("Rec. Media"): model["recall_mean_last_episodes"],
            ("F1 Media"): model["f1_mean_last_episodes"],
            ("F1 Max"): model["best_f1_score"],
            ("F1 SD"): model["f1_std_last_episodes"]
        }
        table_rows.append(row)

    data_frame = pd.DataFrame(latex_rows)
    data_frame.columns = pd.MultiIndex.from_tuples(data_frame)

    data_frame_to_table = pd.DataFrame(table_rows)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")

    table = ax.table(
        cellText=data_frame_to_table.values,
        colLabels=data_frame_to_table.columns,
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    plt.savefig(tables_directory / f"metrics_table_{algorithm}_{phase}_phase.png", dpi=300, bbox_inches="tight")
    plt.close()

    top_models = data_frame.head(10)
    last_models = data_frame.tail(10)

    tabella_relazione = pd.concat([top_models, last_models])
    print(tabella_relazione)

    format_dict = {
        ("Iperparametri", "alpha"): '{:.2f}',
        ("Iperparametri", "gamma"): '{:.2f}',
        ("Iperparametri", "Epsilon Decay"): '{:.3f}',
        ("Iperparametri", "Epsilon Min"): '{:.2f}',
        ("Metriche", "Precisione Media"): '{:.2f}',
        ("Metriche", "F1-Score media"): '{:.2f}',
        ("Metriche", "F1-Score SD"): '{:.2f}',
        ("Metriche", "Recall Media"): '{:.2f}',
        ("Metriche", "F1-Score massima"): '{:.2f}'
    }

    latex_code = tabella_relazione.style.format(format_dict).hide(axis='index').to_latex()
    print(latex_code)

    alpha_groups = defaultdict(list)
    for model in models:
        alpha_groups[model["params"]["alpha"]].append(model)

    gamma_groups = defaultdict(list)
    for model in models:
        gamma_groups[model["params"]["gamma"]].append(model)

    epsilon_decay_groups = defaultdict(list)
    for model in models:
        epsilon_decay_groups[model["params"]["epsilon_decay"]].append(model)

    alphas, alpha_f1_mean, alpha_f1_std = aggregate_f1(alpha_groups)

    plot_aggregate_f1(alphas, alpha_f1_mean, alpha_f1_std, x_label="Learning Rate α", param_name="α",
                      algorithm=algorithm,
                      filename=plots_directory / f"metrics_alpha_{algorithm}_{phase}_phase.png")

    gamma, gamma_f1_mean, gamma_f1_std = aggregate_f1(gamma_groups)

    plot_aggregate_f1(gamma, gamma_f1_mean, gamma_f1_std, x_label="Gamma", param_name="gamma",
                      algorithm=algorithm,
                      filename=plots_directory / f"metrics_gamma_{algorithm}_{phase}_phase.png")

    epsilon_decay, epsilon_decay_f1_mean, epsilon_decay_f1_std = aggregate_f1(epsilon_decay_groups)

    plot_aggregate_f1(epsilon_decay, epsilon_decay_f1_mean, epsilon_decay_f1_std, x_label="epsilon_decay",
                      param_name="epsilon_decay",
                      algorithm=algorithm,
                      filename=plots_directory / f"metrics_epsilon_decay_{algorithm}_{phase}_phase.png")

    alpha_directory = plots_directory / "alpha_comparison"
    alpha_directory.mkdir(parents=True, exist_ok=True)
    gamma_directory = plots_directory / "gamma_comparison"
    gamma_directory.mkdir(parents=True, exist_ok=True)
    epsilon_decay_directory = plots_directory / "epsilon_decay_comparison"
    epsilon_decay_directory.mkdir(parents=True, exist_ok=True)

    metrics = ["precision", "recall", "f1_score"]

    # Confronto modelli migliori per ogni valore di alpha analizzato
    best_models_per_hyperparameter_comp(alpha_groups, "alpha", metrics, alpha_directory, colors,
                                        algorithm=algorithm, phase=phase+"_phase_")

    # Confronto modelli migliori per ogni valore di gamma analizzato
    best_models_per_hyperparameter_comp(gamma_groups, "gamma", metrics, gamma_directory, colors,
                                        algorithm=algorithm, phase=phase+"_phase_")

    # Confronto modelli migliori per ogni valore di epsilon decay analizzato
    best_models_per_hyperparameter_comp(epsilon_decay_groups, "epsilon_decay", metrics, epsilon_decay_directory,
                                        colors, algorithm=algorithm, phase=phase+"_phase_")


def plot_selected_models(first_phase_path, second_phase_path, baseline_path, algorithm):
    with open(first_phase_path, "r", encoding="utf-8") as f:
        first_phase_models = json.load(f)

    with open(second_phase_path, "r", encoding="utf-8") as f:
        second_phase_models = json.load(f)

    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline_data = json.load(f)
        baseline_model = baseline_data[0]

    colors = plt.cm.tab20.colors

    instable_model = max(first_phase_models, key=lambda d: d["f1_std_last_episodes"])
    best_model = max(second_phase_models, key=lambda d: d["f1_mean_last_episodes"])
    stablest_model = min(second_phase_models, key=lambda d: d["f1_std_last_episodes"])
    if best_model == stablest_model:
        best_model = max(second_phase_models, key=lambda d: d["best_f1_score"])
    worst_model = min(first_phase_models, key=lambda d: d["f1_mean_last_episodes"])

    model_labels = ["baseline_model", "instable_model", "best_model", "worst_model", "stablest_model"]

    model_to_plot = {
        "baseline_model": baseline_model,
        "instable_model": instable_model,
        "best_model": best_model,
        "worst_model": worst_model,
        "stablest_model": stablest_model
    }

    for label, model in model_to_plot.items():
        print(model)

    alg_data = []
    for i, label in enumerate(model_labels):
        model = model_to_plot[label]
        alg_data.append((model, legend_title(model) + " - " + label, colors[i % len(colors)], "-"))

    metrics = ["precision", "recall", "f1_score"]

    titles = [f"Confronto Evoluzione Precision Modelli Selezionati",
              f"Confronto Evoluzione Recall Modelli Selezionati",
              f"Confronto Evoluzione F1-Score Modelli Selezionati"]

    base_directory = Path(__file__).resolve().parent

    plots_directory = base_directory.parent / "plots" / f"{algorithm}_training"
    selected_models_directory = plots_directory / "selected_models_comparison"
    selected_models_directory.mkdir(parents=True, exist_ok=True)

    savepaths = [(selected_models_directory / f"{algorithm}_algorithm_selected_models_{metric}_evolution.png") for metric in metrics]

    plot_metrics(alg_data, metrics, titles, savepaths)


def plot_llm_models(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    models = data

    base_directory = Path(__file__).resolve().parent
    tables_directory = base_directory / "tables"

    plots_directory = base_directory.parent / "plots" / f"rl_reward_model_training"
    plots_directory.mkdir(parents=True, exist_ok=True)

    models.sort(
        key=lambda s: s["f1_mean_last_episodes"],
        reverse=True
    )
    for model in models:
        print(model["sample_number"], model["params"],
              " - F1-Score Max: ", model["best_f1_score"],
              " - F1-Score mean: ", model["f1_mean_last_episodes"],
              " - F1-Score SD: ", model["f1_std_last_episodes"],
              " - Precision: ", model["precision_mean_last_episodes"],
              " - Recall: ", model["recall_mean_last_episodes"])

    colors = plt.cm.tab20.colors
    rows = []

    for model in models:
        row = {
            ("Label Modello", ""): str(model["params"]["model_label"]),
            ("Metriche", "Precisione Media"): model["precision_mean_last_episodes"],
            ("Metriche", "Recall Media"): model["recall_mean_last_episodes"],
            ("Metriche", "F1-Score media"): model["f1_mean_last_episodes"],
            ("Metriche", "F1-Score massima"): model["best_f1_score"],
            ("Metriche", "F1-Score SD"): model["f1_std_last_episodes"]
        }
        rows.append(row)

    data_frame = pd.DataFrame(rows)
    data_frame.columns = pd.MultiIndex.from_tuples(data_frame)

    fig, ax = plt.subplots(figsize=(24, 8))
    ax.axis("off")

    table = ax.table(
        cellText=data_frame.values,
        colLabels=[
            f"{c[0]}\n{c[1]}" if c[1] else c[0]
            for c in data_frame.columns
        ],
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    plt.savefig(tables_directory / f"metrics_table_qlearning_with_llm_reward_model_training.png", dpi=300, bbox_inches="tight")
    plt.close()

    format_dict = {
        ("Metriche", "Precisione Media"): '{:.2f}',
        ("Metriche", "F1-Score media"): '{:.2f}',
        ("Metriche", "F1-Score SD"): '{:.2f}',
        ("Metriche", "Recall Media"): '{:.2f}',
        ("Metriche", "F1-Score massima"): '{:.2f}'
    }

    latex_code = data_frame.style.format(format_dict).hide(axis='index').to_latex()
    print(latex_code)

    alg_data = []
    for i, model in enumerate(models):
        label = model["params"]["model_label"]
        alg_data.append((model, legend_title(model) + " - " + label, colors[i % len(colors)], "-"))

    metrics = ["precision", "recall", "f1_score"]

    titles = [f"Confronto Evoluzione Precision Modelli Q-Learning + LLM Reward Model",
              f"Confronto Evoluzione Recall Modelli Q-Learning + LLM Reward Model",
              f"Confronto Evoluzione F1-Score Modelli Q-Learning + LLM Reward Model"]

    savepaths = [(plots_directory / f"q_learning_with_llm_reward_model_{metric}_evolution.png") for
                 metric in metrics]

    plot_metrics(alg_data, metrics, titles, savepaths)


if __name__ == "__main__":
    """model_plot_comparison("training_metrics/metrics_qlearning_first_phase.json", algorithm="QLearning",
                          phase="first")"""
    """model_plot_comparison("training_metrics/metrics_qlearning_second_phase.json", algorithm="QLearning",
                          phase="second")"""
    """plot_selected_models("training_metrics/metrics_qlearning_first_phase.json",
                         "training_metrics/metrics_qlearning_second_phase.json",
                         "training_metrics/metrics_qlearning_baseline.json", algorithm="QLearning")"""
    """model_plot_comparison("training_metrics/metrics_sarsa_first_phase.json", algorithm="SARSA",
                          phase="first")"""
    model_plot_comparison("training_metrics/metrics_sarsa_second_phase.json", algorithm="SARSA",
                          phase="second")
    """plot_selected_models("training_metrics/metrics_sarsa_first_phase.json",
                         "training_metrics/metrics_sarsa_second_phase.json",
                         "training_metrics/metrics_sarsa_baseline.json", algorithm="SARSA")"""
    """plot_llm_models("training_metrics/metrics_qlearning_with_llm_reward_model.json")"""
