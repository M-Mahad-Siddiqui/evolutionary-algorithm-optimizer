"""
Plotting helpers for the EA assignment.
"""

import os
import matplotlib.pyplot as plt


def safe_plot_filename(text):
    """Convert a title-like text into a filesystem-friendly filename."""
    filename = text.lower()
    filename = filename.replace(" ", "_")
    filename = filename.replace("+", "plus")
    filename = filename.replace("-", "_")

    cleaned = ""
    allowed = "abcdefghijklmnopqrstuvwxyz0123456789_"
    for ch in filename:
        if ch in allowed:
            cleaned += ch

    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")

    cleaned = cleaned.strip("_")
    if cleaned == "":
        cleaned = "plot"

    return cleaned + ".png"


def plot_metric(generation_numbers, metric_curves, title, y_label, save_directory):
    """
    Plot one graph where each curve is one selection-method combination.
    """
    plt.figure(figsize=(10, 6))

    for combo_name in metric_curves:
        curve = metric_curves[combo_name]
        plt.plot(generation_numbers, curve, label=combo_name)

    plt.title(title)
    plt.xlabel("Generation")
    plt.ylabel(y_label)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Save each plot as PNG so results can be submitted or reused later.
    file_name = safe_plot_filename(title)
    output_path = os.path.join(save_directory, file_name)
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot: {output_path}")


def plot_final_generation_boxplot(problem_name, metric_name, data_by_combination, save_directory):
    """
    Bonus figure: boxplot of generation-40 values across the 10 runs
    for each selection-method combination.
    """
    labels = []
    values = []
    for combo_name in data_by_combination:
        labels.append(combo_name)
        values.append(data_by_combination[combo_name])

    plt.figure(figsize=(12, 6))
    plt.boxplot(values, tick_labels=labels)
    plt.title(f"{problem_name} - Final Generation ({metric_name}) Distribution")
    plt.ylabel(metric_name)
    plt.xticks(rotation=20, ha="right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    file_name = safe_plot_filename(f"{problem_name} final generation {metric_name} boxplot")
    output_path = os.path.join(save_directory, file_name)
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot: {output_path}")
