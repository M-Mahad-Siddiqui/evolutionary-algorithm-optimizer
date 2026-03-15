"""
Main entry point for the modular EA assignment project.
"""

import os
import csv
import matplotlib.pyplot as plt

from problems import PROBLEMS
from selection import survival_select
from ea_core import (
    initialize_population,
    evaluate_population,
    best_fitness,
    average_fitness,
    worst_fitness,
    generate_offspring,
)
from plotting import plot_metric, plot_final_generation_boxplot


# ------------------------------------------------------------
# EA parameters
# ------------------------------------------------------------

POPULATION_SIZE = 10
OFFSPRING_SIZE = 10
GENERATIONS = 40
RUNS = 10
MUTATION_STEP = 0.25
MUTATION_RATE_PER_GENE = 0.20
PLOTS_OUTPUT_DIR = "plots"
RESULTS_OUTPUT_DIR = "results"


def safe_output_stem(problem_name, combo_name):
    """Build a filesystem-friendly base name for one experiment combination."""
    text = f"{problem_name}_{combo_name}"
    text = text.replace(" + ", "_")
    text = text.replace(" ", "_")

    cleaned = ""
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
    for ch in text:
        if ch in allowed:
            cleaned += ch

    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")

    cleaned = cleaned.strip("_")
    if cleaned == "":
        cleaned = "output"
    return cleaned


def export_combination_results_csv(
    problem_name,
    combo_name,
    all_bsf_runs,
    avg_bsf_curve,
    all_acp_runs,
    avg_acp_curve,
):
    """
    Export one CSV table for a function/selection combination.

    Header format:
    Generation, Run 1 BSF, ..., Run 10 BSF, Average BSF,
    Run 1 ACP, ..., Run 10 ACP, Average ACP
    """
    csv_filename = f"{safe_output_stem(problem_name, combo_name)}_data.csv"
    csv_path = os.path.join(RESULTS_OUTPUT_DIR, csv_filename)

    header = ["Generation"]

    for run_index in range(1, RUNS + 1):
        header.append(f"Run {run_index} BSF")
    header.append("Average BSF")

    for run_index in range(1, RUNS + 1):
        header.append(f"Run {run_index} ACP")
    header.append("Average ACP")

    def format_value(value):
        # Fixed decimals make the table easier to read and compare.
        return f"{value:.6f}"

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)

        for generation_index in range(GENERATIONS):
            row = [generation_index + 1]

            for run_index in range(RUNS):
                row.append(format_value(all_bsf_runs[run_index][generation_index]))
            row.append(format_value(avg_bsf_curve[generation_index]))

            for run_index in range(RUNS):
                row.append(format_value(all_acp_runs[run_index][generation_index]))
            row.append(format_value(avg_acp_curve[generation_index]))

            writer.writerow(row)

    print(f"Saved CSV: {csv_path}")


def plot_individual_metric(
    problem_name,
    combo_name,
    generation_numbers,
    metric_curve,
    metric_name,
    y_label,
):
    """Save one single-combination metric line plot with explicit filename."""
    plt.figure(figsize=(10, 6))
    plt.plot(generation_numbers, metric_curve, label=combo_name)
    plt.title(f"{problem_name} - {combo_name} - Average {metric_name} Over Generations")
    plt.xlabel("Generation")
    plt.ylabel(y_label)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    filename = f"{safe_output_stem(problem_name, combo_name)}_{metric_name}.png"
    output_path = os.path.join(PLOTS_OUTPUT_DIR, filename)
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot: {output_path}")


# ------------------------------------------------------------
# One EA run and repeated experiments
# ------------------------------------------------------------

def run_single_ea_attempt(
    fitness_function,
    x_bounds,
    y_bounds,
    parent_selection_method,
    survival_selection_method,
):
    """
    Run one EA attempt for GENERATIONS generations.

    Returns two lists (length = GENERATIONS):
    - bsf_per_generation (best-so-far)
    - acp_per_generation (average current population)
    - wcp_per_generation (worst current population)
    """
    population = initialize_population(POPULATION_SIZE, x_bounds, y_bounds)
    population = evaluate_population(population, fitness_function)

    bsf_per_generation = []
    acp_per_generation = []
    wcp_per_generation = []

    # Keep track of best value seen in this run across generations
    current_best_so_far = None

    for _ in range(GENERATIONS):
        # Generate offspring and evaluate them
        children = generate_offspring(
            population,
            parent_selection_method,
            x_bounds,
            y_bounds,
            OFFSPRING_SIZE,
            MUTATION_STEP,
            MUTATION_RATE_PER_GENE,
        )
        evaluated_children = evaluate_population(children, fitness_function)

        # Combine parent + offspring, then select survivors
        combined = []
        for individual in population:
            combined.append(individual)
        for individual in evaluated_children:
            combined.append(individual)

        population = survival_select(
            combined,
            survival_selection_method,
            POPULATION_SIZE,
        )

        # Record metrics after survival selection, as required by the slide.
        current_acp = average_fitness(population)
        acp_per_generation.append(current_acp)

        generation_best = best_fitness(population)
        if current_best_so_far is None or generation_best < current_best_so_far:
            current_best_so_far = generation_best
        bsf_per_generation.append(current_best_so_far)

        current_wcp = worst_fitness(population)
        wcp_per_generation.append(current_wcp)

    return bsf_per_generation, acp_per_generation, wcp_per_generation


def average_curves_over_runs(all_runs_curves):
    """
    all_runs_curves is a list of lists:
    - outer length: RUNS
    - inner length: GENERATIONS

    Return one list of generation-wise averages.
    """
    averages = []

    for generation_index in range(GENERATIONS):
        total = 0.0
        for run_index in range(RUNS):
            total += all_runs_curves[run_index][generation_index]
        averages.append(total / RUNS)

    return averages


def run_experiment_for_combination(
    problem_name,
    combo_name,
    fitness_function,
    x_bounds,
    y_bounds,
    parent_selection_method,
    survival_selection_method,
):
    """
    Run RUNS independent attempts for one combination of
    parent/survival selection methods.

    Return:
    - avg_bsf_curve (length GENERATIONS)
    - avg_acp_curve (length GENERATIONS)
    - avg_wcp_curve (length GENERATIONS)
    - all_bsf_runs (RUNS x GENERATIONS)
    - all_acp_runs (RUNS x GENERATIONS)
    - final_gen_bsf_values (length RUNS)
    - final_gen_acp_values (length RUNS)
    - final_gen_wcp_values (length RUNS)
    """
    all_bsf_runs = []
    all_acp_runs = []
    all_wcp_runs = []
    final_gen_bsf_values = []
    final_gen_acp_values = []
    final_gen_wcp_values = []

    for _ in range(RUNS):
        bsf_curve, acp_curve, wcp_curve = run_single_ea_attempt(
            fitness_function,
            x_bounds,
            y_bounds,
            parent_selection_method,
            survival_selection_method,
        )
        all_bsf_runs.append(bsf_curve)
        all_acp_runs.append(acp_curve)
        all_wcp_runs.append(wcp_curve)

        final_gen_bsf_values.append(bsf_curve[-1])
        final_gen_acp_values.append(acp_curve[-1])
        final_gen_wcp_values.append(wcp_curve[-1])

    avg_bsf_curve = average_curves_over_runs(all_bsf_runs)
    avg_acp_curve = average_curves_over_runs(all_acp_runs)
    avg_wcp_curve = average_curves_over_runs(all_wcp_runs)

    export_combination_results_csv(
        problem_name,
        combo_name,
        all_bsf_runs,
        avg_bsf_curve,
        all_acp_runs,
        avg_acp_curve,
    )

    return (
        avg_bsf_curve,
        avg_acp_curve,
        avg_wcp_curve,
        all_bsf_runs,
        all_acp_runs,
        final_gen_bsf_values,
        final_gen_acp_values,
        final_gen_wcp_values,
    )


# ------------------------------------------------------------
# Experiment orchestration
# ------------------------------------------------------------

def run_all_experiments_for_problem(problem_name, problem_data):
    """
    Run all 6 required selection combinations for one function,
    then create plots.
    """
    fitness_function = problem_data["fitness_function"]
    x_bounds = problem_data["x_bounds"]
    y_bounds = problem_data["y_bounds"]

    combinations = [
        ("FPS", "Truncation"),
        ("RBS", "Truncation"),
        ("Binary Tournament", "Truncation"),
        ("FPS", "Binary Tournament"),
        ("RBS", "Binary Tournament"),
        ("Binary Tournament", "Binary Tournament"),
    ]

    average_bsf_by_combination = {}
    average_acp_by_combination = {}
    average_wcp_by_combination = {}
    final_bsf_by_combination = {}
    final_acp_by_combination = {}
    final_wcp_by_combination = {}
    generations = []
    for i in range(1, GENERATIONS + 1):
        generations.append(i)

    for parent_method, survival_method in combinations:
        combo_name = f"{parent_method} + {survival_method}"
        print(f"Running {problem_name}: {combo_name}")

        (
            avg_bsf,
            avg_acp,
            avg_wcp,
            all_bsf_runs,
            all_acp_runs,
            final_bsf_values,
            final_acp_values,
            final_wcp_values,
        ) = run_experiment_for_combination(
            problem_name,
            combo_name,
            fitness_function,
            x_bounds,
            y_bounds,
            parent_method,
            survival_method,
        )

        average_bsf_by_combination[combo_name] = avg_bsf
        average_acp_by_combination[combo_name] = avg_acp
        average_wcp_by_combination[combo_name] = avg_wcp
        final_bsf_by_combination[combo_name] = final_bsf_values
        final_acp_by_combination[combo_name] = final_acp_values
        final_wcp_by_combination[combo_name] = final_wcp_values

        plot_individual_metric(
            problem_name,
            combo_name,
            generations,
            avg_bsf,
            "BSF",
            "Average Best-So-Far Fitness",
        )

        plot_individual_metric(
            problem_name,
            combo_name,
            generations,
            avg_acp,
            "ACP",
            "Average Population Fitness",
        )

    plot_metric(
        generations,
        average_bsf_by_combination,
        f"{problem_name} - Average BSF Over Generations",
        "Average Best-So-Far Fitness",
        PLOTS_OUTPUT_DIR,
    )

    plot_metric(
        generations,
        average_acp_by_combination,
        f"{problem_name} - Average ACP Over Generations",
        "Average Population Fitness",
        PLOTS_OUTPUT_DIR,
    )

    plot_metric(
        generations,
        average_wcp_by_combination,
        f"{problem_name} - Average WCP Over Generations",
        "Average Worst Population Fitness",
        PLOTS_OUTPUT_DIR,
    )

    plot_final_generation_boxplot(
        problem_name,
        "BSF",
        final_bsf_by_combination,
        PLOTS_OUTPUT_DIR,
    )

    plot_final_generation_boxplot(
        problem_name,
        "ACP",
        final_acp_by_combination,
        PLOTS_OUTPUT_DIR,
    )

    plot_final_generation_boxplot(
        problem_name,
        "WCP",
        final_wcp_by_combination,
        PLOTS_OUTPUT_DIR,
    )


# ------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------

def main():
    # Optional seed for reproducibility. Comment this out for full randomness.
    # random.seed(42)

    os.makedirs(PLOTS_OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_OUTPUT_DIR, exist_ok=True)

    for problem_name in PROBLEMS:
        run_all_experiments_for_problem(problem_name, PROBLEMS[problem_name])

    plt.show()


if __name__ == "__main__":
    main()
