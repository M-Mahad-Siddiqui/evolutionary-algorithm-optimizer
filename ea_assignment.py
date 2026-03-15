"""
Evolutionary Algorithm (EA) Assignment Script

This script runs a simple and readable EA for two optimization functions.
It demonstrates the standard EA cycle:
1) Initialization
2) Fitness evaluation
3) Parent selection
4) Crossover + mutation
5) Survival selection

The script repeats experiments for 6 combinations of selection methods,
tracks BSF and ACP over generations, averages over runs, and plots results.
"""

import random
import os
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Problem definitions
# ------------------------------------------------------------

def function_1(x, y):
    """
    Function 1:
        f(x, y) = x^2 + y^2
    Bounds:
        -5 < x, y < 5
    """
    return (x ** 2) + (y ** 2)


def function_2(x, y):
    """
    Function 2:
        f(x, y) = 100 * (x^2 - y)^2 + (1 - x)^2
    Bounds:
        -2 < x < 2, -1 < y < 3
    """
    return 100 * ((x ** 2) - y) ** 2 + (1 - x) ** 2


PROBLEMS = {
    "Function 1": {
        "fitness_function": function_1,
        "x_bounds": (-5.0, 5.0),
        "y_bounds": (-5.0, 5.0),
    },
    "Function 2": {
        "fitness_function": function_2,
        "x_bounds": (-2.0, 2.0),
        "y_bounds": (-1.0, 3.0),
    },
}


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


# ------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------

def random_gene(lower_bound, upper_bound):
    """Generate one random float inside [lower_bound, upper_bound]."""
    return random.uniform(lower_bound, upper_bound)


def clip(value, lower_bound, upper_bound):
    """Keep value inside the allowed range."""
    if value < lower_bound:
        return lower_bound
    if value > upper_bound:
        return upper_bound
    return value


def create_individual(x_bounds, y_bounds):
    """Create one individual represented as a dictionary with x and y genes."""
    x = random_gene(x_bounds[0], x_bounds[1])
    y = random_gene(y_bounds[0], y_bounds[1])
    return {"x": x, "y": y}


def initialize_population(pop_size, x_bounds, y_bounds):
    """Create the starting population with random individuals."""
    population = []
    for _ in range(pop_size):
        individual = create_individual(x_bounds, y_bounds)
        population.append(individual)
    return population


def evaluate_population(population, fitness_function):
    """
    Return a new list where each individual has a fitness value attached.
    Smaller fitness is better (minimization problem).
    """
    evaluated = []
    for individual in population:
        x_value = individual["x"]
        y_value = individual["y"]
        fitness = fitness_function(x_value, y_value)

        evaluated_individual = {
            "x": x_value,
            "y": y_value,
            "fitness": fitness,
        }
        evaluated.append(evaluated_individual)

    return evaluated


def best_fitness(population):
    """Get the best (minimum) fitness in the current population."""
    best_value = population[0]["fitness"]
    for individual in population:
        if individual["fitness"] < best_value:
            best_value = individual["fitness"]
    return best_value


def average_fitness(population):
    """Get the average fitness in the current population."""
    total = 0.0
    for individual in population:
        total += individual["fitness"]
    return total / len(population)


def worst_fitness(population):
    """Get the worst (maximum) fitness in the current population."""
    worst_value = population[0]["fitness"]
    for individual in population:
        if individual["fitness"] > worst_value:
            worst_value = individual["fitness"]
    return worst_value


# ------------------------------------------------------------
# Parent selection methods
# ------------------------------------------------------------

def fps_select_one(population):
    """
    Fitness Proportionate Selection (roulette wheel) for minimization.

    Because lower fitness is better, convert each fitness to a positive
    selection weight using:
        weight = 1 / (1 + fitness)
    """
    weights = []
    for individual in population:
        weight = 1.0 / (1.0 + individual["fitness"])
        weights.append(weight)

    total_weight = sum(weights)
    pick = random.uniform(0, total_weight)

    running_sum = 0.0
    for index, individual in enumerate(population):
        running_sum += weights[index]
        if running_sum >= pick:
            return individual

    return population[-1]


def rbs_select_one(population):
    """
    Rank-Based Selection.

    Sort population by fitness (best to worst), then assign rank weights:
    best gets highest weight, worst gets lowest weight.
    """
    ranked = sorted(population, key=lambda ind: ind["fitness"])
    n = len(ranked)

    rank_weights = []
    for i in range(n):
        # Best individual at i=0 gets weight n, next gets n-1, ..., worst gets 1
        rank_weights.append(n - i)

    total_weight = sum(rank_weights)
    pick = random.uniform(0, total_weight)

    running_sum = 0
    for i, individual in enumerate(ranked):
        running_sum += rank_weights[i]
        if running_sum >= pick:
            return individual

    return ranked[-1]


def tournament_select_one(population):
    """
    Binary tournament selection.
    Randomly pick two individuals and return the one with better fitness.
    """
    first = random.choice(population)
    second = random.choice(population)

    if first["fitness"] <= second["fitness"]:
        return first
    return second


def select_parent(population, method_name):
    """Dispatch helper for parent selection method."""
    if method_name == "FPS":
        return fps_select_one(population)
    if method_name == "RBS":
        return rbs_select_one(population)
    if method_name == "Binary Tournament":
        return tournament_select_one(population)

    raise ValueError(f"Unknown parent selection method: {method_name}")


# ------------------------------------------------------------
# Variation operators (crossover + mutation)
# ------------------------------------------------------------

def crossover(parent_a, parent_b):
    """
    Arithmetic crossover for real-valued genes.

    We create two children by mixing parent genes using random alpha.
    """
    alpha = random.random()

    child_1 = {
        "x": alpha * parent_a["x"] + (1 - alpha) * parent_b["x"],
        "y": alpha * parent_a["y"] + (1 - alpha) * parent_b["y"],
    }

    child_2 = {
        "x": (1 - alpha) * parent_a["x"] + alpha * parent_b["x"],
        "y": (1 - alpha) * parent_a["y"] + alpha * parent_b["y"],
    }

    return child_1, child_2


def mutate(individual, x_bounds, y_bounds):
    """
    Mutate each gene with a probability.
    If mutation happens, add or subtract 0.25.
    """
    child = {"x": individual["x"], "y": individual["y"]}

    # Mutate x with probability
    if random.random() < MUTATION_RATE_PER_GENE:
        if random.random() < 0.5:
            child["x"] = child["x"] + MUTATION_STEP
        else:
            child["x"] = child["x"] - MUTATION_STEP

    # Mutate y with probability
    if random.random() < MUTATION_RATE_PER_GENE:
        if random.random() < 0.5:
            child["y"] = child["y"] + MUTATION_STEP
        else:
            child["y"] = child["y"] - MUTATION_STEP

    # Keep genes inside allowed bounds
    child["x"] = clip(child["x"], x_bounds[0], x_bounds[1])
    child["y"] = clip(child["y"], y_bounds[0], y_bounds[1])

    return child


def generate_offspring(population, parent_selection_method, x_bounds, y_bounds):
    """
    Generate exactly OFFSPRING_SIZE children using selection,
    crossover, and mutation.
    """
    offspring = []

    while len(offspring) < OFFSPRING_SIZE:
        parent_1 = select_parent(population, parent_selection_method)
        parent_2 = select_parent(population, parent_selection_method)

        child_1, child_2 = crossover(parent_1, parent_2)

        child_1 = mutate(child_1, x_bounds, y_bounds)
        child_2 = mutate(child_2, x_bounds, y_bounds)

        offspring.append(child_1)
        if len(offspring) < OFFSPRING_SIZE:
            offspring.append(child_2)

    return offspring


# ------------------------------------------------------------
# Survival selection methods
# ------------------------------------------------------------

def truncation_survival(combined_population, survivors_count):
    """Pick best survivors_count individuals by fitness."""
    sorted_population = sorted(combined_population, key=lambda ind: ind["fitness"])
    return sorted_population[:survivors_count]


def binary_tournament_survival(combined_population, survivors_count):
    """
    Pick survivors by repeated binary tournaments from the combined pool.

    Winner is removed from pool so each survivor is unique.
    """
    pool = []
    for individual in combined_population:
        pool.append(individual)

    survivors = []

    while len(survivors) < survivors_count:
        if len(pool) == 1:
            survivors.append(pool.pop())
            continue

        idx_a = random.randrange(len(pool))
        idx_b = random.randrange(len(pool))

        while idx_b == idx_a:
            idx_b = random.randrange(len(pool))

        candidate_a = pool[idx_a]
        candidate_b = pool[idx_b]

        if candidate_a["fitness"] <= candidate_b["fitness"]:
            winner_index = idx_a
        else:
            winner_index = idx_b

        winner = pool.pop(winner_index)
        survivors.append(winner)

    return survivors


def survival_select(combined_population, method_name, survivors_count):
    """Dispatch helper for survival selection method."""
    if method_name == "Truncation":
        return truncation_survival(combined_population, survivors_count)
    if method_name == "Binary Tournament":
        return binary_tournament_survival(combined_population, survivors_count)

    raise ValueError(f"Unknown survival selection method: {method_name}")


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

    return (
        avg_bsf_curve,
        avg_acp_curve,
        avg_wcp_curve,
        final_gen_bsf_values,
        final_gen_acp_values,
        final_gen_wcp_values,
    )


# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------

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


def run_all_experiments_for_problem(problem_name, problem_data):
    """
    Run all 6 required selection combinations for one function,
    then create two plots:
    1) Average BSF vs generation
    2) Average ACP vs generation
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

    for parent_method, survival_method in combinations:
        combo_name = f"{parent_method} + {survival_method}"
        print(f"Running {problem_name}: {combo_name}")

        (
            avg_bsf,
            avg_acp,
            avg_wcp,
            final_bsf_values,
            final_acp_values,
            final_wcp_values,
        ) = run_experiment_for_combination(
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

    generations = []
    for i in range(1, GENERATIONS + 1):
        generations.append(i)

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

    for problem_name in PROBLEMS:
        run_all_experiments_for_problem(problem_name, PROBLEMS[problem_name])

    plt.show()


if __name__ == "__main__":
    main()
