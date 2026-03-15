"""
EA core helpers for population management and genetic operators.
"""

import random

from selection import select_parent


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


def mutate(individual, x_bounds, y_bounds, mutation_step, mutation_rate_per_gene):
    """
    Mutate each gene with a probability.
    If mutation happens, add or subtract mutation_step.
    """
    child = {"x": individual["x"], "y": individual["y"]}

    # Mutate x with probability
    if random.random() < mutation_rate_per_gene:
        if random.random() < 0.5:
            child["x"] = child["x"] + mutation_step
        else:
            child["x"] = child["x"] - mutation_step

    # Mutate y with probability
    if random.random() < mutation_rate_per_gene:
        if random.random() < 0.5:
            child["y"] = child["y"] + mutation_step
        else:
            child["y"] = child["y"] - mutation_step

    # Keep genes inside allowed bounds
    child["x"] = clip(child["x"], x_bounds[0], x_bounds[1])
    child["y"] = clip(child["y"], y_bounds[0], y_bounds[1])

    return child


def generate_offspring(
    population,
    parent_selection_method,
    x_bounds,
    y_bounds,
    offspring_size,
    mutation_step,
    mutation_rate_per_gene,
):
    """
    Generate exactly offspring_size children using selection,
    crossover, and mutation.
    """
    offspring = []

    while len(offspring) < offspring_size:
        parent_1 = select_parent(population, parent_selection_method)
        parent_2 = select_parent(population, parent_selection_method)

        child_1, child_2 = crossover(parent_1, parent_2)

        child_1 = mutate(child_1, x_bounds, y_bounds, mutation_step, mutation_rate_per_gene)
        child_2 = mutate(child_2, x_bounds, y_bounds, mutation_step, mutation_rate_per_gene)

        offspring.append(child_1)
        if len(offspring) < offspring_size:
            offspring.append(child_2)

    return offspring
