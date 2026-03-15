"""
Selection operators for the EA assignment.
Includes parent selection and survival selection.
"""

import random


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
