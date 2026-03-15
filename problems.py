"""
Problem definitions for the EA assignment.
"""


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
