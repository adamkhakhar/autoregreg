import numpy as np


def sin_small(x: float) -> float:
    return np.sin(2 * np.pi * x) + 1


def sin_large(x: float) -> float:
    return 5 * 10**6 * (np.sin(2 * np.pi * x) + 1) + 5 * 10**6


def log_small(x: float) -> float:
    return 2 * (np.log(x + 0.4) + 1)


def log_large(x: float) -> float:
    return 5 * 10**6 * (np.log(x + 0.4) + 1) + 5 * 10**6
