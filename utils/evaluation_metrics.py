import numpy as np


def r_sq(pred: np.ndarray, targ: np.ndarray) -> float:
    return 1 - (np.square(pred - targ)).sum() / np.var(targ)
