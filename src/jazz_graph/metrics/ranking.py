import numpy as np

from tests.metrics.test_ranking import test_precision
from tests.metrics.test_ranking import test_map_at_k

def map_at_k(ranking: np.ndarray, relevant_items: np.ndarray, k: int) -> float:
    mean_precision = 0
    num_relevant = 0
    for i in range(k):
        if i == ranking.size:
            break
        if ranking[i] in relevant_items:
            num_relevant += 1
            mean_precision += precision(ranking[:i + 1], relevant_items)
    if num_relevant == 0:
        return 0.
    return mean_precision / num_relevant

def precision(ranking, relevant_items):
    return np.intersect1d(ranking, relevant_items).size / ranking.size
