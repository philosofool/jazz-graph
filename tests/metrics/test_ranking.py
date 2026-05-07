from jazz_graph.metrics.ranking import map_at_k, precision


import numpy as np


def test_precision():
    ranking = np.array([1, 5, 3, 4])
    relevant_items = np.array([3, 5])
    assert precision(ranking, relevant_items) == .5


def test_map_at_k():
    ranking = np.array([5, 4, 3, 1, 2, 6])
    relevant_items = np.array([5, 1, 2])
    result = map_at_k(ranking, relevant_items, 6)
    assert np.isclose(result, .7)

    ranking = np.array([4, 3, 6])
    relevant_items = np.array([5, 1, 2])
    result = map_at_k(ranking, relevant_items, 3)
    assert np.isclose(result, .0)

    # test k > len(ranking)
    ranking = np.array([4, 3, 6])
    relevant_items = np.array([5, 1, 2])
    result = map_at_k(ranking, relevant_items, 4)
    assert np.isclose(result, .0)