import numpy as np
from jazz_graph.etl.transforms import map_array, map_by_index

def test_map_array():
    arr = np.array([5, 1, 3])
    mapping = {5: 0, 1: 1, 3: 2}
    result = map_array(arr, mapping)
    np.testing.assert_array_equal(result, np.array([0, 1, 2]))

    arr = np.array([4, 1])
    result = map_array(arr, mapping)
    assert np.isnan(result[0])
    assert result[1] == 1

def test_map_by_index():
    arr = np.array([5, 1, 3])
    result = map_by_index(arr)
    expected = {5: 0, 1: 1, 3: 2}
    assert result == expected
